"""Segmenter en régions toutes les pages du dépôt TLG, en vue de l'océrisation.

Un processus tient un GPU et fait tourner le modèle de mise en page ; le rendu
des pages, qui est du calcul CPU, part dans un pool de processus, et le volume
suivant se télécharge pendant que le courant est segmenté. Le GPU n'attend donc
ni le disque ni le réseau.

Les volumes viennent de l'arborescence du dépôt, pas du manifeste : celui-ci
est régénéré à part et retarde sur ce qui est réellement publié. Plusieurs
exemplaires se partagent le travail par revendication atomique ; voir lancer.sh.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import shutil
import sys
import tarfile
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DEPOT = "Zual/TLG_libre_scans"
ARBRE = f"https://huggingface.co/api/datasets/{DEPOT}/tree/main"
BRUT = f"https://huggingface.co/datasets/{DEPOT}/resolve/main"
RACINES = os.environ.get("RACINES", "source_pdfs,webdataset").split(",")
# Huit workers interrogeant l'API en même temps valent un 429 : l'inventaire
# se construit une fois et se relit depuis ce fichier.
CACHE = Path(os.environ.get("INVENTAIRE", "")) if os.environ.get("INVENTAIRE") else None
SOURCES = Path.home() / "tlg-sources"
UTILES = {"Text", "TextInlineMath", "ListItem", "SectionHeader", "Title",
          "Caption", "Formula"}
INTITULE = {"SectionHeader", "Title"}
IMAGES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2", ".j2k")
EMBALLAGES = (".pdf", ".tar", ".tar.gz", ".tgz", ".zip")

# Le modèle ramène toute page à 768×768 : au-delà de 150 dpi on paie du rendu
# pour rien. Mesuré sur un volume, 100, 150 et 300 dpi rendent les mêmes
# régions ; les boîtes restent reportables ailleurs grâce à « resolution ».
RESOLUTION = int(os.environ.get("RESOLUTION", "150"))
# Les formats de page varient énormément d'une source à l'autre : à 150 dpi
# fixe, un Gallica sort en 7201×9744, que surya redécoupe en tuiles. Borner le
# côté long à 1400 px rend les mêmes régions (mesuré : 81 des deux côtés) pour
# 7,6× moins de rendu et 1,4× moins de GPU.
COTE_MAX = int(os.environ.get("COTE_MAX", "1400"))
LOT = int(os.environ.get("LOT", "32"))
RENDUS = int(os.environ.get("RENDUS", "2"))
QUALITE = 92


def journal(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{os.environ.get('PART', '0')}] {message}",
          flush=True)


def inventaire() -> list[tuple[str, int]]:
    """Ce que le dépôt contient réellement, le manifeste étant en retard.

    Les volumes reviennent du plus gros au plus petit : avec une file de
    travail, servir les longs d'abord raccourcit la traîne de fin.
    """
    if CACHE and CACHE.is_file():
        return [(c, o) for c, o in json.loads(CACHE.read_text(encoding="utf-8"))]
    trouves: dict[str, int] = {}
    for racine in RACINES:
        curseur = None
        while True:
            url = f"{ARBRE}/{racine}?recursive=true&expand=true"
            if curseur:
                url += "&cursor=" + urllib.parse.quote(curseur)
            with urllib.request.urlopen(url, timeout=180) as flux:
                lot = json.loads(flux.read().decode("utf-8"))
                lien = flux.headers.get("Link") or ""
            if not lot:
                break
            for entree in lot:
                chemin = entree.get("path") or ""
                if entree.get("type") == "file" and chemin.lower().endswith(EMBALLAGES):
                    trouves[chemin] = entree.get("size") or 0
            suite = re.search(r"cursor=([^&>;]+)", lien)
            if not suite:
                break
            curseur = urllib.parse.unquote(suite.group(1))
    liste = sorted(trouves.items(), key=lambda c: -c[1])
    if CACHE:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(liste), encoding="utf-8")
    return liste


def rapatrier(chemin_distant: str) -> Path | None:
    """Tirer un volume du dépôt, une seule fois."""
    local = SOURCES / chemin_distant.replace("/", "__")
    if local.is_file() and local.stat().st_size > 0:
        return local
    SOURCES.mkdir(parents=True, exist_ok=True)
    url = f"{BRUT}/{urllib.parse.quote(chemin_distant)}"
    partiel = local.with_suffix(local.suffix + f".part{os.getpid()}")
    # Le dépôt limite le débit quand huit workers tirent de front : patienter
    # vaut mieux que renoncer au volume.
    for essai in range(5):
        try:
            with urllib.request.urlopen(url, timeout=600) as flux, \
                    partiel.open("wb") as sortie:
                shutil.copyfileobj(flux, sortie, 1 << 20)
            partiel.rename(local)
            return local
        except Exception as souci:  # noqa: BLE001
            partiel.unlink(missing_ok=True)
            dernier = souci
            time.sleep(min(60, 4 ** essai))
    journal(f"  ✗ téléchargement {chemin_distant[:52]} : {type(dernier).__name__}")
    return None


# --- rendu, exécuté dans le pool -------------------------------------------

_documents: dict[str, object] = {}


def _document(chemin: str):
    """Garder le PDF ouvert d'un appel à l'autre : l'ouvrir coûte plus que rendre.

    Un seul à la fois : sur NFS, un fichier effacé alors qu'il reste ouvert
    survit en .nfsXXXX, et le corpus entier resterait donc sur le disque.
    """
    doc = _documents.get(chemin)
    if doc is None:
        for ouvert in _documents.values():
            ouvert.close()
        _documents.clear()
        import pypdfium2 as pdfium
        doc = _documents[chemin] = pdfium.PdfDocument(chemin)
    return doc


def rendre(tache: tuple[str, int]) -> tuple[int, float, bytes] | None:
    """Rendre une page en JPEG. Le JPEG tient le tuyau du pool : une page brute
    pèse 6 Mo, comprimée 300 Ko, et le scan d'origine est déjà du JPEG."""
    chemin, page = tache
    try:
        feuille = _document(chemin)[page - 1]
        cote = max(feuille.get_size()) or 1.0
        echelle = min(RESOLUTION / 72, COTE_MAX / cote)
        image = feuille.render(scale=echelle).to_pil().convert("RGB")
    except Exception:  # noqa: BLE001
        return None
    tampon = io.BytesIO()
    image.save(tampon, "JPEG", quality=QUALITE)
    return page, echelle * 72, tampon.getvalue()


def pages_du_pdf(volume: Path) -> int:
    import pypdfium2 as pdfium
    document = pdfium.PdfDocument(str(volume))
    total = len(document)
    document.close()
    return total


def pages_de_l_archive(volume: Path) -> list[str]:
    nom = volume.name.lower()
    if nom.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(volume) as archive:
            return sorted(m.name for m in archive.getmembers()
                          if m.isfile() and m.name.lower().endswith(IMAGES))
    with zipfile.ZipFile(volume) as archive:
        return sorted(n for n in archive.namelist()
                      if n.lower().endswith(IMAGES))


def image_de_l_archive(volume: Path, membre: str):
    from PIL import Image
    nom = volume.name.lower()
    if nom.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(volume) as archive:
            donnees = archive.extractfile(membre).read()
    else:
        with zipfile.ZipFile(volume) as archive:
            donnees = archive.read(membre)
    try:
        return Image.open(io.BytesIO(donnees)).convert("RGB")
    except Exception:  # noqa: BLE001
        # L'openjpeg embarqué dans Pillow refuse certains JPEG2000 dont le
        # codestream est pourtant complet (marqueur EOC présent) ; imagecodecs
        # les décode sans broncher. Une archive entière en dépendait.
        import imagecodecs
        return Image.fromarray(imagecodecs.jpeg2k_decode(donnees)).convert("RGB")


# --- segmentation ----------------------------------------------------------


def decrire(resultat, image, complement: dict) -> dict:
    """Mettre une page segmentée sous la forme que l'océrisation attendra."""
    regions = []
    for boite in sorted(resultat.bboxes, key=lambda b: getattr(b, "position", 0)):
        nature = str(getattr(boite, "label", "") or "")
        x0, y0, x1, y1 = (int(v) for v in boite.bbox)
        regions.append({
            "nature": nature, "boite": [x0, y0, x1, y1],
            "ordre": int(getattr(boite, "position", 0) or 0),
            "utile": nature in UTILES,
            "intitule": nature in INTITULE})
    # Les boîtes sont en pixels de l'image donnée au modèle : sans sa taille,
    # elles ne se reportent pas sur une image océrisée autrement. Une page
    # rendue deux fois plus grande veut des boîtes deux fois plus grandes.
    return {"taille": [image.width, image.height], **complement,
            "regions": regions}


def segmenter(predicteur, lot, dossier: Path) -> tuple[int, int]:
    """Segmenter un paquet de pages ; en cas d'échec, retomber page à page."""
    images = [image for _, image, _ in lot]
    try:
        resultats = predicteur(images, batch_size=len(images))
    except Exception as souci:  # noqa: BLE001
        journal(f"  ✗ lot de {len(lot)} : {type(souci).__name__}, page à page")
        # Reprendre sans rendre la mémoire, c'est échouer 32 fois de plus :
        # le dépassement qui a tué le lot tue aussi chaque page isolée.
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        resultats = []
        for page, image, _ in lot:
            try:
                resultats.append(predicteur([image], batch_size=1)[0])
            except Exception as ennui:  # noqa: BLE001
                journal(f"  ✗ page {page} : {type(ennui).__name__}")
                resultats.append(None)
    faites = echecs = 0
    for (page, image, complement), resultat in zip(lot, resultats):
        if resultat is None:
            echecs += 1
            continue
        page_decrite = decrire(resultat, image, complement)
        page_decrite["page"] = page
        (dossier / f"p{page:06d}.json").write_text(
            json.dumps(page_decrite), encoding="utf-8")
        faites += 1
    return faites, echecs


def volume_pdf(predicteur, volume: Path, dossier: Path) -> tuple[int, int]:
    """Segmenter un PDF : le rendu part au pool, le GPU consomme au fil de l'eau."""
    from PIL import Image

    total = pages_du_pdf(volume)
    restantes = [p for p in range(1, total + 1)
                 if not (dossier / f"p{p:06d}.json").is_file()]
    if not restantes:
        return 0, 0
    journal(f"  {total} pages, {len(restantes)} à faire")
    faites = echecs = 0
    lot = []
    # Le pool naît et meurt avec le volume : ses processus relâchent ainsi le
    # PDF avant qu'on l'efface.
    with ProcessPoolExecutor(max_workers=RENDUS) as pool:
        for rendu in pool.map(rendre, [(str(volume), p) for p in restantes],
                              chunksize=4):
            if rendu is None:
                continue
            page, dpi, jpeg = rendu
            lot.append((page, Image.open(io.BytesIO(jpeg)).convert("RGB"),
                        {"resolution": round(dpi, 1)}))
            if len(lot) == LOT:
                f, e = segmenter(predicteur, lot, dossier)
                faites += f; echecs += e
                lot = []
        if lot:
            f, e = segmenter(predicteur, lot, dossier)
            faites += f; echecs += e
    return faites, echecs


def volume_archive(predicteur, volume: Path, dossier: Path) -> tuple[int, int]:
    """Segmenter une archive d'images : décoder coûte peu, on reste ici."""
    membres = pages_de_l_archive(volume)
    faites = echecs = 0
    lot = []
    for page, membre in enumerate(membres, 1):
        if (dossier / f"p{page:06d}.json").is_file():
            continue
        try:
            image = image_de_l_archive(volume, membre)
        except Exception as souci:  # noqa: BLE001
            journal(f"  ✗ page {page} illisible : {type(souci).__name__}")
            echecs += 1
            continue
        source = image.size
        if max(source) > COTE_MAX:
            image = image.copy()
            image.thumbnail((COTE_MAX, COTE_MAX))
        lot.append((page, image, {"taille_source": list(source)}))
        if len(lot) == LOT:
            f, e = segmenter(predicteur, lot, dossier)
            faites += f; echecs += e
            lot = []
    if lot:
        f, e = segmenter(predicteur, lot, dossier)
        faites += f; echecs += e
    return faites, echecs


# --- file de travail partagée ----------------------------------------------


def revendiquer(dossier: Path) -> bool:
    """Réserver un volume pour ce processus, sans concertation.

    La création exclusive d'un fichier est atomique : le premier qui l'obtient
    prend le volume, les autres passent. Une file partagée plutôt qu'un
    découpage fixe, car les volumes vont de 0,2 Mo à 700 Mo — un worker à qui
    le sort donne les gros finirait des heures après les autres.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    try:
        with (dossier / "_encours").open("x", encoding="utf-8") as marque:
            marque.write(f"{os.getpid()} {time.strftime('%F %T')}\n")
        return True
    except FileExistsError:
        return False


def prefetcheur(volumes, sortie: Path, fil: queue.Queue) -> None:
    """Prendre le volume suivant et le tirer pendant que le GPU travaille."""
    for rang, (distant, _) in enumerate(volumes, 1):
        dossier = sortie / distant.replace("/", "__")
        if (dossier / "_termine").is_file() or (dossier / "_ignore").is_file():
            continue
        if not revendiquer(dossier):
            continue
        fil.put((rang, distant, dossier, rapatrier(distant)))
    fil.put(None)


def main() -> int:
    sortie = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "regions"
    plafond = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sortie.mkdir(parents=True, exist_ok=True)

    volumes = inventaire()
    if plafond:
        volumes = volumes[:plafond]
    restants = sum(1 for c, _ in volumes
                   if not (sortie / c.replace("/", "__") / "_termine").is_file()
                   and not (sortie / c.replace("/", "__") / "_ignore").is_file())
    octets = sum(o for c, o in volumes
                 if not (sortie / c.replace("/", "__") / "_termine").is_file()
                 and not (sortie / c.replace("/", "__") / "_ignore").is_file())
    journal(f"{len(volumes)} volumes au dépôt, {restants} à faire "
            f"({octets/1e9:.1f} Go)")
    if not restants:
        return 0

    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.settings import settings
    predicteur = LayoutPredictor(
        FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    journal("modèle chargé")

    fil: queue.Queue = queue.Queue(maxsize=1)
    threading.Thread(target=prefetcheur, args=(volumes, sortie, fil),
                     daemon=True).start()

    faites = 0
    debut = time.monotonic()
    # le pool de rendu vit désormais le temps d’un volume, dans volume_pdf
    while True:
        article = fil.get()
        if article is None:
            break
        rang, distant, dossier, volume = article
        if volume is None:
            # Rendre la réservation : un échec de réseau doit se retenter,
            # pas condamner le volume au prochain passage.
            (dossier / "_encours").unlink(missing_ok=True)
            continue
        dossier.mkdir(parents=True, exist_ok=True)
        journal(f"[{rang}/{len(volumes)}] {distant[-56:]}")
        nom = volume.name.lower()
        try:
            if nom.endswith(".pdf"):
                f, echecs = volume_pdf(predicteur, volume, dossier)
            elif nom.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                f, echecs = volume_archive(predicteur, volume, dossier)
            else:
                # .djvu : aucun décodeur ici (ni ddjvu, ni roue pip
                # autonome). On le marque pour ne pas le retélécharger.
                journal(f"  ⚠ emballage non géré, ignoré : {volume.name[:52]}")
                (dossier / "_ignore").write_text("", encoding="utf-8")
                (dossier / "_encours").unlink(missing_ok=True)
                volume.unlink(missing_ok=True)
                continue
        except Exception as souci:  # noqa: BLE001
            journal(f"  ✗ volume {distant[-40:]} : {type(souci).__name__}: {souci}")
            (dossier / "_encours").unlink(missing_ok=True)
            volume.unlink(missing_ok=True)
            continue
        faites += f
        if echecs:
            # Sans marque d'achèvement, un prochain passage reprendra les
            # pages manquantes : les déclarer faites les perdrait pour de bon.
            journal(f"  ⚠ {echecs} page(s) en échec, volume laissé à reprendre")
        else:
            (dossier / "_termine").write_text("", encoding="utf-8")
        (dossier / "_encours").unlink(missing_ok=True)
        # Le volume ne sert plus : le disque compte plus que le téléchargement.
        volume.unlink(missing_ok=True)
        ecoule = time.monotonic() - debut
        journal(f"  cumul {faites} pages · {faites/max(ecoule,1e-9):.2f} pages/s")

    ecoule = time.monotonic() - debut
    journal(f"terminé : {faites} pages en {ecoule/60:.1f} min "
            f"({faites/max(ecoule,1e-9):.2f} pages/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
