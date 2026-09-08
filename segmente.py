"""Segmenter en régions toutes les pages du dépôt TLG, en vue de l'océrisation.

Un processus tient un GPU et fait tourner le modèle de mise en page ; le rendu
des pages, qui est du calcul CPU, part dans un pool de processus, et le volume
suivant se télécharge pendant que le courant est segmenté. Le GPU n'attend donc
ni le disque ni le réseau.

Plusieurs exemplaires se partagent le manifeste par PART/PARTS ; voir lancer.sh.
"""

from __future__ import annotations

import csv
import io
import json
import os
import queue
import shutil
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DEPOT = "Zual/TLG_libre_scans"
MANIFESTE = f"https://huggingface.co/datasets/{DEPOT}/resolve/main/hf_tlg_manifest.csv"
SOURCES = Path.home() / "tlg-sources"
UTILES = {"Text", "TextInlineMath", "ListItem", "SectionHeader", "Title",
          "Caption", "Formula"}
INTITULE = {"SectionHeader", "Title"}
IMAGES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

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


def manifeste() -> list[dict]:
    """La liste des volumes, telle que le dépôt la publie."""
    with urllib.request.urlopen(MANIFESTE, timeout=120) as flux:
        texte = flux.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(texte)))


def rapatrier(chemin_distant: str) -> Path | None:
    """Tirer un volume du dépôt, une seule fois."""
    local = SOURCES / chemin_distant.replace("/", "__")
    if local.is_file() and local.stat().st_size > 0:
        return local
    SOURCES.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{DEPOT}/resolve/main/{chemin_distant}"
    partiel = local.with_suffix(local.suffix + f".part{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=600) as flux, \
                partiel.open("wb") as sortie:
            shutil.copyfileobj(flux, sortie, 1 << 20)
    except Exception as souci:  # noqa: BLE001
        partiel.unlink(missing_ok=True)
        journal(f"  ✗ téléchargement {chemin_distant[:52]} : {type(souci).__name__}")
        return None
    partiel.rename(local)
    return local


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
    return Image.open(io.BytesIO(donnees)).convert("RGB")


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


def segmenter(predicteur, lot, dossier: Path) -> int:
    """Segmenter un paquet de pages ; en cas d'échec, retomber page à page."""
    images = [image for _, image, _ in lot]
    try:
        resultats = predicteur(images, batch_size=len(images))
    except Exception as souci:  # noqa: BLE001
        journal(f"  ✗ lot de {len(lot)} : {type(souci).__name__}, page à page")
        resultats = []
        for _, image, _ in lot:
            try:
                resultats.append(predicteur([image], batch_size=1)[0])
            except Exception:  # noqa: BLE001
                resultats.append(None)
    faites = 0
    for (page, image, complement), resultat in zip(lot, resultats):
        if resultat is None:
            journal(f"  ✗ page {page}")
            continue
        page_decrite = decrire(resultat, image, complement)
        page_decrite["page"] = page
        (dossier / f"p{page:06d}.json").write_text(
            json.dumps(page_decrite), encoding="utf-8")
        faites += 1
    return faites


def volume_pdf(predicteur, volume: Path, dossier: Path) -> int:
    """Segmenter un PDF : le rendu part au pool, le GPU consomme au fil de l'eau."""
    from PIL import Image

    total = pages_du_pdf(volume)
    restantes = [p for p in range(1, total + 1)
                 if not (dossier / f"p{p:06d}.json").is_file()]
    if not restantes:
        return 0
    journal(f"  {total} pages, {len(restantes)} à faire")
    faites = 0
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
                faites += segmenter(predicteur, lot, dossier)
                lot = []
        if lot:
            faites += segmenter(predicteur, lot, dossier)
    return faites


def volume_archive(predicteur, volume: Path, dossier: Path) -> int:
    """Segmenter une archive d'images : décoder coûte peu, on reste ici."""
    membres = pages_de_l_archive(volume)
    faites = 0
    lot = []
    for page, membre in enumerate(membres, 1):
        if (dossier / f"p{page:06d}.json").is_file():
            continue
        try:
            image = image_de_l_archive(volume, membre)
        except Exception:  # noqa: BLE001
            journal(f"  ✗ page {page} illisible")
            continue
        source = image.size
        if max(source) > COTE_MAX:
            image = image.copy()
            image.thumbnail((COTE_MAX, COTE_MAX))
        lot.append((page, image, {"taille_source": list(source)}))
        if len(lot) == LOT:
            faites += segmenter(predicteur, lot, dossier)
            lot = []
    if lot:
        faites += segmenter(predicteur, lot, dossier)
    return faites


# --- téléchargement en avance ----------------------------------------------


def prefetcheur(travaux, fil: queue.Queue) -> None:
    """Tirer le volume suivant pendant que le courant occupe le GPU."""
    for rang, distant, dossier in travaux:
        fil.put((rang, distant, dossier, rapatrier(distant)))
    fil.put(None)


def main() -> int:
    sortie = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "regions"
    plafond = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    part = int(os.environ.get("PART", "0"))
    parts = int(os.environ.get("PARTS", "1"))
    sortie.mkdir(parents=True, exist_ok=True)

    volumes = manifeste()
    if plafond:
        volumes = volumes[:plafond]

    # Le manifeste répète un fichier quand plusieurs œuvres en proviennent :
    # 415 lignes pour 409 chemins. Sans ce dédoublonnage, deux rangs différents
    # tombent sur deux workers qui segmentent le même volume dans le même
    # dossier — le travail est fait deux fois.
    uniques, vus = [], set()
    for entree in volumes:
        distant = entree.get("repo_path") or ""
        if distant and distant not in vus:
            vus.add(distant)
            uniques.append(distant)

    travaux = []
    for rang, distant in enumerate(uniques, 1):
        if rang % parts != part % parts:
            continue
        dossier = sortie / distant.replace("/", "__")
        if (dossier / "_termine").is_file() or (dossier / "_ignore").is_file():
            continue
        travaux.append((rang, distant, dossier))
    journal(f"{len(volumes)} lignes, {len(uniques)} volumes distincts, "
            f"{len(travaux)} à faire ici")
    if not travaux:
        return 0

    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.settings import settings
    predicteur = LayoutPredictor(
        FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    journal("modèle chargé")

    fil: queue.Queue = queue.Queue(maxsize=1)
    threading.Thread(target=prefetcheur, args=(travaux, fil), daemon=True).start()

    faites = 0
    debut = time.monotonic()
    # le pool de rendu vit désormais le temps d’un volume, dans volume_pdf
    while True:
        article = fil.get()
        if article is None:
            break
        rang, distant, dossier, volume = article
        if volume is None:
            continue
        dossier.mkdir(parents=True, exist_ok=True)
        journal(f"[{rang}/{len(uniques)}] {distant[-56:]}")
        nom = volume.name.lower()
        try:
            if nom.endswith(".pdf"):
                faites += volume_pdf(predicteur, volume, dossier)
            elif nom.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                faites += volume_archive(predicteur, volume, dossier)
            else:
                # .djvu : aucun décodeur ici (ni ddjvu, ni roue pip
                # autonome). On le marque pour ne pas le retélécharger.
                journal(f"  ⚠ emballage non géré, ignoré : {volume.name[:52]}")
                (dossier / "_ignore").write_text("", encoding="utf-8")
                volume.unlink(missing_ok=True)
                continue
        except Exception as souci:  # noqa: BLE001
            journal(f"  ✗ volume {distant[-40:]} : {type(souci).__name__}: {souci}")
            volume.unlink(missing_ok=True)
            continue
        (dossier / "_termine").write_text("", encoding="utf-8")
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
