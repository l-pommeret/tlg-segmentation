from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

DEPOT = "Zual/TLG_libre_scans"
MANIFESTE = f"https://huggingface.co/datasets/{DEPOT}/resolve/main/hf_tlg_manifest.csv"
SOURCES = Path.home() / "tlg-sources"
UTILES = {"Text", "TextInlineMath", "ListItem", "SectionHeader", "Title",
          "Caption", "Formula"}
INTITULE = {"SectionHeader", "Title"}


def journal(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


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
    partiel = local.with_suffix(local.suffix + ".part")
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


def pages_du_volume(volume: Path, travail: Path):
    """Rendre les pages d'un volume, quel que soit son emballage.

    Un PDF se rend page à page ; une archive porte déjà des images, qu'il
    suffit d'extraire dans l'ordre. Le reste est ignoré plutôt que deviné.
    """
    nom = volume.name.lower()
    if nom.endswith(".pdf"):
        compte = subprocess.run(["pdfinfo", str(volume)], capture_output=True,
                                text=True, check=False).stdout
        total = 0
        for ligne in compte.splitlines():
            if ligne.startswith("Pages:"):
                total = int(ligne.split()[1])
                break
        for page in range(1, total + 1):
            souche = travail / f"p{page:06d}"
            subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page),
                            "-r", "300", "-jpeg", "-singlefile",
                            str(volume), str(souche)],
                           capture_output=True, check=False)
            image = souche.with_suffix(".jpg")
            if image.is_file():
                yield page, image
                image.unlink(missing_ok=True)
        return
    if nom.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(volume) as archive:
            membres = sorted((m for m in archive.getmembers()
                              if m.isfile() and m.name.lower().endswith(
                                  (".jpg", ".jpeg", ".png", ".tif", ".tiff"))),
                             key=lambda m: m.name)
            for page, membre in enumerate(membres, 1):
                extrait = travail / f"p{page:06d}{Path(membre.name).suffix}"
                with archive.extractfile(membre) as source, \
                        extrait.open("wb") as sortie:
                    shutil.copyfileobj(source, sortie)
                yield page, extrait
                extrait.unlink(missing_ok=True)
        return
    if nom.endswith(".zip"):
        with zipfile.ZipFile(volume) as archive:
            noms = sorted(n for n in archive.namelist()
                          if n.lower().endswith((".jpg", ".jpeg", ".png",
                                                 ".tif", ".tiff")))
            for page, membre in enumerate(noms, 1):
                extrait = travail / f"p{page:06d}{Path(membre).suffix}"
                extrait.write_bytes(archive.read(membre))
                yield page, extrait
                extrait.unlink(missing_ok=True)
        return
    journal(f"  ⚠ emballage inconnu, volume ignoré : {volume.name[:52]}")


def main() -> int:
    sortie = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "regions"
    plafond = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sortie.mkdir(parents=True, exist_ok=True)

    volumes = manifeste()
    if plafond:
        volumes = volumes[:plafond]
    journal(f"{len(volumes)} volume(s) au manifeste")

    from PIL import Image
    from surya.layout import LayoutPredictor
    predicteur = LayoutPredictor()
    journal("modèle chargé")

    faites = sautees = 0
    debut = time.monotonic()
    for rang, entree in enumerate(volumes, 1):
        distant = entree.get("repo_path") or ""
        if not distant:
            continue
        marque = distant.replace("/", "__")
        dossier = sortie / marque
        if (dossier / "_termine").is_file():
            sautees += 1
            continue
        volume = rapatrier(distant)
        if volume is None:
            continue
        dossier.mkdir(parents=True, exist_ok=True)
        journal(f"[{rang}/{len(volumes)}] {distant[-56:]}")
        with tempfile.TemporaryDirectory() as travail:
            for page, image_page in pages_du_volume(volume, Path(travail)):
                cible = dossier / f"p{page:06d}.json"
                if cible.is_file():
                    continue
                try:
                    image = Image.open(image_page).convert("RGB")
                    resultat = predicteur([image])[0]
                except Exception as souci:  # noqa: BLE001
                    journal(f"  ✗ page {page} : {type(souci).__name__}")
                    continue
                regions = []
                for boite in sorted(resultat.bboxes,
                                    key=lambda b: getattr(b, "position", 0)):
                    nature = str(getattr(boite, "label", "") or "")
                    x0, y0, x1, y1 = (int(v) for v in boite.bbox)
                    regions.append({
                        "nature": nature, "boite": [x0, y0, x1, y1],
                        "ordre": int(getattr(boite, "position", 0) or 0),
                        "utile": nature in UTILES,
                        "intitule": nature in INTITULE})
                cible.write_text(json.dumps({"page": page, "regions": regions}),
                                 encoding="utf-8")
                faites += 1
                if faites % 200 == 0:
                    ecoule = time.monotonic() - debut
                    journal(f"  {faites} pages · {ecoule/faites:.2f} s/page")
        (dossier / "_termine").write_text("", encoding="utf-8")
        # Le volume ne sert plus : le disque compte plus que le téléchargement.
        volume.unlink(missing_ok=True)

    ecoule = time.monotonic() - debut
    journal(f"terminé : {faites} pages segmentées, {sautees} volume(s) déjà faits "
            f"({ecoule/max(faites,1):.2f} s/page)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
