import time
from pathlib import Path
import pypdfium2 as pdfium
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

VOL = Path.home()/"tlg-sources"/"source_pdfs__gallica_2026-09__bpt6k9120m.pdf"

def main():
    doc = pdfium.PdfDocument(str(VOL))
    feuille = doc[40]
    cote_pt = max(feuille.get_size())
    print("page en points :", feuille.get_size())
    pred = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    for etiquette, echelle in (("150 dpi fixe", 150/72), ("cote max 1400", 1400/cote_pt)):
        t0 = time.monotonic()
        ims = [doc[p].render(scale=echelle).to_pil().convert("RGB") for p in range(40, 48)]
        tr = time.monotonic()-t0
        pred(ims[:1])
        t0 = time.monotonic()
        res = pred(ims, batch_size=8)
        tg = time.monotonic()-t0
        print(f"{etiquette:16s} {ims[0].size}  rendu {tr/8:.3f} s/page  GPU {tg/8:.3f} s/page  regions={sum(len(r.bboxes) for r in res)}")
    doc.close()

if __name__ == "__main__":
    main()
