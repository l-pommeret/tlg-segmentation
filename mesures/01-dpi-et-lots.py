import time, os
from pathlib import Path
import pypdfium2 as pdfium
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

VOL = Path.home()/"tlg-sources"/"source_pdfs__alternatives_2026-09__Internet-Archive__EuthymiusEpistolae.pdf"
PAGES = list(range(120, 136))   # 16 pages

def rendre(dpi):
    doc = pdfium.PdfDocument(str(VOL))
    t0 = time.monotonic()
    ims = [doc[p-1].render(scale=dpi/72).to_pil().convert("RGB") for p in PAGES]
    dt = time.monotonic()-t0
    doc.close()
    return ims, dt

def main():
    t0 = time.monotonic()
    found = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    pred = LayoutPredictor(found)
    print("modele charge en %.1fs" % (time.monotonic()-t0))
    for dpi in (300, 150, 100):
        ims, dtr = rendre(dpi)
        print(f"\n--- {dpi} dpi, {ims[0].size}, rendu {dtr:.2f}s ({dtr/len(ims):.3f} s/page CPU) ---")
        pred(ims[:2])  # rechauffe
        for bs in (8, 16, 32):
            t0 = time.monotonic()
            res = pred(ims, batch_size=bs)
            dt = time.monotonic()-t0
            n = sum(len(r.bboxes) for r in res)
            print(f"  batch={bs:3d}: {dt:5.2f}s  {dt/len(ims):.3f} s/page GPU  regions={n}")

if __name__ == "__main__":
    main()
