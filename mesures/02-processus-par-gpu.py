import sys, time, os
from pathlib import Path
import pypdfium2 as pdfium
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

def main():
    n = int(sys.argv[1])
    doc = pdfium.PdfDocument(str(Path.home()/"tlg-sources"/"source_pdfs__alternatives_2026-09__Internet-Archive__EuthymiusEpistolae.pdf"))
    ims = [doc[p-1].render(scale=150/72).to_pil().convert("RGB") for p in range(120, 120+64)]
    doc.close()
    pred = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    pred(ims[:4])
    print(f"pret {n}", flush=True)
    t0 = time.monotonic()
    pred(ims, batch_size=32)
    print("PROC %d: %.2fs pour %d pages -> %.2f pages/s" % (n, time.monotonic()-t0, len(ims), len(ims)/(time.monotonic()-t0)), flush=True)

if __name__ == "__main__":
    main()
