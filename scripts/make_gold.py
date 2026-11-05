"""Build public gold sets under data/gold/ from the downloads in data/public/.

  funsd/   50 real scanned forms (FUNSD test split); reference = annotated words, rough reading order
  arxiv/   scanned copies (150 DPI, image-only PDF) of the first pages of arXiv papers;
           reference = the original PDF's own text layer (so OCR is scored against exact text)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pdfplumber                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data"
PAGES_PER_PAPER = 3


def funsd():
    src = ROOT / "public/funsd_raw/dataset/testing_data"
    out = ROOT / "gold/funsd"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for ann in sorted((src / "annotations").glob("*.json")):
        words = [(w["box"], w["text"]) for f in json.load(open(ann))["form"] for w in f["words"] if w["text"].strip()]
        words.sort(key=lambda bw: (round((bw[0][1] + bw[0][3]) / 2 / 12), bw[0][0]))     # ~12px line buckets
        ref = " ".join(t for _, t in words)
        img = src / "images" / (ann.stem + ".png")
        if not img.exists() or not ref:
            continue
        (out / img.name).write_bytes(img.read_bytes())
        (out / (ann.stem + ".txt")).write_text(ref)
        n += 1
    print(f"funsd: {n} files")


def arxiv():
    out = ROOT / "gold/arxiv"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for pdf in sorted((ROOT / "public/arxiv").glob("*.pdf")):
        with pdfplumber.open(pdf) as doc:
            for i, page in enumerate(doc.pages[:PAGES_PER_PAPER], 1):
                ref = page.extract_text(x_tolerance_ratio=0.15) or ""
                if len(ref) < 200:
                    continue
                name = f"{pdf.stem}_p{i}"
                page.to_image(resolution=150).original.convert("RGB").save(out / f"{name}.pdf", resolution=150)
                (out / f"{name}.txt").write_text(ref)
                n += 1
    print(f"arxiv: {n} scanned pages")


def cord():
    """CORD-v2 test split: receipt photos. Reference = all annotated words, rough reading order. Needs data/cord_test.parquet."""
    from docscan.data import load_cord
    import cv2
    out = ROOT / "gold/cord"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in load_cord("test"):
        words = sorted(s["words"], key=lambda w: (round(w[2] / 15), w[1]))
        (out / f"{s['id']}.txt").write_text(" ".join(w[0] for w in words))
        cv2.imwrite(str(out / f"{s['id']}.jpg"), s["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"cord: {n} receipts")


if __name__ == "__main__":
    funsd()
    arxiv()
    if (ROOT / "cord_test.parquet").exists():
        cord()
    else:
        print("cord: skipped (data/cord_test.parquet missing)")
