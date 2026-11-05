"""Evaluate docextract text accuracy on a gold set. Resumable: finished files are skipped on rerun.

Gold set layout (one folder):   name.pdf | name.png | name.jpg   and   name.txt  (the reference text)

  python scripts/eval_extract.py data/gold --engine rapidocr [--capture]
Writes outputs/results/extract_<engine>.csv, appending one row per file as it finishes.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docextract import extract                       # noqa: E402
from docextract.metrics import cer, wer              # noqa: E402
from docscan.metrics import word_prf                 # noqa: E402
from docscan.ocr import tokens                       # noqa: E402

EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FIELDS = ["file", "engine", "pages", "born_digital_pages", "cer", "wer", "f1", "recall", "seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--engine", default="rapidocr")
    ap.add_argument("--capture", action="store_true", help="run docscan on image inputs (phone photos)")
    a = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / "outputs" / "results" / f"extract_{Path(a.folder).name}_{a.engine}{'_capture' if a.capture else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r["file"] for r in csv.DictReader(f)}
    new = not out.exists()

    files = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in EXTS and p.with_suffix(".txt").exists())
    print(f"{len(files)} gold files, {len(done)} already done")
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for p in files:
            if p.name in done:
                continue
            t = time.time()
            doc = extract(p, ocr_engine=a.engine, capture=a.capture)
            ref = p.with_suffix(".txt").read_text()
            _, rec, f1 = word_prf(tokens(doc.text), tokens(ref))     # order-free; the headline metric
            w.writerow({"file": p.name, "engine": a.engine, "pages": len(doc.pages),
                        "born_digital_pages": sum(pg.kind == "born_digital" for pg in doc.pages),
                        "cer": round(cer(doc.text, ref), 4), "wer": round(wer(doc.text, ref), 4),
                        "f1": round(f1, 4), "recall": round(rec, 4),
                        "seconds": round(time.time() - t, 2)})
            f.flush()                                    # a disconnect loses at most the current file
            print(f"  {p.name}  f1={f1:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
