"""Demo: extract a document and save Markdown, JSON and a one-line summary.

  python scripts/demo.py path/to/file.pdf|png|jpg [--engine rapidocr] [--capture] [--out demo/]
Light on CPU for born-digital PDFs (no OCR). Scanned pages run OCR; prefer Colab for long documents.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docextract import extract                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--engine", default="rapidocr")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--out", default="demo")
    a = ap.parse_args()
    doc = extract(a.file, ocr_engine=a.engine, capture=a.capture)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.file).stem
    (out / f"{stem}.md").write_text(doc.to_markdown())
    (out / f"{stem}.json").write_text(doc.model_dump_json(indent=1))
    kinds = [p.kind for p in doc.pages]
    tables = sum(b.type == "table" for p in doc.pages for b in p.blocks)
    print(f"{a.file}: {len(doc.pages)} pages {dict((k, kinds.count(k)) for k in set(kinds))}, "
          f"{sum(len(p.blocks) for p in doc.pages)} blocks, {tables} tables -> {out}/{stem}.md|json")


if __name__ == "__main__":
    main()
