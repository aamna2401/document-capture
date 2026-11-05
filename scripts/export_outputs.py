"""Run the extractor over documents and save the outputs in an organised folder tree (this is what the API will serve).

  python scripts/export_outputs.py --out OUT_DIR --engine rapidocr --inputs data/public/arxiv data/gold/funsd data/gold/cord [--limit 10]

OUT_DIR layout:
  manifest.csv                      one row per document: type, pages, page kinds, blocks, tables, seconds, output folder
  pdf/<name>/document.md|json       Markdown and full JSON (blocks with boxes, confidence, source)
  pdf/<name>/pages/page_N.png       (--render-pdf-pages K) first K pages with extracted boxes drawn on them
  pdf/<name>/tables/page_N_table_M.csv
  images/<name>/document.md|json
  images/<name>/annotated.png       the image with OCR boxes drawn (green = confident, red = low confidence)
  tables/<name>__page_N_table_M.csv every table from every document in one place, plus tables/index.csv
Tables come from ruled tables in PDF text layers only; tables inside scanned pages or images are not detected yet.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docextract import extract                              # noqa: E402
from docextract.engines import get_engine                    # noqa: E402
from docextract.triage import file_kind                      # noqa: E402

LOW_CONF = 0.6


def draw_boxes(img, blocks, scale=1.0):
    import cv2
    out = img.copy()
    for b in blocks:
        x0, y0, x1, y1 = [int(v * scale) for v in b.bbox]
        color = (0, 160, 0) if b.confidence >= LOW_CONF else (0, 0, 220)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    return out


def write_tables(doc, dirs, stem, index_rows):
    n = 0
    for page in doc.pages:
        t = 0
        for b in page.blocks:
            if b.type != "table" or not b.table:
                continue
            t += 1
            name = f"page_{page.number}_table_{t}.csv"
            for d, fname in dirs:
                d.mkdir(parents=True, exist_ok=True)
                with open(d / (fname or name).replace("{n}", name), "w", newline="") as f:
                    csv.writer(f).writerows(b.table)
            index_rows.append({"document": stem, "page": page.number, "table": t, "rows": len(b.table),
                               "cols": max(len(r) for r in b.table), "file": f"{stem}__{name}"})
            n += 1
    return n


def render_pages(pdf_path, doc, out_dir, k):
    import cv2
    import numpy as np
    import pdfplumber
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = 100
    with pdfplumber.open(pdf_path) as pdf:
        for page, pp in list(zip(doc.pages, pdf.pages))[:k]:
            img = cv2.cvtColor(np.asarray(pp.to_image(resolution=dpi).original.convert("RGB")), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"page_{page.number}.png"), draw_boxes(img, page.blocks, scale=dpi / 72))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--engine", default="rapidocr")
    ap.add_argument("--checkpoint", default=None, help="for --engine trocr: path of a fine-tuned checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="max documents per input folder")
    ap.add_argument("--render-pdf-pages", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    engine = get_engine(a.engine, **({"checkpoint": a.checkpoint, "label": "trocr_finetuned"} if a.checkpoint else {}))
    manifest, table_index = [], []
    for folder in a.inputs:
        files = [p for p in sorted(Path(folder).iterdir()) if file_kind(p) in ("pdf", "image")]
        for p in files[: a.limit]:
            kind = file_kind(p)
            sub = out / ("pdf" if kind == "pdf" else "images") / p.stem
            sub.mkdir(parents=True, exist_ok=True)
            t = time.time()
            try:
                doc = extract(p, ocr_engine=engine)
            except Exception as e:                           # one bad file must not stop the batch
                manifest.append({"document": p.name, "source": folder, "type": kind, "error": f"{type(e).__name__}: {e}"})
                print(f"FAILED {p.name}: {e}")
                continue
            (sub / "document.md").write_text(doc.to_markdown())
            (sub / "document.json").write_text(doc.model_dump_json(indent=1))
            if kind == "image":
                import cv2
                cv2.imwrite(str(sub / "annotated.png"), draw_boxes(cv2.imread(str(p)), doc.pages[0].blocks))
            elif a.render_pdf_pages:
                render_pages(p, doc, sub / "pages", a.render_pdf_pages)
            n_tables = write_tables(doc, [(sub / "tables", None), (out / "tables", f"{p.stem}__{{n}}")], p.stem, table_index)
            manifest.append({"document": p.name, "source": folder, "type": kind, "pages": len(doc.pages),
                             "page_kinds": "/".join(sorted({pg.kind for pg in doc.pages})),
                             "blocks": sum(len(pg.blocks) for pg in doc.pages), "tables": n_tables,
                             "seconds": round(time.time() - t, 2), "output": str(sub.relative_to(out)), "error": ""})
            print(f"{p.name}: {manifest[-1]['blocks']} blocks, {n_tables} tables", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    cols = ["document", "source", "type", "pages", "page_kinds", "blocks", "tables", "seconds", "output", "error"]
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, restval="")
        w.writeheader()
        w.writerows(manifest)
    if table_index:
        with open(out / "tables" / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["document", "page", "table", "rows", "cols", "file"])
            w.writeheader()
            w.writerows(table_index)
    print(f"saved {len(manifest)} documents, {len(table_index)} tables -> {out}")


if __name__ == "__main__":
    main()
