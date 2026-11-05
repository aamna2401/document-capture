"""Run the extractor over documents and save the outputs in an organised folder tree (this is what the API will serve).

  python scripts/export_outputs.py --out OUT_DIR --engine rapidocr --inputs data/public/arxiv data/gold/funsd data/gold/cord [--limit 10]

OUT_DIR layout:
  manifest.csv                      one row per document: type, pages, page kinds, blocks, tables, seconds, output folder
  pdf/<name>/document.md|json       Markdown and full JSON (blocks with boxes, confidence, source)
  pdf/<name>/pages/page_N.png       (--render-pdf-pages K) first K pages with extracted boxes drawn on them
  pdf/<name>/tables/page_N_table_M.csv
  images/<name>/document.md|json
  images/<name>/annotated.png       the image with OCR boxes drawn (green = confident, red = low confidence)
  <pdf|images>/<name>/review.html   SIDE BY SIDE: image with boxes | extracted text + tables, then reference, F1, missed/extra words
  index.html                        every document, lowest F1 first, linking to its review page
  tables/<name>__page_N_table_M.csv every table from every document in one place, plus tables/index.csv
Tables come from ruled tables in PDF text layers only; tables inside scanned pages or images are not detected yet.
"""
import argparse
import csv
import html
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docextract import extract                              # noqa: E402
from docextract.engines import get_engine                    # noqa: E402
from docextract.metrics import alnum_tokens, text_scores     # noqa: E402
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
    dpi, made = 100, {}
    with pdfplumber.open(pdf_path) as pdf:
        for page, pp in list(zip(doc.pages, pdf.pages))[:k]:
            img = cv2.cvtColor(np.asarray(pp.to_image(resolution=dpi).original.convert("RGB")), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"page_{page.number}.png"), draw_boxes(img, page.blocks, scale=dpi / 72))
            made[page.number] = f"pages/page_{page.number}.png"
    return made


CSS = """body{font:14px system-ui,sans-serif;margin:16px;color:#222}h1{font-size:18px}h2{font-size:15px;margin:18px 0 6px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;margin-bottom:20px}
.row img{width:100%;border:1px solid #ccc}.txt{white-space:pre-wrap;font:13px/1.45 ui-monospace,monospace;border:1px solid #ddd;padding:8px;max-height:900px;overflow:auto}
.low{background:#ffd9d9}table.t{border-collapse:collapse;margin:6px 0}table.t td,table.t th{border:1px solid #aaa;padding:3px 7px}
.miss{color:#c00}.extra{color:#b60}.meta{color:#555}a{color:#06c}table.idx{border-collapse:collapse}table.idx td,table.idx th{border:1px solid #ccc;padding:4px 8px;text-align:left}"""


def _block_html(b):
    cls = ' class="low"' if b.confidence < LOW_CONF else ""
    return f'<span{cls} title="confidence {b.confidence:.2f} ({b.source})">{html.escape(b.text)}</span>'


def _table_html(rows):
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="t">{body}</table>'


def write_review(sub, p, doc, page_imgs, engine_name):
    """One HTML page per document: image with boxes | extracted text and tables, then reference, score and word diff."""
    ref_path = p.with_suffix(".txt")
    ref = ref_path.read_text() if ref_path.exists() else None
    sc = text_scores(doc.text, ref) if ref else None
    parts = [f"<!doctype html><meta charset=utf-8><title>{html.escape(p.name)}</title><style>{CSS}</style>",
             f"<h1>{html.escape(p.name)}</h1><div class=meta>engine: {html.escape(engine_name)} &middot; "
             f"{len(doc.pages)} page(s) &middot; red = confidence below {LOW_CONF}"
             + (f" &middot; <b>F1 {sc['f1']:.3f}</b> (recall {sc['recall']:.3f}, precision {sc['precision']:.3f}, CER {sc['cer']:.3f})" if sc else "")
             + "</div>"]
    for page in doc.pages:
        img = page_imgs.get(page.number)
        left = f'<img src="{img}">' if img else "<i>page image not rendered (use --render-pdf-pages)</i>"
        tables = "".join(_table_html(b.table) for b in page.blocks if b.type == "table" and b.table)
        text = "\n".join(_block_html(b) for b in page.blocks if b.type != "table")
        parts.append(f"<h2>Page {page.number} ({page.kind})</h2><div class=row><div>{left}</div>"
                     f"<div>{tables}<div class=txt>{text}</div></div></div>")
    if ref is not None:
        r, pr = Counter(alnum_tokens(ref)), Counter(alnum_tokens(doc.text))
        miss = " ".join(f'<span class=miss>{html.escape(w)}</span>' for w in (r - pr).elements())
        extra = " ".join(f'<span class=extra>{html.escape(w)}</span>' for w in (pr - r).elements())
        parts.append(f"<h2>Reference text</h2><div class=txt>{html.escape(ref)}</div>"
                     f"<h2>Missed words (in reference, not extracted)</h2><div class=txt>{miss or '-'}</div>"
                     f"<h2>Extra words (extracted, not in reference)</h2><div class=txt>{extra or '-'}</div>")
    (sub / "review.html").write_text("\n".join(parts))
    return sc


def write_index(out, manifest):
    rows = sorted((m for m in manifest if not m.get("error")), key=lambda m: (m.get("f1") is None, m.get("f1") or 0))
    body = "".join(
        f"<tr><td><a href='{html.escape(m['output'])}/review.html'>{html.escape(m['document'])}</a></td><td>{m['type']}</td>"
        f"<td>{html.escape(str(m['source']))}</td><td>{'' if m.get('f1') is None else m['f1']}</td><td>{m['blocks']}</td><td>{m['tables']}</td></tr>"
        for m in rows)
    bad = "".join(f"<li>{html.escape(m['document'])}: {html.escape(m['error'])}</li>" for m in manifest if m.get("error"))
    (out / "index.html").write_text(f"<!doctype html><meta charset=utf-8><title>Review index</title><style>{CSS}</style>"
                                    f"<h1>Documents (lowest F1 first)</h1><table class=idx><tr><th>document<th>type<th>source<th>F1<th>blocks<th>tables</tr>{body}</table>"
                                    + (f"<h2>Failed</h2><ul>{bad}</ul>" if bad else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--engine", default="rapidocr")
    ap.add_argument("--checkpoint", default=None, help="for --engine trocr: path of a fine-tuned checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="max documents per input folder")
    ap.add_argument("--render-pdf-pages", type=int, default=3, help="pages per PDF to render with boxes for the review page")
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
            page_imgs = {}
            if kind == "image":
                import cv2
                cv2.imwrite(str(sub / "annotated.png"), draw_boxes(cv2.imread(str(p)), doc.pages[0].blocks))
                page_imgs = {1: "annotated.png"}
            elif a.render_pdf_pages:
                page_imgs = render_pages(p, doc, sub / "pages", a.render_pdf_pages)
            n_tables = write_tables(doc, [(sub / "tables", None), (out / "tables", f"{p.stem}__{{n}}")], p.stem, table_index)
            sc = write_review(sub, p, doc, page_imgs, getattr(engine, "name", a.engine))
            manifest.append({"document": p.name, "source": folder, "type": kind, "pages": len(doc.pages),
                             "f1": None if sc is None else round(sc["f1"], 4),
                             "page_kinds": "/".join(sorted({pg.kind for pg in doc.pages})),
                             "blocks": sum(len(pg.blocks) for pg in doc.pages), "tables": n_tables,
                             "seconds": round(time.time() - t, 2), "output": str(sub.relative_to(out)), "error": ""})
            print(f"{p.name}: {manifest[-1]['blocks']} blocks, {n_tables} tables", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    cols = ["document", "source", "type", "pages", "f1", "page_kinds", "blocks", "tables", "seconds", "output", "error"]
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, restval="")
        w.writeheader()
        w.writerows(manifest)
    if table_index:
        with open(out / "tables" / "index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["document", "page", "table", "rows", "cols", "file"])
            w.writeheader()
            w.writerows(table_index)
    write_index(out, manifest)
    print(f"saved {len(manifest)} documents, {len(table_index)} tables -> {out}")


if __name__ == "__main__":
    main()
