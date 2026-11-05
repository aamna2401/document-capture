"""Born-digital extraction: text lines with boxes, and ruled tables, straight from the PDF (no OCR)."""
import pdfplumber

from .schema import Block, Page
from .triage import text_layer_usable

SPACE_RATIO = 0.15      # gap as a fraction of font size that counts as a space; the default of 3pt glues words in tight fonts


def extract_page(plumber_page, number: int) -> tuple[Page, bool]:
    """Return (page, usable). usable=False means the caller should OCR this page instead."""
    w, h = float(plumber_page.width), float(plumber_page.height)
    raw = plumber_page.extract_text(**({"x_tolerance_ratio": SPACE_RATIO} if SPACE_RATIO else {})) or ""
    if not text_layer_usable(raw):
        return Page(number=number, width=w, height=h, kind="scanned"), False

    blocks: list[Block] = []
    table_boxes = []
    for t in plumber_page.find_tables():
        rows = [[(c or "").strip() for c in r] for r in t.extract()]
        if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
            blocks.append(Block(type="table", bbox=tuple(map(float, t.bbox)), table=rows,
                                text="\n".join(" | ".join(r) for r in rows)))
            table_boxes.append(t.bbox)

    def inside_table(line) -> bool:
        cx, cy = (line["x0"] + line["x1"]) / 2, (line["top"] + line["bottom"]) / 2
        return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in table_boxes)

    for ln in plumber_page.extract_text_lines(**({"x_tolerance_ratio": SPACE_RATIO} if SPACE_RATIO else {})):
        if not inside_table(ln) and ln["text"].strip():
            blocks.append(Block(type="text", bbox=(ln["x0"], ln["top"], ln["x1"], ln["bottom"]), text=ln["text"].strip()))

    blocks.sort(key=lambda b: (round(b.bbox[1]), b.bbox[0]))          # simple top-to-bottom, left-to-right order
    return Page(number=number, width=w, height=h, kind="born_digital", blocks=blocks), True


def open_pdf(path):
    return pdfplumber.open(path)
