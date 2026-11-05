"""extract(): triage each page, use the text layer when it is usable, otherwise OCR (cheapest engine first)."""
import numpy as np

from .cache import PageCache
from .schema import Document, Page
from .triage import file_kind, sha256_file
from .textlayer import extract_page, open_pdf

OCR_DPI = 300


def _render(plumber_page, dpi=OCR_DPI) -> np.ndarray:
    import cv2
    img = plumber_page.to_image(resolution=dpi).original.convert("RGB")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _ocr_page(page: Page, img_bgr, engine) -> Page:
    scale = 72.0 / OCR_DPI                      # pixels -> PDF points, so boxes share one coordinate system
    blocks = engine.read(img_bgr)
    for b in blocks:
        b.bbox = tuple(v * scale for v in b.bbox)
    page.blocks = sorted(blocks, key=lambda b: (round(b.bbox[1] / 4), b.bbox[0]))
    page.kind = "scanned" if blocks else "empty"
    return page


def _extract_image(path, sha, ocr_engine, cache, capture: bool, engine_obj=None) -> Document:
    """A photo or scan: optionally run docscan (detect, rectify, orient, deskew), then OCR as one page."""
    import cv2
    key = ocr_engine + ("+docscan" if capture else "")
    doc = Document(path=str(path), sha256=sha)
    hit = cache.get(sha, 1, key)
    if hit:
        doc.pages.append(hit)
        return doc
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    if capture:
        from docscan.pipeline import scan
        img = scan(img).image
    h, w = img.shape[:2]
    from .engines import get_engine
    page = Page(number=1, width=float(w), height=float(h), kind="scanned")
    blocks = (engine_obj or get_engine(ocr_engine)).read(img)  # image pixels are the coordinate system here
    page.blocks = sorted(blocks, key=lambda b: (round(b.bbox[1] / 4), b.bbox[0]))
    page.kind = "scanned" if blocks else "empty"
    cache.put(sha, 1, key, page)
    doc.pages.append(page)
    return doc


def extract(path, ocr_engine="rapidocr", cache: PageCache | None = None, capture: bool = False) -> Document:
    """capture=True runs the docscan cleanup on image inputs (use for phone photos, not flat scans)."""
    """ocr_engine is a registry name or an already-built engine object (needed for options such as a checkpoint)."""
    kind = file_kind(path)
    sha = sha256_file(path)
    cache = cache or PageCache()
    engine_obj = None if isinstance(ocr_engine, str) else ocr_engine
    ocr_engine = ocr_engine if isinstance(ocr_engine, str) else engine_obj.name
    if kind == "image":
        return _extract_image(path, sha, ocr_engine, cache, capture, engine_obj)
    if kind != "pdf":
        raise NotImplementedError(f"{kind}: only PDFs and images are handled so far (Office files are next)")
    doc = Document(path=str(path), sha256=sha)
    engine = engine_obj
    with open_pdf(path) as pdf:
        for i, pp in enumerate(pdf.pages, 1):
            hit = cache.get(sha, i, ocr_engine)
            if hit:
                doc.pages.append(hit)
                continue
            page, usable = extract_page(pp, i)
            if not usable:
                from .engines import get_engine
                engine = engine or get_engine(ocr_engine)
                page = _ocr_page(page, _render(pp), engine)
            cache.put(sha, i, ocr_engine, page)
            doc.pages.append(page)
    return doc
