"""Decide per page whether the text layer is usable or the page must be OCR'd."""
import hashlib
from pathlib import Path

MIN_CHARS = 20          # fewer real characters than this on a page: treat as scanned


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_kind(path) -> str:
    """'pdf' | 'image' | 'office' | 'text' | 'unknown', from the magic bytes then the extension."""
    p = Path(path)
    with open(p, "rb") as f:
        head = f.read(16)
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith((b"\x89PNG", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*", b"BM", b"RIFF")) or head[:4] == b"GIF8":
        return "image"
    ext = p.suffix.lower()
    if ext in {".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt", ".odt"}:
        return "office"
    if ext in {".txt", ".md", ".csv"}:
        return "text"
    return "unknown"


def text_layer_usable(chars: str) -> bool:
    """A page is born-digital if its text layer holds enough real characters and is not mostly replacement glyphs."""
    s = "".join(chars.split())
    if len(s) < MIN_CHARS:
        return False
    bad = sum(1 for c in s if c == "�" or ord(c) < 32)
    return bad / len(s) < 0.1
