import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docextract import extract                         # noqa: E402
from docextract.cache import PageCache                 # noqa: E402
from docextract.metrics import cer, wer                # noqa: E402
from docextract.triage import file_kind, text_layer_usable  # noqa: E402


def make_pdf(path, lines):
    """Minimal one-page PDF with Helvetica text, written by hand (no PDF library needed)."""
    content = "BT /F1 14 Tf 72 720 Td 18 TL " + " ".join(f"({t}) Tj T*" for t in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    Path(path).write_bytes(out)


def test_born_digital_pdf_uses_text_layer(tmp_path):
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf, ["Invoice number 12345", "Total due 99.50 USD", "Thank you for your business"])
    doc = extract(pdf, cache=PageCache(tmp_path / "cache"))
    assert doc.pages[0].kind == "born_digital"
    assert all(b.source == "text_layer" for b in doc.pages[0].blocks)
    assert "Total due 99.50 USD" in doc.text
    assert cer(doc.text, "Invoice number 12345\nTotal due 99.50 USD\nThank you for your business") < 0.01


def test_cache_roundtrip(tmp_path):
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf, ["Hello world this is a test page"])
    cache = PageCache(tmp_path / "cache")
    first = extract(pdf, cache=cache)
    second = extract(pdf, cache=cache)
    assert first.model_dump() == second.model_dump()
    assert list((tmp_path / "cache").rglob("*.json"))


def test_file_kind_and_usable(tmp_path):
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf, ["x"])
    assert file_kind(pdf) == "pdf"
    assert not text_layer_usable("  ab  ")
    assert text_layer_usable("a reasonably long line of real text")


def test_metrics():
    assert cer("abc", "abc") == 0 and cer("abd", "abc") == pytest.approx(1 / 3)
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)


def _text_image(lines, w=1200, h=700):
    import cv2
    import numpy as np
    img = np.full((h, w, 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (60, 120 + i * 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3, cv2.LINE_AA)
    return img


LINES = ["INVOICE 20458", "Total due 99.50 USD", "Thank you"]


def test_image_input_is_ocrd(tmp_path):
    import cv2
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), _text_image(LINES))
    doc = extract(p, cache=PageCache(tmp_path / "cache"))
    assert doc.pages[0].kind == "scanned" and all(b.source == "ocr" for b in doc.pages[0].blocks)
    assert cer(doc.text, "\n".join(LINES)) < 0.15


def test_scanned_pdf_falls_back_to_ocr(tmp_path):
    from PIL import Image
    import cv2
    img = cv2.cvtColor(_text_image(LINES), cv2.COLOR_BGR2RGB)
    pdf = tmp_path / "scan.pdf"
    Image.fromarray(img).save(pdf, resolution=150)               # image-only PDF: no text layer
    doc = extract(pdf, cache=PageCache(tmp_path / "cache"))
    assert doc.pages[0].kind == "scanned"
    assert cer(doc.text, "\n".join(LINES)) < 0.15
    x0, y0, x1, y1 = doc.pages[0].blocks[0].bbox                   # boxes are in PDF points, inside the page
    assert 0 <= x0 < x1 <= doc.pages[0].width and 0 <= y0 < y1 <= doc.pages[0].height


def test_photo_through_docscan(tmp_path):
    import cv2
    from docscan.simulate import capture
    doc_img = _text_image(LINES, 600, 900)
    photo, _, _ = capture(doc_img, "easy", seed=1)
    p = tmp_path / "photo.jpg"
    cv2.imwrite(str(p), photo)
    out = extract(p, cache=PageCache(tmp_path / "cache"), capture=True)
    assert out.pages[0].blocks


def test_tight_kerning_keeps_word_spaces(tmp_path):
    """Regression: with a 3pt gap tolerance, tightly set text came out as 'Providedproperattribution'."""
    pdf = tmp_path / "k.pdf"
    content = "BT /F1 10 Tf -0.9 Tc 72 720 Td (Provided proper attribution is provided here for you) Tj ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream", "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out)); out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    x = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode() + b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF".encode()
    pdf.write_bytes(out)
    doc = extract(pdf, cache=PageCache(tmp_path / "cache"))
    assert "Provided proper attribution" in doc.text
