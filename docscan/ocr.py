"""OCR wrappers (Tesseract and RapidOCR) and token normalisation used by the metrics."""
import os
import re
from pathlib import Path

import cv2
import numpy as np

_TESSDATA = Path(__file__).resolve().parents[1] / "data" / "tessdata"
if _TESSDATA.exists():
    os.environ.setdefault("TESSDATA_PREFIX", str(_TESSDATA))
os.environ.setdefault("OMP_THREAD_LIMIT", "1")    # Tesseract: one thread per call; parallelism comes from worker processes

import pytesseract  # noqa: E402

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,:/\-][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    """Lower-case alphanumeric tokens; keeps inner . , : / - so '12.50' and '2019-03-01' stay whole."""
    return _TOKEN.findall(text.lower())


class Tesseract:
    name = "tesseract"

    def read(self, img_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_string(rgb, config="--psm 6")


class Rapid:
    name = "rapidocr"

    def __init__(self, threads: int = 1):
        from rapidocr_onnxruntime import RapidOCR
        self._eng = RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=threads)

    def read(self, img_bgr: np.ndarray) -> str:
        res, _ = self._eng(img_bgr)
        return " ".join(r[1] for r in (res or []))


def word_confidence_score(img_bgr: np.ndarray, max_side: int = 900) -> float:
    """Quality of a quick Tesseract pass: number of confident real-looking words. Used to pick a page rotation."""
    h, w = img_bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    d = pytesseract.image_to_data(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), config="--psm 6",
                                  output_type=pytesseract.Output.DICT)
    score = 0.0
    for t, c in zip(d["text"], d["conf"]):
        t = t.strip()
        if len(t) >= 2 and any(ch.isalnum() for ch in t) and float(c) > 50:
            score += float(c) / 100.0
    return score
