"""Engine interfaces. Free engines now, paid ones later: swap by registering a different adapter."""
from typing import Protocol

import numpy as np

from .schema import Block


class OCREngine(Protocol):
    name: str

    def read(self, img_bgr: np.ndarray) -> list[Block]:
        """OCR a page image and return text blocks (bbox in image pixels, confidence in 0..1)."""


class RapidOCREngine:
    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._eng = RapidOCR()

    def read(self, img_bgr):
        res, _ = self._eng(img_bgr)
        out = []
        for box, text, conf in (res or []):
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            out.append(Block(type="text", bbox=(min(xs), min(ys), max(xs), max(ys)), text=text,
                             confidence=float(conf), source="ocr", engine=self.name))
        return out


class TesseractEngine:
    name = "tesseract"

    def read(self, img_bgr):
        import cv2
        import pytesseract
        d = pytesseract.image_to_data(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), config="--psm 6",
                                      output_type=pytesseract.Output.DICT)
        out = []
        for i, t in enumerate(d["text"]):
            if t.strip() and float(d["conf"][i]) >= 0:
                x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
                out.append(Block(type="text", bbox=(x, y, x + w, y + h), text=t.strip(),
                                 confidence=float(d["conf"][i]) / 100, source="ocr", engine=self.name))
        return out


REGISTRY = {"rapidocr": RapidOCREngine, "tesseract": TesseractEngine}


def get_engine(name: str) -> OCREngine:
    return REGISTRY[name]()
