"""Engine interfaces. Free engines now, paid ones later: swap by registering a different adapter."""
from pathlib import Path
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


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, gpu: bool = True, langs=("en",)):
        import easyocr
        self._r = easyocr.Reader(list(langs), gpu=gpu, verbose=False)

    def read(self, img_bgr):
        out = []
        for box, text, conf in self._r.readtext(img_bgr):
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            out.append(Block(type="text", bbox=(min(xs), min(ys), max(xs), max(ys)), text=text,
                             confidence=float(conf), source="ocr", engine=self.name))
        return out


class TrOCRHybridEngine:
    """RapidOCR finds the text lines; a TrOCR model (pretrained or our fine-tuned checkpoint) reads each crop."""

    def __init__(self, checkpoint: str = "microsoft/trocr-base-printed", label: str | None = None, batch: int = 16,
                 device: str | None = None):
        import torch
        from rapidocr_onnxruntime import RapidOCR
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.name = label or f"trocr[{Path(checkpoint).name}]"
        self._torch = torch
        self._det = RapidOCR()
        self._proc = TrOCRProcessor.from_pretrained(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = VisionEncoderDecoderModel.from_pretrained(checkpoint).to(self.device).eval()
        self._batch = batch

    def read(self, img_bgr):
        import cv2
        from PIL import Image
        boxes, _ = self._det.text_detector(img_bgr)
        if boxes is None or len(boxes) == 0:
            return []
        boxes = self._det.sorted_boxes(boxes)
        crops = self._det.get_crop_img_list(img_bgr, boxes)
        out = []
        for i in range(0, len(crops), self._batch):
            chunk = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[i:i + self._batch]]
            pix = self._proc(images=chunk, return_tensors="pt").pixel_values.to(self.device)
            with self._torch.no_grad():
                gen = self._model.generate(pix, max_length=64, output_scores=True, return_dict_in_generate=True)
            texts = self._proc.batch_decode(gen.sequences, skip_special_tokens=True)
            scores = self._model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
            confs = self._torch.exp(scores.mean(dim=1)).tolist()
            for box, text, conf in zip(boxes[i:i + self._batch], texts, confs):
                xs, ys = box[:, 0], box[:, 1]
                if text.strip():
                    out.append(Block(type="text", bbox=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                                     text=text.strip(), confidence=float(conf), source="ocr", engine=self.name))
        return out


REGISTRY = {"rapidocr": RapidOCREngine, "tesseract": TesseractEngine, "easyocr": EasyOCREngine,
            "trocr": TrOCRHybridEngine}


def get_engine(name: str, **options) -> OCREngine:
    """Build an engine by registry name; options go to its constructor (e.g. checkpoint=... for trocr)."""
    return REGISTRY[name](**options)
