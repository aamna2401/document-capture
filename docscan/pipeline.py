"""End-to-end scan: detect -> rectify -> orient -> deskew."""
from dataclasses import dataclass

import numpy as np

from .detect import DetectParams, Detection, detect_document
from .orient import best_rotation, deskew, estimate_skew, rotate_k
from .rectify import rectify


@dataclass
class ScanResult:
    image: np.ndarray
    detection: Detection
    rotation_k: int = 0
    skew_deg: float = 0.0
    size: tuple = (0, 0)


def scan(img_bgr, orient: bool = True, fix_skew: bool = True, aspect: str = "zhang",
         params: DetectParams | None = None) -> ScanResult:
    det = detect_document(img_bgr, params)
    flat, size = rectify(img_bgr, det.quad, aspect)
    k, skew = 0, 0.0
    if orient:
        k, _ = best_rotation(flat)
        flat = rotate_k(flat, k)
    if fix_skew:
        skew = estimate_skew(flat)
        flat = deskew(flat, skew)
    return ScanResult(image=flat, detection=det, rotation_k=k, skew_deg=skew, size=size)
