"""CORD-v2 loading (CC BY 4.0, Clova AI). Receipts photographed in the wild, with word-level annotations.

Two uses:
  * `paper_crop`: when the dataset marks the paper region (`roi`), flatten the paper to an upright rectangle.
    That gives a clean source document with a known word list, which the capture simulator then re-photographs
    with exact ground-truth geometry.
  * the native photos themselves, as an uncontrolled real-world check.
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .geometry import order_corners, quad_area, warp_to_rect
from .ocr import tokens

DATA = Path(__file__).resolve().parents[1] / "data"
SPLITS = {"val": DATA / "cord_val.parquet", "test": DATA / "cord_test.parquet"}


def _roi_quad(gt, shape):
    roi = gt.get("roi") or {}
    if len(roi) < 8:
        return None
    q = np.array([[roi[f"x{k}"], roi[f"y{k}"]] for k in range(1, 5)], np.float32)
    h, w = shape
    frac = quad_area(q) / (h * w)
    if not (0.25 <= frac <= 0.97):
        return None
    return order_corners(q)


def _words(gt):
    out = []
    for line in gt["valid_line"]:
        for w in line["words"]:
            qd = w["quad"]
            cx = np.mean([qd[f"x{k}"] for k in range(1, 5)])
            cy = np.mean([qd[f"y{k}"] for k in range(1, 5)])
            out.append((w["text"], cx, cy))
    return out


def load_cord(split: str, limit: int | None = None):
    df = pd.read_parquet(SPLITS[split])
    for i in range(len(df) if limit is None else min(limit, len(df))):
        row = df.iloc[i]
        img = cv2.cvtColor(np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")), cv2.COLOR_RGB2BGR)
        gt = json.loads(row["ground_truth"])
        words = _words(gt)
        yield {"id": f"{split}_{i:03d}", "image": img, "words": words,
               "tokens_all": tokens(" ".join(w[0] for w in words)), "roi": _roi_quad(gt, img.shape[:2])}


def paper_crop(sample, max_side: int = 1400):
    """Flatten the annotated paper region. Returns (document_bgr, tokens_inside) or None when no usable ROI."""
    q = sample["roi"]
    if q is None:
        return None
    top, right, bottom, left = [float(np.linalg.norm(q[i] - q[(i + 1) % 4])) for i in range(4)]
    w, h = max(top, bottom), max(left, right)
    s = min(1.0, max_side / max(w, h))
    doc = warp_to_rect(sample["image"], q, (w * s, h * s))
    inside = [t for t, cx, cy in sample["words"] if cv2.pointPolygonTest(q.astype(np.float32).reshape(-1, 1, 2), (float(cx), float(cy)), False) >= 0]
    toks = tokens(" ".join(inside))
    return (doc, toks) if len(toks) >= 8 else None
