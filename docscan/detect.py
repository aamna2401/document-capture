"""Find the document quadrilateral in a photo.

Several cheap strategies each propose candidate quadrilaterals:
  * Otsu thresholds (bright and dark paper), adaptive threshold, Canny edge closing,
  * background modelling: the border of the frame is assumed to be background, everything that differs in Lab
    colour is foreground,
  * GrabCut initialised with a frame-inset rectangle.
A scoring function picks the candidate that looks most like a sheet of paper: edge support along its border,
plausible area, brightness contrast with the surroundings, sane corner angles, and how well it matches the
foreground mask it came from. If nothing scores well enough the whole frame is returned, so the pipeline
degrades to "no change" instead of failing.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import interior_angles, is_convex, order_corners, quad_area

FEATURES = ("edge", "area", "contrast", "angle", "fit")
_TUNED = json.loads((Path(__file__).parent / "tuned_weights.json").read_text())   # fitted on the validation split only


def _tuned_weights():
    w = _TUNED["weights"]
    feat = dict(zip(FEATURES, w[:len(FEATURES)]))
    bias = dict(zip(_TUNED["methods"], w[len(FEATURES):]))
    return feat, bias


@dataclass
class DetectParams:
    work_size: int = 800            # longest side used for detection
    min_area: float = 0.12          # smallest accepted quad, fraction of the frame
    max_area: float = 0.995
    min_score: float = _TUNED["threshold"]      # below this we fall back to the full frame
    w_edge: float = _tuned_weights()[0]["edge"]
    w_area: float = _tuned_weights()[0]["area"]
    w_contrast: float = _tuned_weights()[0]["contrast"]
    w_angle: float = _tuned_weights()[0]["angle"]
    w_fit: float = _tuned_weights()[0]["fit"]
    method_bias: dict = field(default_factory=lambda: _tuned_weights()[1])
    use_grabcut: bool = True
    use_bgdist: bool = True


@dataclass
class Detection:
    quad: np.ndarray                # (4, 2) float32, tl,tr,br,bl, original image coordinates
    score: float
    method: str
    all_quads: list = field(default_factory=list)   # dicts: quad, method, feats, score (original coordinates)


def _fit_iou(hull, q):
    inter, _ = cv2.intersectConvexConvex(hull.reshape(-1, 2).astype(np.float32), q.astype(np.float32))
    union = abs(cv2.contourArea(hull.astype(np.float32))) + quad_area(q) - inter
    return float(inter / union) if union > 0 else 0.0


def _quads_from_mask(mask, name, frame_area):
    cands = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(c) < 0.05 * frame_area:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        got = False
        for eps in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                q = order_corners(approx.reshape(4, 2))
                cands.append((q, f"{name}:poly", _fit_iou(hull, q)))
                got = True
                break
        if not got:
            q = order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))
            cands.append((q, f"{name}:rect", _fit_iou(hull, q)))
    return cands


def _edge_support(q, edge_d):
    h, w = edge_d.shape
    hits, total = 0, 0
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        for t in np.linspace(0, 1, 60):
            x, y = a + (b - a) * t
            xi, yi = int(round(np.clip(x, 0, w - 1))), int(round(np.clip(y, 0, h - 1)))
            on_border = x <= 2 or y <= 2 or x >= w - 3 or y >= h - 3
            hits += 1 if (edge_d[yi, xi] > 0 or on_border) else 0
            total += 1
    return hits / total


def _contrast(q, gray):
    inside = np.zeros_like(gray, np.uint8)
    cv2.fillConvexPoly(inside, q.astype(np.int32), 255)
    out = cv2.bitwise_not(inside)
    ring = cv2.dilate(inside, np.ones((25, 25), np.uint8)) & out
    if ring.sum() == 0:
        return 0.0
    mi = cv2.mean(gray, mask=cv2.erode(inside, np.ones((9, 9), np.uint8)))[0]
    mo = cv2.mean(gray, mask=ring)[0]
    return min(abs(mi - mo) / 80.0, 1.0)


def _features(q, fit, gray, edge_d, p):
    frame = gray.shape[0] * gray.shape[1]
    area = quad_area(q) / frame
    if not is_convex(q) or area < p.min_area or area > p.max_area:
        return None
    angles = interior_angles(q)
    return {"edge": _edge_support(q, edge_d), "area": min(area / 0.5, 1.0), "contrast": _contrast(q, gray),
            "angle": float(np.mean([45 <= a <= 135 for a in angles])), "fit": fit}


def combine(feats, p: DetectParams, method: str = "") -> float:
    return (p.w_edge * feats["edge"] + p.w_area * feats["area"] + p.w_contrast * feats["contrast"]
            + p.w_angle * feats["angle"] + p.w_fit * feats["fit"] + p.method_bias.get(method.split(":")[0], 0.0))


def _bgdist_mask(small):
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    t = max(4, int(0.04 * min(h, w)))
    border = np.concatenate([lab[:t].reshape(-1, 3), lab[-t:].reshape(-1, 3),
                             lab[:, :t].reshape(-1, 3), lab[:, -t:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    spread = np.median(np.linalg.norm(border - bg, axis=1))
    d = np.linalg.norm(lab - bg, axis=2)
    d8 = np.clip(d / max(d.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    thr, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    return m


def _grabcut_mask(small):
    s = 400 / max(small.shape[:2])
    g = cv2.resize(small, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else small
    h, w = g.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(0.04 * w), int(0.04 * h), int(0.92 * w), int(0.92 * h))
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(g, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    return cv2.resize(m, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)


def detect_document(img_bgr: np.ndarray, params: DetectParams | None = None) -> Detection:
    p = params or DetectParams()
    h0, w0 = img_bgr.shape[:2]
    scale = min(1.0, p.work_size / max(h0, w0))
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr.copy()
    scale = small.shape[1] / w0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    frame_area = gray.shape[0] * gray.shape[1]
    edges = cv2.Canny(blur, 40, 120)
    edge_d = cv2.dilate(edges, np.ones((5, 5), np.uint8))

    cands = []
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    for inv, tag in ((0, "otsu_bright"), (1, "otsu_dark")):
        flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
        _, m = cv2.threshold(blur, 0, 255, flag | cv2.THRESH_OTSU)
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k), cv2.MORPH_OPEN, open_k)
        cands += _quads_from_mask(m, tag, frame_area)
    ad = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -4)
    ad = cv2.morphologyEx(ad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    cands += _quads_from_mask(ad, "adaptive", frame_area)
    e = cv2.morphologyEx(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2), cv2.MORPH_CLOSE, close_k)
    cands += _quads_from_mask(e, "canny", frame_area)
    if p.use_bgdist:
        cands += _quads_from_mask(_bgdist_mask(small), "bgdist", frame_area)
    if p.use_grabcut:
        gm = _grabcut_mask(small)
        if gm is not None:
            cands += _quads_from_mask(gm, "grabcut", frame_area)

    scored = []
    for q, name, fit in cands:
        f = _features(q, fit, gray, edge_d, p)
        if f is not None:
            scored.append({"score": combine(f, p, name), "method": name, "feats": f, "quad": (q / scale).astype(np.float32)})
    scored.sort(key=lambda d: -d["score"])
    if scored and scored[0]["score"] >= p.min_score:
        b = scored[0]
        return Detection(quad=b["quad"], score=float(b["score"]), method=b["method"], all_quads=scored)
    full = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], np.float32)
    return Detection(quad=full, score=0.0, method="fallback:full_frame", all_quads=scored)
