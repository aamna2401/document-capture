"""Rectification: quadrilateral -> upright rectangle, with the output aspect ratio recovered from the camera model.

Naively warping to the longest edges stretches a document seen in perspective. Zhang & He (2007) recover the
true width/height ratio of a rectangle from its four image corners, assuming square pixels and a principal point
at the image centre. We fall back to edge lengths when the estimate is degenerate (e.g. fronto-parallel views).
"""
import numpy as np

from .geometry import order_corners, side_lengths, warp_to_rect


def aspect_zhang_he(quad, image_size):
    """Estimate document width/height from corners. Returns None if degenerate."""
    h, w = image_size
    u0, v0 = w / 2.0, h / 2.0
    tl, tr, br, bl = order_corners(quad).astype(np.float64)
    m1, m2, m3, m4 = [np.array([x - u0, y - v0, 1.0]) for x, y in (tl, tr, bl, br)]
    k2n = np.dot(np.cross(m1, m4), m3)
    k2d = np.dot(np.cross(m2, m4), m3)
    k3n = np.dot(np.cross(m1, m4), m2)
    k3d = np.dot(np.cross(m3, m4), m2)
    if abs(k2d) < 1e-9 or abs(k3d) < 1e-9:
        return None
    k2, k3 = k2n / k2d, k3n / k3d
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    if abs(n2[2] * n3[2]) < 1e-6:        # parallel sides: no perspective, focal length unobservable
        return None
    f2 = -((n2[0] * n3[0] + n2[1] * n3[1]) / (n2[2] * n3[2]))
    if not np.isfinite(f2) or f2 <= 0:
        return None
    f = np.sqrt(f2)
    A_inv = np.array([[1 / f, 0, 0], [0, 1 / f, 0], [0, 0, 1]])
    num = n2 @ A_inv.T @ A_inv @ n2
    den = n3 @ A_inv.T @ A_inv @ n3
    if den <= 0 or num <= 0:
        return None
    ratio = float(np.sqrt(num / den))
    return ratio if 0.05 < ratio < 20 else None


def output_size(quad, image_size, mode="zhang", max_side=2000):
    top, right, bottom, left = side_lengths(order_corners(quad))
    w_edge, h_edge = max(top, bottom), max(left, right)
    ratio = None
    if mode == "zhang":
        ratio = aspect_zhang_he(quad, image_size)
    edge_ratio = w_edge / max(h_edge, 1e-6)
    # Zhang-He amplifies corner noise: distrust it when it strongly disagrees with the edge-length ratio
    if ratio is not None and not (0.8 <= ratio / edge_ratio <= 1.25):
        ratio = None
    if ratio is None:
        w, h = w_edge, h_edge
    else:
        # keep the pixel budget of the edge-length estimate but impose the recovered aspect ratio
        area = w_edge * h_edge
        h = (area / ratio) ** 0.5
        w = h * ratio
    s = min(1.0, max_side / max(w, h))
    return max(int(round(w * s)), 16), max(int(round(h * s)), 16)


def rectify(img, quad, aspect="zhang"):
    size = output_size(quad, img.shape[:2], aspect)
    return warp_to_rect(img, quad, size), size
