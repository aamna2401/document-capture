"""Small geometry helpers: corner ordering, quad metrics, homography utilities."""
import cv2
import numpy as np


def order_corners(pts) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)


def quad_area(q) -> float:
    return float(abs(cv2.contourArea(np.asarray(q, dtype=np.float32).reshape(-1, 1, 2))))


def is_convex(q) -> bool:
    return bool(cv2.isContourConvex(np.asarray(q, dtype=np.float32).reshape(-1, 1, 2)))


def side_lengths(q):
    q = np.asarray(q, dtype=np.float32)
    return [float(np.linalg.norm(q[i] - q[(i + 1) % 4])) for i in range(4)]


def interior_angles(q):
    q = np.asarray(q, dtype=np.float64)
    out = []
    for i in range(4):
        a, b, c = q[i - 1] - q[i], q[(i + 1) % 4] - q[i], None
        cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        out.append(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
    return out


def quad_iou(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = quad_area(a) + quad_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def corner_error(pred, gt) -> float:
    """Mean corner distance divided by the ground-truth diagonal (0 = perfect)."""
    pred, gt = order_corners(pred), order_corners(gt)
    diag = np.linalg.norm(gt[0] - gt[2])
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)) / diag)


def warp_to_rect(img, quad, size):
    """Perspective-warp the quad (tl,tr,br,bl) to an upright rectangle of size (w, h)."""
    w, h = int(round(size[0])), int(round(size[1]))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(order_corners(quad), dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
