"""Page orientation (0/90/180/270) and residual skew."""
import cv2
import numpy as np

from .ocr import word_confidence_score

_ROT = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate_k(img, k):
    return img if _ROT[k % 4] is None else cv2.rotate(img, _ROT[k % 4])


def best_rotation(img):
    """Return (k, scores). Picks the quarter-turn whose text OCRs best. Ties favour the current orientation."""
    scores = [word_confidence_score(rotate_k(img, k)) for k in range(4)]
    best = int(np.argmax(scores))
    if scores[best] <= scores[0] * 1.15:
        best = 0
    return best, scores


def estimate_skew(img, max_angle=4.0, step=0.5, width=600):
    """Projection-profile skew estimate in degrees (positive = rotate counter-clockwise to fix)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = width / g.shape[1]
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    _, b = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h, w = b.shape
    best, best_a, base = -1.0, 0.0, None
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_NEAREST)
        v = float(np.var(r.sum(axis=1)))
        if abs(a) < 1e-9:
            base = v
        if v > best:
            best, best_a = v, float(a)
    # only act when the gain over "no rotation" is clear
    return best_a if base and best > base * 1.05 else 0.0


def deskew(img, angle):
    if abs(angle) < 1e-6:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
