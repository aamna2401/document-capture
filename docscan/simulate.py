"""Phone-capture simulator with exact ground truth.

A flat document is placed in 3D, viewed by a pinhole camera (yaw, pitch, roll, distance, focal length),
then composited on a random background with lighting, blur, noise and JPEG compression. Because the camera
model is known, the true document corners are known exactly, which lets us score detection and rectification.
"""
from dataclasses import dataclass

import cv2
import numpy as np

CANVAS = (1200, 1600)  # width, height


@dataclass(frozen=True)
class Level:
    roll: float          # max |roll| in degrees
    tilt: float          # max |yaw|,|pitch| in degrees
    scale: tuple         # document height as a fraction of canvas height
    blur: float          # max gaussian sigma
    noise: float         # max gaussian noise sigma (0-255)
    shadow: float        # shadow strength 0-1
    rot90: float         # probability of an extra 90/180/270 rotation
    bg_contrast: str     # "random" or "low" (background similar in brightness to the document)


LEVELS = {
    "easy":   Level(roll=10, tilt=15, scale=(0.70, 0.85), blur=0.6, noise=4,  shadow=0.10, rot90=0.0, bg_contrast="random"),
    "medium": Level(roll=30, tilt=30, scale=(0.60, 0.80), blur=1.2, noise=8,  shadow=0.25, rot90=0.0, bg_contrast="random"),
    "hard":   Level(roll=45, tilt=40, scale=(0.50, 0.75), blur=2.0, noise=12, shadow=0.40, rot90=0.5, bg_contrast="low"),
}


def _rotation(yaw, pitch, roll):
    y, p, r = np.radians([yaw, pitch, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _background(rng, shape, doc_mean, contrast):
    h, w = shape
    if contrast == "low":
        base = np.clip(doc_mean + rng.uniform(-35, 35), 40, 235)
        color = np.clip(base + rng.uniform(-20, 20, 3), 0, 255)
    else:
        color = rng.uniform(20, 235, 3)
    kind = rng.integers(0, 3)
    bg = np.ones((h, w, 3), np.float32) * color
    if kind >= 1:  # gradient
        ang = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        g = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
        bg *= (0.8 + 0.4 * g[..., None])
    if kind == 2:  # low-frequency "desk" texture
        n = cv2.resize(rng.normal(0, 1, (h // 40 + 2, w // 40 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        bg += 18 * n[..., None]
    return np.clip(bg, 0, 255)


def capture(doc_bgr: np.ndarray, level: str, seed: int):
    """Return (image, gt_corners[tl,tr,br,bl as they appear in the image], info)."""
    L = LEVELS[level]
    rng = np.random.default_rng(seed)
    W, H = CANVAS
    dh, dw = doc_bgr.shape[:2]

    roll = rng.uniform(-L.roll, L.roll)
    if rng.random() < L.rot90:
        roll += float(rng.choice([90, 180, 270]))
    yaw, pitch = rng.uniform(-L.tilt, L.tilt, 2)

    # document in 3D: plane z=0, centred, scaled so its height is a fraction of the canvas
    s = rng.uniform(*L.scale) * H / dh
    pw, ph = dw * s, dh * s
    corners3d = np.array([[-pw / 2, -ph / 2, 0], [pw / 2, -ph / 2, 0], [pw / 2, ph / 2, 0], [-pw / 2, ph / 2, 0]], np.float64)
    R = _rotation(yaw, pitch, roll)
    f = rng.uniform(0.9, 1.6) * W
    dist = f * 0.85 + rng.uniform(0, 0.2) * f
    off = np.array([rng.uniform(-0.08, 0.08) * W, rng.uniform(-0.08, 0.08) * H, 0])
    cam = (R @ corners3d.T).T + np.array([0, 0, dist]) + off
    proj = np.stack([f * cam[:, 0] / cam[:, 2] + W / 2, f * cam[:, 1] / cam[:, 2] + H / 2], 1).astype(np.float32)

    src = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, proj)
    bg = _background(rng, (H, W), float(doc_bgr.mean()), L.bg_contrast)
    warped = cv2.warpPerspective(doc_bgr, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    mask = cv2.warpPerspective(np.full((dh, dw), 255, np.uint8), Hm, (W, H), flags=cv2.INTER_LINEAR)
    m = (cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0)[..., None]
    img = warped * m + bg * (1 - m)

    # lighting gradient and optional shadow band
    yy, xx = np.mgrid[0:H, 0:W]
    ang = rng.uniform(0, 2 * np.pi)
    light = 1.0 + 0.25 * rng.uniform(-1, 1) * (np.cos(ang) * (xx / W - 0.5) + np.sin(ang) * (yy / H - 0.5)) * 2
    if rng.random() < 0.6 and L.shadow > 0:
        c = rng.uniform(0.2, 0.8)
        band = np.exp(-(((np.cos(ang) * (xx / W) + np.sin(ang) * (yy / H)) - c) ** 2) / (2 * 0.08 ** 2))
        light *= 1 - L.shadow * rng.uniform(0.4, 1.0) * band
    img = img * light[..., None]

    sigma = rng.uniform(0, L.blur)
    if sigma > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img += rng.normal(0, rng.uniform(0, L.noise), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 92))])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    info = {"roll": float(roll), "yaw": float(yaw), "pitch": float(pitch), "scale": float(s), "level": level, "seed": seed,
            "doc_aspect": dw / dh}
    return img, proj, info
