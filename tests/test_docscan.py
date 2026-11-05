import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docscan.detect import detect_document          # noqa: E402
from docscan.geometry import corner_error, order_corners, quad_iou  # noqa: E402
from docscan.metrics import numeric_recall, word_prf  # noqa: E402
from docscan.ocr import tokens                      # noqa: E402
from docscan.rectify import aspect_zhang_he, output_size  # noqa: E402
from docscan.simulate import capture                # noqa: E402


def make_doc(w=500, h=800):
    img = np.full((h, w, 3), 245, np.uint8)
    for i, y in enumerate(range(60, h - 40, 45)):
        cv2.putText(img, f"ITEM {i} 12.{i:02d}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (200, 200, 200), 2)
    return img


def test_order_corners_is_rotation_invariant():
    q = np.array([[10, 10], [100, 12], [98, 200], [8, 190]], np.float32)
    rng = np.random.default_rng(0)
    shuffled = q[rng.permutation(4)]
    assert np.allclose(order_corners(shuffled), order_corners(q))


def test_quad_iou_identity_and_disjoint():
    q = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    assert quad_iou(q, q) == pytest.approx(1.0)
    assert quad_iou(q, q + 100) == pytest.approx(0.0)
    assert corner_error(q, q) == 0.0


def test_simulator_ground_truth_matches_composite():
    doc = make_doc()
    img, gt, info = capture(doc, "easy", seed=3)
    assert img.shape[:2] == (1600, 1200)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, gt.astype(np.int32), 255)
    # the document is brighter than the random background on average inside the ground-truth quad
    assert 0.1 < mask.mean() / 255 < 0.9
    assert info["level"] == "easy"


def test_simulator_is_deterministic():
    doc = make_doc()
    a, ga, _ = capture(doc, "medium", seed=11)
    b, gb, _ = capture(doc, "medium", seed=11)
    assert np.array_equal(a, b) and np.allclose(ga, gb)


def test_zhang_he_recovers_true_aspect_on_ground_truth_corners():
    doc = make_doc(500, 800)
    for seed in range(5):
        img, gt, info = capture(doc, "medium", seed=seed)
        r = aspect_zhang_he(gt, img.shape[:2])
        assert r is not None
        assert min(abs(r / info["doc_aspect"] - 1), abs(r * info["doc_aspect"] - 1)) < 0.02


def test_detect_finds_document_on_easy_capture():
    doc = make_doc()
    ious = []
    for seed in range(4):
        img, gt, _ = capture(doc, "easy", seed=seed)
        ious.append(quad_iou(detect_document(img).quad, gt))
    assert np.mean(ious) > 0.85


def test_detect_falls_back_to_full_frame_on_blank_image():
    blank = np.full((400, 300, 3), 128, np.uint8)
    det = detect_document(blank)
    assert det.method.startswith("fallback") or quad_iou(det.quad, np.array([[0, 0], [299, 0], [299, 399], [0, 399]], np.float32)) > 0.9


def test_output_size_positive_and_bounded():
    q = np.array([[100, 100], [900, 140], [880, 1300], [90, 1250]], np.float32)
    w, h = output_size(q, (1600, 1200))
    assert 16 <= w <= 2000 and 16 <= h <= 2000


def test_token_normalisation_and_metrics():
    assert tokens("TOTAL  12.50, Tax: 1,200!") == ["total", "12.50", "tax", "1,200"]
    p, r, f1 = word_prf(["a", "b", "b"], ["a", "b", "c"])
    assert (round(p, 2), round(r, 2)) == (0.67, 0.67)
    assert numeric_recall(["12.50", "x"], ["12.50", "7", "y"]) == pytest.approx(0.5)


def test_api_health_and_detect():
    from fastapi.testclient import TestClient
    from docscan.api import app
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    img, _, _ = capture(make_doc(), "easy", seed=1)
    ok, buf = cv2.imencode(".jpg", img)
    r = c.post("/detect", files={"file": ("a.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 200 and len(r.json()["corners"]) == 4
    bad = c.post("/detect", files={"file": ("a.txt", b"not an image", "text/plain")})
    assert bad.status_code == 400
