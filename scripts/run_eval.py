"""Full evaluation on the TEST split (the detector was tuned on the validation split only).

  python scripts/run_eval.py sim      # simulated phone captures with exact ground truth (3 difficulty levels)
  python scripts/run_eval.py native   # the dataset's own photos (uncontrolled, real)

Variants OCR'd for each capture (Tesseract and RapidOCR):
  raw        the capture as is
  rect       detect + perspective rectification only
  rect_ori   + page orientation (0/90/180/270)
  rect_full  + residual deskew        <- the full pipeline
  oracle     rectified with the TRUE corners (upper bound for geometry)
  clean      the original flat document (upper bound for OCR)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2                                   # noqa: E402
import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402
from multiprocessing import get_context      # noqa: E402

from docscan.data import load_cord, paper_crop                    # noqa: E402
from docscan.detect import detect_document                        # noqa: E402
from docscan.geometry import corner_error, order_corners, quad_iou  # noqa: E402
from docscan.metrics import score_text                            # noqa: E402
from docscan.ocr import Rapid, Tesseract                          # noqa: E402
from docscan.orient import best_rotation, deskew, estimate_skew, rotate_k  # noqa: E402
from docscan.rectify import output_size, rectify                  # noqa: E402
from docscan.simulate import LEVELS, capture                      # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "results"
SEEDS_PER_DOC = 6
_G = {}


def _init(split):
    cv2.setNumThreads(1)
    if split == "sim":
        _G["docs"] = [c for s in load_cord("test") if (c := paper_crop(s))]
    else:
        _G["samples"] = list(load_cord("test"))
    _G["engines"] = [Tesseract(), Rapid()]


def _ocr_all(prefix, img, gt_tokens, row, engines):
    for e in engines:
        t = time.time()
        sc = score_text(e.read(img), gt_tokens)
        row[f"{prefix}__{e.name}__f1"] = sc["f1"]
        row[f"{prefix}__{e.name}__num"] = sc["num_recall"]
        row[f"{prefix}__{e.name}__rec"] = sc["recall"]
        row[f"{prefix}__{e.name}__prec"] = sc["precision"]


def _needed_k(det_quad, gt_doc_order):
    """Quarter-turns needed to make the rectified image upright (rotate_k convention, 1 = clockwise)."""
    q = order_corners(det_quad)
    j = int(np.argmin([np.linalg.norm(q[i] - gt_doc_order[0]) for i in range(4)]))   # where the document's top-left landed
    return (-j) % 4


def sim_task(args):
    di, level, k = args
    doc, toks = _G["docs"][di]
    engines = _G["engines"]
    img, gt, info = capture(doc, level, seed=200000 + di * 1000 + k)
    row = {"doc": di, "level": level, "k": k, "roll": info["roll"], "yaw": info["yaw"], "pitch": info["pitch"]}

    t0 = time.time()
    det = detect_document(img)
    row["t_detect"] = time.time() - t0
    row["iou"], row["corner_err"] = quad_iou(det.quad, gt), corner_error(det.quad, gt)
    row["method"], row["det_score"] = det.method, det.score
    # aspect-ratio accuracy of the two sizing modes (compare with the true document aspect, orientation-agnostic)
    for mode in ("zhang", "edges"):
        w, h = output_size(det.quad, img.shape[:2], mode)
        r, a = w / h, info["doc_aspect"]
        row[f"aspect_err_{mode}"] = min(abs(r / a - 1), abs(r * a - 1))

    flat, _ = rectify(img, det.quad, "zhang")
    k_pred, _ = best_rotation(flat)
    row["k_pred"], row["k_true"] = k_pred, _needed_k(det.quad, gt)
    flat_o = rotate_k(flat, k_pred)
    skew = estimate_skew(flat_o)
    flat_full = deskew(flat_o, skew)
    # oracle: the same rectification (same sizing rule) but with the TRUE corners and the TRUE orientation
    oracle, _ = rectify(img, gt, "zhang")
    oracle = rotate_k(oracle, _needed_k(gt, gt))

    for name, im in (("raw", img), ("rect", flat), ("rect_ori", flat_o), ("rect_full", flat_full), ("oracle", oracle)):
        _ocr_all(name, im, toks, row, engines)
    if k == 0:
        _ocr_all("clean", doc, toks, row, engines)
    return row


def native_task(i):
    s = _G["samples"][i]
    engines = _G["engines"]
    img, toks = s["image"], s["tokens_all"]
    row = {"id": s["id"], "has_roi": s["roi"] is not None}
    det = detect_document(img)
    row["method"], row["det_score"] = det.method, det.score
    row["frac_area"] = float(cv2.contourArea(det.quad.reshape(-1, 1, 2)) / (img.shape[0] * img.shape[1]))
    if s["roi"] is not None:
        row["iou_roi"] = quad_iou(det.quad, s["roi"])
    flat, _ = rectify(img, det.quad, "zhang")
    k_pred, _ = best_rotation(flat)
    flat_o = rotate_k(flat, k_pred)
    flat_full = deskew(flat_o, estimate_skew(flat_o))
    row["k_pred"] = k_pred
    for name, im in (("raw", img), ("rect", flat), ("rect_ori", flat_o), ("rect_full", flat_full)):
        _ocr_all(name, im, toks, row, engines)
    return row


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = get_context("spawn")
    t0 = time.time()
    if mode == "sim":
        ndocs = len([1 for s in load_cord("test") if paper_crop(s)])
        tasks = [(d, lvl, k) for d in range(ndocs) for lvl in LEVELS for k in range(SEEDS_PER_DOC)]
        fn, path = sim_task, OUT / "sim_results.csv"
    else:
        tasks, fn, path = list(range(100)), native_task, OUT / "native_results.csv"
    print(f"{mode}: {len(tasks)} tasks on 8 workers", flush=True)
    rows = []
    with ctx.Pool(8, initializer=_init, initargs=(mode,)) as pool:
        for n, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=2), 1):
            rows.append(r)
            if n % 40 == 0:
                print(f"  {n}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
