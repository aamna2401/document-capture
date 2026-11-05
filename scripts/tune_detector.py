"""Tune the candidate-scoring weights on the VALIDATION split only (never the test split).

Caches every candidate's features and its true IoU for simulated validation captures, then random-searches the
weights. Documents are split in two halves: fit on one, verify on the other.
Usage: python scripts/tune_detector.py   -> writes outputs/results/detector_tuning.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docscan.data import load_cord, paper_crop          # noqa: E402
from docscan.detect import FEATURES, detect_document     # noqa: E402
from docscan.geometry import quad_iou                    # noqa: E402
from docscan.simulate import capture                     # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "results"
CACHE = OUT / "tuning_cache.pkl"
METHODS = ["otsu_bright", "otsu_dark", "adaptive", "canny", "bgdist", "grabcut"]
SEEDS_PER_DOC = 6


def build_cache():
    docs = [c for s in load_cord("val") if (c := paper_crop(s))]
    rows = []
    for di, (doc, _) in enumerate(docs):
        for lvl in ("easy", "medium", "hard"):
            for k in range(SEEDS_PER_DOC):
                img, gt, _ = capture(doc, lvl, seed=7000 + di * 100 + k)
                det = detect_document(img)
                if not det.all_quads:
                    continue
                X = np.array([[c["feats"][f] for f in FEATURES] + [c["method"].split(":")[0] == m for m in METHODS]
                              for c in det.all_quads], float)
                rows.append({"doc": di, "level": lvl, "X": X, "iou": np.array([quad_iou(c["quad"], gt) for c in det.all_quads])})
    return rows


def pick_iou(rows, w, thr):
    out = []
    for r in rows:
        s = r["X"] @ w
        i = int(np.argmax(s))
        out.append(r["iou"][i] if s[i] >= thr else 0.0)     # 0.0 stands in for the full-frame fallback (IoU ~0 vs a small doc)
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pickle.load(open(CACHE, "rb")) if CACHE.exists() else build_cache()
    pickle.dump(rows, open(CACHE, "wb"))
    ndoc = max(r["doc"] for r in rows) + 1
    A = [r for r in rows if r["doc"] % 2 == 0]
    B = [r for r in rows if r["doc"] % 2 == 1]
    dim = len(FEATURES) + len(METHODS)
    rng = np.random.default_rng(0)
    base = np.array([0.30, 0.10, 0.20, 0.10, 0.30] + [0.0] * len(METHODS))
    print("default weights  fit-half %.3f  check-half %.3f" % (pick_iou(A, base, 0.35).mean(), pick_iou(B, base, 0.35).mean()))
    best, best_s = base, pick_iou(A, base, 0.35).mean()
    for _ in range(6000):
        w = np.concatenate([rng.dirichlet(np.ones(len(FEATURES))), rng.normal(0, 0.06, len(METHODS))])
        s = pick_iou(A, w, 0.30).mean()
        if s > best_s:
            best, best_s = w, s
    print("tuned weights    fit-half %.3f  check-half %.3f" % (best_s, pick_iou(B, best, 0.30).mean()))
    # refit on all validation docs for the final weights
    best_all, best_all_s = best, pick_iou(rows, best, 0.30).mean()
    for _ in range(3000):
        w = best_all + rng.normal(0, 0.03, dim)
        s = pick_iou(rows, w, 0.30).mean()
        if s > best_all_s:
            best_all, best_all_s = w, s
    print("final (all val)  mean IoU %.3f  | IoU>0.9: %.0f%%" % (best_all_s, 100 * (pick_iou(rows, best_all, 0.30) > 0.9).mean()))
    for lvl in ("easy", "medium", "hard"):
        sub = [r for r in rows if r["level"] == lvl]
        print(f"  {lvl}: default {pick_iou(sub, base, 0.35).mean():.3f} -> tuned {pick_iou(sub, best_all, 0.30).mean():.3f}")
    json.dump({"features": FEATURES, "methods": METHODS, "weights": best_all.tolist(), "threshold": 0.30,
               "default_mean_iou": float(pick_iou(rows, base, 0.35).mean()), "tuned_mean_iou": float(best_all_s)},
              open(OUT / "detector_tuning.json", "w"), indent=1)


if __name__ == "__main__":
    main()
