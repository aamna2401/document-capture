"""Turn the evaluation CSVs into tables (outputs/results/*.md|json) and figures (outputs/figures/*.png).

Usage: python scripts/make_report.py
Uncertainty: 95% bootstrap intervals resampling DOCUMENTS (captures of one document are not independent).
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES, FIG = ROOT / "outputs" / "results", ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LEVELS = ["easy", "medium", "hard"]
VARIANTS = ["raw", "rect", "rect_ori", "rect_full", "oracle"]
LABEL = {"raw": "Raw photo", "rect": "Rectified", "rect_ori": "+ orientation", "rect_full": "+ deskew (full)", "oracle": "Oracle corners"}
ENGINES = ["tesseract", "rapidocr"]
COL = {"raw": "#b0b0b0", "rect": "#8fb4e3", "rect_ori": "#4f86c6", "rect_full": "#2f5d9e", "oracle": "#2e8b57"}
rng = np.random.default_rng(0)


def boot_ci(df, col, group="doc", n=2000):
    """Mean and 95% CI, resampling groups (documents)."""
    g = df.groupby(group)[col].mean()
    v = g.to_numpy()
    means = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta(df, a, b, group="doc", n=2000):
    d = (df[a] - df[b]).groupby(df[group]).mean().to_numpy()
    means = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sim_report():
    df = pd.read_csv(RES / "sim_results.csv")
    out = {"n_captures": len(df), "n_docs": int(df.doc.nunique())}
    # ---- geometry
    geo = []
    for lvl in LEVELS:
        d = df[df.level == lvl]
        geo.append({"level": lvl, "n": len(d), "mean_iou": d.iou.mean(), "iou_gt_0.9_%": 100 * (d.iou > 0.9).mean(),
                    "iou_gt_0.8_%": 100 * (d.iou > 0.8).mean(), "median_corner_err_%": 100 * d.corner_err.median(),
                    "fallback_%": 100 * d.method.str.startswith("fallback").mean(),
                    "aspect_err_zhang_med_%": 100 * d.aspect_err_zhang.median(), "aspect_err_edges_med_%": 100 * d.aspect_err_edges.median(),
                    "orientation_acc_%": 100 * (d.k_pred == d.k_true).mean()})
    geo = pd.DataFrame(geo).round(2)
    geo.to_csv(RES / "geometry_summary.csv", index=False)
    out["geometry"] = geo.to_dict("records")
    # ---- OCR by level / engine / variant
    rows = []
    for eng in ENGINES:
        for lvl in LEVELS + ["all"]:
            d = df if lvl == "all" else df[df.level == lvl]
            for v in VARIANTS:
                for met in ("f1", "num"):
                    m, lo, hi = boot_ci(d, f"{v}__{eng}__{met}")
                    rows.append({"engine": eng, "level": lvl, "variant": v, "metric": met, "mean": m, "lo": lo, "hi": hi})
    ocr = pd.DataFrame(rows)
    ocr.to_csv(RES / "ocr_by_variant.csv", index=False)
    # ---- paired deltas for the headline claims
    deltas = []
    for eng in ENGINES:
        for lvl in LEVELS + ["all"]:
            d = df if lvl == "all" else df[df.level == lvl]
            for a, b in (("rect_full", "raw"), ("rect_ori", "rect"), ("rect_full", "rect_ori"), ("oracle", "rect_full")):
                m, lo, hi = paired_delta(d, f"{a}__{eng}__f1", f"{b}__{eng}__f1")
                deltas.append({"engine": eng, "level": lvl, "comparison": f"{a} - {b}", "delta_f1": m, "lo": lo, "hi": hi})
    pd.DataFrame(deltas).to_csv(RES / "paired_deltas.csv", index=False)
    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, eng in zip(axes, ENGINES):
        w = 0.16
        for i, v in enumerate(VARIANTS):
            vals = [ocr[(ocr.engine == eng) & (ocr.level == l) & (ocr.variant == v) & (ocr.metric == "f1")].iloc[0] for l in LEVELS]
            ax.bar(np.arange(3) + (i - 2) * w, [x["mean"] for x in vals], w, yerr=[[x["mean"] - x["lo"] for x in vals], [x["hi"] - x["mean"] for x in vals]],
                   color=COL[v], label=LABEL[v], capsize=2, error_kw={"lw": 0.8})
        ax.set_xticks(range(3)); ax.set_xticklabels(LEVELS); ax.set_title(f"{eng}: word F1 by capture difficulty"); ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("word F1 (higher is better)"); axes[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "ocr_f1_by_level.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for l, c in zip(LEVELS, ["#4f86c6", "#e0a030", "#c0504d"]):
        d = df[df.level == l]
        ax[0].hist(d.iou, bins=np.linspace(0, 1, 26), alpha=0.6, color=c, label=l)
        ax[1].hist(100 * d.corner_err.clip(upper=0.3), bins=30, alpha=0.6, color=c, label=l)
    ax[0].set_xlabel("IoU of detected vs true quad"); ax[0].legend(); ax[1].set_xlabel("mean corner error (% of diagonal, clipped at 30)")
    tilt = np.sqrt(df.yaw ** 2 + df.pitch ** 2)
    ax[2].scatter(tilt, df.iou, s=8, alpha=0.4, color="#2f5d9e"); ax[2].set_xlabel("camera tilt (deg)"); ax[2].set_ylabel("IoU")
    for a in ax: a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "detection_quality.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    roll = ((df.roll + 180) % 360) - 180
    ax.scatter(roll.abs(), df["rect_full__rapidocr__f1"] - df["raw__rapidocr__f1"], s=8, alpha=0.4, color="#2f5d9e", label="rapidocr")
    ax.scatter(roll.abs(), df["rect_full__tesseract__f1"] - df["raw__tesseract__f1"], s=8, alpha=0.4, color="#e0a030", label="tesseract")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("|in-plane rotation| (deg)"); ax.set_ylabel("F1 gain from the pipeline"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "gain_vs_rotation.png", dpi=140); plt.close()
    out["worst_cases"] = df.nsmallest(8, "iou")[["doc", "level", "k", "iou", "method"]].to_dict("records")
    json.dump(out, open(RES / "sim_summary.json", "w"), indent=1, default=float)
    return df, geo, ocr, pd.DataFrame(deltas)


def native_report():
    p = RES / "native_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["doc"] = df["id"]
    rows = []
    for eng in ENGINES:
        for v in ("raw", "rect", "rect_ori", "rect_full"):
            for met in ("f1", "num"):
                m, lo, hi = boot_ci(df, f"{v}__{eng}__{met}", group="doc")
                rows.append({"engine": eng, "variant": v, "metric": met, "mean": m, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(RES / "native_by_variant.csv", index=False)
    dd = []
    for eng in ENGINES:
        m, lo, hi = paired_delta(df, f"rect_full__{eng}__f1", f"raw__{eng}__f1", group="doc")
        dd.append({"engine": eng, "delta_f1_full_vs_raw": m, "lo": lo, "hi": hi,
                   "improved_%": 100 * (df[f"rect_full__{eng}__f1"] > df[f"raw__{eng}__f1"] + 0.02).mean(),
                   "worse_%": 100 * (df[f"rect_full__{eng}__f1"] < df[f"raw__{eng}__f1"] - 0.02).mean()})
    pd.DataFrame(dd).to_csv(RES / "native_deltas.csv", index=False)
    roi = df[df.has_roi]
    summ = {"n": len(df), "n_with_roi": len(roi), "mean_iou_vs_roi": float(roi.iou_roi.mean()) if len(roi) else None,
            "iou_roi_gt_0.8_%": float(100 * (roi.iou_roi > 0.8).mean()) if len(roi) else None,
            "fallback_%": float(100 * df.method.str.startswith("fallback").mean())}
    json.dump({"summary": summ, "deltas": dd}, open(RES / "native_summary.json", "w"), indent=1, default=float)
    return df, res, pd.DataFrame(dd), summ


def example_figures():
    """Before/after panels re-generated from fixed seeds (test documents)."""
    from docscan.data import load_cord, paper_crop
    from docscan.pipeline import scan
    from docscan.simulate import capture
    docs = [c for s in load_cord("test") if (c := paper_crop(s))]
    picks = [(0, "easy", 1), (3, "medium", 2), (7, "hard", 0), (11, "hard", 3), (15, "medium", 1), (20, "hard", 2)]
    cols = []
    for di, lvl, k in picks:
        doc, _ = docs[di % len(docs)]
        img, gt, _ = capture(doc, lvl, seed=200000 + (di % len(docs)) * 1000 + k)
        res = scan(img)
        a = img.copy()
        cv2.polylines(a, [gt.astype(np.int32)], True, (0, 200, 0), 5)
        cv2.polylines(a, [res.detection.quad.astype(np.int32)], True, (0, 0, 255), 5)
        cols.append((lvl, cv2.cvtColor(a, cv2.COLOR_BGR2RGB), cv2.cvtColor(res.image, cv2.COLOR_BGR2RGB)))
    fig, ax = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 7.2))
    for j, (lvl, a, b) in enumerate(cols):
        ax[0, j].imshow(a); ax[0, j].set_title(f"{lvl}: capture", fontsize=9); ax[1, j].imshow(b); ax[1, j].set_title("scan", fontsize=9)
        for r in (0, 1): ax[r, j].axis("off")
    fig.suptitle("Green = true document corners, red = detected. Bottom row = pipeline output", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "before_after_examples.png", dpi=130); plt.close()


if __name__ == "__main__":
    sim_report()
    native_report()
    example_figures()
    print("report written to", RES, "and", FIG)
