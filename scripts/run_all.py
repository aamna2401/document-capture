"""Run every engine on every document type from a config, and save per-file results, a summary and run metadata.

  python scripts/run_all.py configs/experiment.yaml [--only-engines rapidocr,tesseract] [--only-types forms] [--out outputs/runs]
Resumable: finished (type, engine, file) rows are skipped when the same run directory is reused (--run-id).
Writes <out>/<run_id>/{meta.json, config.yaml, results_<type>_<engine>.csv, summary.csv}.
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docextract import extract, runmeta               # noqa: E402
from docextract.engines import get_engine             # noqa: E402
from docextract.metrics import text_scores            # noqa: E402

EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FIELDS = ["file", "f1", "precision", "recall", "strict_f1", "cer", "wer", "seconds", "pages"]


def gold_files(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS and p.with_suffix(".txt").exists())


def bootstrap_ci(x, n=1000, ci=0.95, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return float(np.quantile(means, (1 - ci) / 2)), float(np.quantile(means, 1 - (1 - ci) / 2))


def run_one(files, engine, capture, out_csv: Path):
    done = set(pd.read_csv(out_csv).file) if out_csv.exists() else set()
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for p in files:
            if p.name in done:
                continue
            t = time.time()
            doc = extract(p, ocr_engine=engine, capture=capture)
            sc = text_scores(doc.text, p.with_suffix(".txt").read_text())
            w.writerow({"file": p.name, **{k: round(sc[k], 4) for k in ("f1", "precision", "recall", "strict_f1", "cer", "wer")},
                        "seconds": round(time.time() - t, 2), "pages": len(doc.pages)})
            f.flush()                                      # a disconnect loses at most the current file


def summarise(run_dir: Path, cfg: dict):
    rows = []
    for csv_path in sorted(run_dir.glob("results_*.csv")):
        dtype, engine = csv_path.stem[len("results_"):].split("__", 1)
        d = pd.read_csv(csv_path)
        lo, hi = bootstrap_ci(d.f1, **cfg["bootstrap"], seed=cfg["seed"])
        rows.append({"type": dtype, "engine": engine, "files": len(d), "f1": d.f1.mean(), "f1_ci_lo": lo, "f1_ci_hi": hi,
                     "strict_f1": d.strict_f1.mean(), "recall": d.recall.mean(), "cer": d.cer.mean(),
                     f"share_f1>={cfg['target_f1']}": (d.f1 >= cfg["target_f1"]).mean(), "sec_per_file": d.seconds.mean()})
    s = pd.DataFrame(rows).round(4)
    s.to_csv(run_dir / "summary.csv", index=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="outputs/runs")
    ap.add_argument("--run-id", default=None, help="reuse a run directory to resume it")
    ap.add_argument("--only-engines", default=None)
    ap.add_argument("--only-types", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    run_dir = Path(a.out) / (a.run_id or f"{cfg['run_name']}_{time.strftime('%Y%m%d_%H%M%S')}")
    runmeta.save(run_dir, cfg, cfg["seed"])
    shutil.copy(a.config, run_dir / "config.yaml")
    engines = {k: v for k, v in cfg["engines"].items() if not a.only_engines or k in a.only_engines.split(",")}
    types = {k: v for k, v in cfg["datasets"].items() if not a.only_types or k in a.only_types.split(",")}
    print(f"run dir: {run_dir}\n{len(types)} types x {len(engines)} engines")
    for label, spec in engines.items():
        try:
            engine = get_engine(spec["name"], **spec.get("options", {}))
        except Exception as e:                             # a missing optional dependency must not kill the whole run
            print(f"SKIP engine {label}: {type(e).__name__}: {e}")
            continue
        for dtype, folder in types.items():
            files = gold_files(Path(folder))
            print(f"[{dtype} / {label}] {len(files)} files", flush=True)
            run_one(files, engine, cfg["capture"], run_dir / f"results_{dtype}__{label}.csv")
    print(summarise(run_dir, cfg).to_string(index=False))
    print("saved to", run_dir)


if __name__ == "__main__":
    main()
