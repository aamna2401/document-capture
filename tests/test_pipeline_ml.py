"""CPU-only tests for the ML-engineering parts: metrics, leakage-safe splits, crop building, run bookkeeping."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from docextract import runmeta                                          # noqa: E402
from docextract.crops import cord_items, funsd_items, split_forms, write_dataset  # noqa: E402
from docextract.metrics import alnum_tokens, prf, text_scores         # noqa: E402

FUNSD_TRAIN = ROOT / "data/public/funsd_raw/dataset/training_data"
needs_funsd = pytest.mark.skipif(not FUNSD_TRAIN.exists(), reason="FUNSD not downloaded")


def test_alnum_f1_ignores_punctuation_but_strict_does_not():
    s = text_scores("prentice-hall new york", "prentice hall new york")
    assert s["f1"] == 1.0 and s["strict_f1"] < 1.0


def test_prf_known_values():
    p, r, f = prf(["a", "b", "c"], ["a", "b", "d", "e"])
    assert (round(p, 3), round(r, 3)) == (0.667, 0.5) and round(f, 3) == 0.571
    assert alnum_tokens("Total: 12.50 USD") == ["total", "12", "50", "usd"]


def test_split_is_deterministic_disjoint_and_covers_all():
    ids = [f"f{i}" for i in range(100)]
    a, b = split_forms(ids, 0.15, seed=3), split_forms(ids, 0.15, seed=3)
    assert a == b and not (a[0] & a[1]) and a[0] | a[1] == set(ids) and len(a[1]) == 15
    assert split_forms(ids, 0.15, seed=4) != a


@needs_funsd
def test_funsd_crops_come_from_training_only_and_split_by_form(tmp_path):
    with pytest.raises(AssertionError):
        list(funsd_items(FUNSD_TRAIN.parent / "testing_data"))
    items = list(funsd_items(FUNSD_TRAIN))
    assert len(items) > 1000
    counts = write_dataset(items, tmp_path, seed=0)
    assert counts["train"] > counts["val"] > 0
    rows = {s: [json.loads(l) for l in open(tmp_path / f"labels_{s}.jsonl")] for s in ("train", "val")}
    assert not ({r["form"] for r in rows["train"]} & {r["form"] for r in rows["val"]})
    assert all(1 <= len(r["text"]) <= 64 for r in rows["train"])


def test_cord_crops_from_local_val_parquet():
    pq = ROOT / "data/cord_val.parquet"
    if not pq.exists():
        pytest.skip("cord parquet missing")
    import pandas as pd
    rows = pd.read_parquet(pq).head(5).to_dict("records")
    items = list(cord_items(rows))
    assert items and all(c.ndim == 3 and t.strip() for _, c, t in items)


def test_runmeta_records_config_and_seed(tmp_path):
    m = runmeta.save(tmp_path, {"a": 1}, seed=7)
    assert m["seed"] == 7 and len(m["config_sha256"]) == 16 and (tmp_path / "meta.json").exists()
    assert runmeta.collect({"a": 1}, 7)["config_sha256"] == m["config_sha256"]
    assert runmeta.collect({"a": 2}, 7)["config_sha256"] != m["config_sha256"]


def test_run_all_end_to_end_and_resume(tmp_path):
    """Two tiny gold files, RapidOCR only: results, summary and metadata are written, and a rerun skips finished files."""
    import cv2
    from test_docextract import LINES, _text_image
    gold = tmp_path / "gold"; gold.mkdir()
    for i in range(2):
        cv2.imwrite(str(gold / f"d{i}.png"), _text_image(LINES))
        (gold / f"d{i}.txt").write_text("\n".join(LINES))
    cfg = {"seed": 0, "run_name": "t", "datasets": {"toy": str(gold)}, "engines": {"rapidocr": {"name": "rapidocr"}},
           "capture": False, "target_f1": 0.9, "bootstrap": {"n": 50, "ci": 0.95}}
    cpath = tmp_path / "c.yaml"; cpath.write_text(yaml.safe_dump(cfg))
    cmd = [sys.executable, str(ROOT / "scripts/run_all.py"), str(cpath), "--out", str(tmp_path / "runs"), "--run-id", "r1"]
    subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
    run = tmp_path / "runs" / "r1"
    assert {"meta.json", "config.yaml", "summary.csv", "results_toy__rapidocr.csv"} <= {p.name for p in run.iterdir()}
    first = (run / "results_toy__rapidocr.csv").read_text()
    subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
    assert (run / "results_toy__rapidocr.csv").read_text() == first          # resumed: nothing re-run or duplicated
    import pandas as pd
    s = pd.read_csv(run / "summary.csv")
    assert s.loc[0, "files"] == 2 and s.loc[0, "f1"] > 0.8
