"""Build the recognizer training set (line crops + labels) from FUNSD train and CORD train.

  python scripts/prepare_crops.py [--cord-train path.parquet] [--out data/crops] [--seed 0]
CORD train is not downloaded by default; on Colab: datasets.load_dataset('naver-clova-ix/cord-v2', split='train').to_parquet(path).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docextract.crops import cord_items, funsd_items, write_dataset   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funsd", default=str(ROOT / "data/public/funsd_raw/dataset/training_data"))
    ap.add_argument("--cord-train", default=None, help="parquet of the CORD-v2 TRAIN split")
    ap.add_argument("--out", default=str(ROOT / "data/crops"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items = list(funsd_items(Path(a.funsd)))
    print(f"funsd: {len(items)} line crops")
    if a.cord_train:
        import pandas as pd
        rows = pd.read_parquet(a.cord_train).to_dict("records")
        c = list(cord_items(rows))
        print(f"cord:  {len(c)} line crops from {len(rows)} receipts")
        items += c
    print("written:", write_dataset(items, Path(a.out), seed=a.seed), "->", a.out)


if __name__ == "__main__":
    main()
