"""Line-crop datasets for training the recognizer, with a seeded split BY FORM (no form appears in both splits).

Sources: FUNSD training split (field boxes) and CORD train split (line boxes). Test splits are never read here.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np

MAX_CHARS = 64          # TrOCR is a line model; longer fields are multi-line paragraphs
MIN_ASPECT = 1.5        # width/height; filters out tall or square boxes that are not single lines
PAD = 2


def _crop(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(int(x0) - PAD, 0), max(int(y0) - PAD, 0), min(int(x1) + PAD, w), min(int(y1) + PAD, h)
    return img[y0:y1, x0:x1] if x1 - x0 > 4 and y1 - y0 > 4 else None


def _keep(text: str, crop) -> bool:
    if crop is None or not text.strip() or len(text) > MAX_CHARS:
        return False
    return crop.shape[1] / crop.shape[0] >= MIN_ASPECT


def funsd_items(funsd_dir: Path):
    """Yield (form_id, crop, text) from FUNSD <dir>/training_data. `funsd_dir` must be the TRAINING folder."""
    assert funsd_dir.name == "training_data", "refusing to read non-training data for training crops"
    for ann in sorted((funsd_dir / "annotations").glob("*.json")):
        img = cv2.imread(str(funsd_dir / "images" / (ann.stem + ".png")))
        if img is None:
            continue
        for f in json.load(open(ann))["form"]:
            x0, y0, x1, y1 = f["box"]
            text = f["text"].strip()
            crop = _crop(img, x0, y0, x1, y1)
            if _keep(text, crop):
                yield ann.stem, crop, text


def cord_items(rows, prefix: str = "cord"):
    """Yield (form_id, crop, text) from CORD rows (dicts with 'image' {'bytes'} and 'ground_truth' json string)."""
    import io
    from PIL import Image
    for i, row in enumerate(rows):
        img = cv2.cvtColor(np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")), cv2.COLOR_RGB2BGR)
        for line in json.loads(row["ground_truth"])["valid_line"]:
            xs = [w["quad"][f"x{k}"] for w in line["words"] for k in range(1, 5)]
            ys = [w["quad"][f"y{k}"] for w in line["words"] for k in range(1, 5)]
            if not xs:
                continue
            text = " ".join(w["text"] for w in line["words"]).strip()
            crop = _crop(img, min(xs), min(ys), max(xs), max(ys))
            if _keep(text, crop):
                yield f"{prefix}_{i}", crop, text


def split_forms(form_ids, val_fraction: float = 0.15, seed: int = 0) -> tuple[set, set]:
    """Deterministic split of FORM ids into (train, val)."""
    ids = sorted(set(form_ids))
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    return set(ids[n_val:]), set(ids[:n_val])


def write_dataset(items, out_dir: Path, val_fraction: float = 0.15, seed: int = 0) -> dict:
    """Write crops as PNG + labels_{train,val}.jsonl. Returns counts."""
    items = list(items)
    train_ids, _ = split_forms([f for f, _, _ in items], val_fraction, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, files = {"train": 0, "val": 0}, {}
    for split in counts:
        (out_dir / split).mkdir(exist_ok=True)
        files[split] = open(out_dir / f"labels_{split}.jsonl", "w")
    for n, (form, crop, text) in enumerate(items):
        split = "train" if form in train_ids else "val"
        rel = f"{split}/{n:06d}.png"
        cv2.imwrite(str(out_dir / rel), crop)
        files[split].write(json.dumps({"image": rel, "text": text, "form": form}) + "\n")
        counts[split] += 1
    for f in files.values():
        f.close()
    return counts
