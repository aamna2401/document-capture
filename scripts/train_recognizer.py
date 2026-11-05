"""Fine-tune a TrOCR line recognizer on the crops from prepare_crops.py. Validation CER picks the checkpoint.

  python scripts/train_recognizer.py configs/train_recognizer.yaml
Needs a GPU for anything beyond smoke: true. Writes <out_dir>/best (model+processor), history.jsonl, meta.json.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docextract import runmeta                       # noqa: E402
from docextract.metrics import cer                   # noqa: E402


def load(data_dir: Path, split: str, limit=None):
    rows = [json.loads(l) for l in open(data_dir / f"labels_{split}.jsonl")]
    return rows[:limit] if limit else rows


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir, out = Path(cfg["data_dir"]), Path(cfg["out_dir"])
    smoke = cfg.get("smoke", False)
    train, val = load(data_dir, "train", 64 if smoke else None), load(data_dir, "val", 32 if smoke else None)
    assert not ({r["form"] for r in train} & {r["form"] for r in val}), "form leaked across train/val"
    print(f"device={dev} train={len(train)} val={len(val)} smoke={smoke}")

    proc = TrOCRProcessor.from_pretrained(cfg["base_model"])
    model = VisionEncoderDecoderModel.from_pretrained(cfg["base_model"]).to(dev)
    model.config.decoder_start_token_id = proc.tokenizer.cls_token_id
    model.config.pad_token_id = proc.tokenizer.pad_token_id
    model.config.eos_token_id = proc.tokenizer.sep_token_id
    maxlen = cfg["max_target_len"]

    def batch(rows):
        imgs = [Image.open(data_dir / r["image"]).convert("RGB") for r in rows]
        px = proc(images=imgs, return_tensors="pt").pixel_values
        lab = proc.tokenizer([r["text"] for r in rows], padding="max_length", max_length=maxlen,
                             truncation=True, return_tensors="pt").input_ids
        lab[lab == proc.tokenizer.pad_token_id] = -100
        return px.to(dev), lab.to(dev)

    def val_cer() -> float:
        model.eval()
        errs, tot = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), 32):
                rows = val[i:i + 32]
                px = proc(images=[Image.open(data_dir / r["image"]).convert("RGB") for r in rows],
                          return_tensors="pt").pixel_values.to(dev)
                preds = proc.batch_decode(model.generate(px, max_length=maxlen), skip_special_tokens=True)
                for p, r in zip(preds, rows):
                    errs += cer(p, r["text"]) * len(r["text"]); tot += len(r["text"])
        model.train()
        return errs / max(tot, 1)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=cfg["fp16"] and dev == "cuda")
    out.mkdir(parents=True, exist_ok=True)
    runmeta.save(out, cfg, seed)
    best, stale, step, max_steps = 9.9, 0, 0, (20 if smoke else None)
    base_cer = val_cer()
    print(f"val CER before training: {base_cer:.4f}")
    hist = open(out / "history.jsonl", "a")
    model.train()
    for epoch in range(1, cfg["epochs"] + 1):
        order = list(range(len(train))); random.Random(seed + epoch).shuffle(order)
        losses = []
        for i in range(0, len(order), cfg["batch_size"]):
            px, lab = batch([train[j] for j in order[i:i + cfg["batch_size"]]])
            with torch.autocast(dev, enabled=cfg["fp16"] and dev == "cuda"):
                loss = model(pixel_values=px, labels=lab).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item()); step += 1
            if max_steps and step >= max_steps:
                break
        v = val_cer()
        hist.write(json.dumps({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_cer": v}) + "\n"); hist.flush()
        print(f"epoch {epoch}: train_loss={np.mean(losses):.4f} val_cer={v:.4f}")
        if v < best:
            best, stale = v, 0
            model.save_pretrained(out / "best"); proc.save_pretrained(out / "best")
        else:
            stale += 1
            if stale >= cfg["patience"]:
                print("early stop"); break
        if max_steps and step >= max_steps:
            break
    print(f"best val CER {best:.4f} (before training {base_cer:.4f}) -> {out / 'best'}")


if __name__ == "__main__":
    main()
