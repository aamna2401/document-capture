"""Run metadata for reproducibility: code version, config, library versions, hardware, seed."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ["numpy", "opencv-python-headless", "pdfplumber", "pydantic", "rapidocr-onnxruntime", "pytesseract",
            "torch", "transformers", "easyocr"]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def collect(config: dict, seed: int) -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    gpu = ""
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = "cpu (torch not installed)"
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
        "seed": seed, "python": sys.version.split()[0], "platform": platform.platform(), "device": gpu,
        "packages": versions,
    }


def save(run_dir: Path, config: dict, seed: int) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = collect(config, seed)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
