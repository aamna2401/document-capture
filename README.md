# document-capture

Phone photo or document in, clean text, tables and structured blocks out, with measured accuracy.

- `docscan/`: photo cleanup (detect page, rectify, orient, deskew) with a capture simulator and its own evaluation.
- `docextract/`: extraction pipeline. PDF text layer when usable, OCR otherwise; engines are swappable (`engines.py`).
- `scripts/`: `make_gold.py` (public gold sets), `prepare_crops.py` + `train_recognizer.py` (fine-tune a line recognizer),
  `run_all.py` (every engine on every document type, saved with run metadata), `demo.py`.
- `configs/`: all experiment settings. `notebooks/colab_eval.ipynb`: the heavy runs on a Colab GPU.

```bash
pip install -r requirements.txt && python -m pytest tests -q     # CPU, no model downloads
python scripts/demo.py some.pdf                                  # Markdown + JSON in demo/
```

## Engineering rules used here
- **No test leakage:** training crops are read only from FUNSD/CORD *train*; train/val is split by form; tests assert both.
- **Everything reproducible:** configs in `configs/`, seeds fixed, each run saves `meta.json` (git commit, versions, device).
- **Honest reporting:** two F1 variants, bootstrap intervals over documents, share of documents at target.
- **Resumable runs:** results are written per file; rerunning a run directory skips finished work.
- **Heavy compute stays off the laptop:** training and big evaluations run in Colab.
