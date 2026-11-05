# Dataset

**CORD-v2** (Clova AI), receipts photographed in the wild with word-level annotations. Licence: **CC BY 4.0**.
Source: https://huggingface.co/datasets/naver-clova-ix/cord-v2 . Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", 2019.

| Use | Split | Documents |
|---|---|---|
| Tune detector weights | validation | 34 receipts with a usable paper region |
| Final evaluation (simulated captures) | test | 30 receipts with a usable paper region, each re-photographed 6 times at 3 difficulty levels (540 captures) |
| Real-photo check | test | all 100 native photos |

## Why paper crops
The dataset's photos already contain a table or a hand around the receipt, and only about a third have an annotated paper region (`roi`). To get a clean source document with a known word list, the paper region is flattened with a homography and only words whose centre lies inside it are kept. The capture simulator then re-photographs that document with exact geometry.

## Known limits
- Small number of source documents (30 test, 34 validation). Intervals therefore resample documents.
- CORD blurs sensitive text for privacy, but the blurred words are still in the annotations. No OCR can reach F1 = 1.0 on these receipts; the `oracle` variant is the meaningful ceiling.
- Receipts only. Full-page documents (invoices, forms) would need a re-run.
- Raw data is not committed (`data/` is gitignored); `scripts/` and `docscan/data.py` show how to fetch it.
