"""Text accuracy metrics for the evaluation harness.

Two word-F1 variants are reported on purpose:
  strict_f1  tokens keep inner . , : / - (so '12.50' is one token); harsh on OCR that glues or splits words
  f1         tokens are runs of letters/digits only; the headline metric, robust to punctuation and spacing
"""
import re
from collections import Counter

_ALNUM = re.compile(r"[a-z0-9]+")


def _edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cer(pred: str, ref: str) -> float:
    """Character error rate against the reference (whitespace-normalised). Order-sensitive."""
    pred, ref = _norm(pred), _norm(ref)
    return _edit_distance(pred, ref) / max(len(ref), 1)


def wer(pred: str, ref: str) -> float:
    p, r = _norm(pred).split(), _norm(ref).split()
    return _edit_distance(p, r) / max(len(r), 1)


def alnum_tokens(text: str) -> list[str]:
    return _ALNUM.findall(text.lower())


def prf(pred_tokens, ref_tokens) -> tuple[float, float, float]:
    """Multiset precision, recall, F1 (reading-order free)."""
    p, g = Counter(pred_tokens), Counter(ref_tokens)
    overlap = sum((p & g).values())
    prec = overlap / max(sum(p.values()), 1)
    rec = overlap / max(sum(g.values()), 1)
    return prec, rec, (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))


def text_scores(pred: str, ref: str) -> dict:
    from docscan.ocr import tokens as strict_tokens         # imported lazily: docscan pulls in OCR deps
    _, _, strict = prf(strict_tokens(pred), strict_tokens(ref))
    prec, rec, f1 = prf(alnum_tokens(pred), alnum_tokens(ref))
    return {"f1": f1, "precision": prec, "recall": rec, "strict_f1": strict, "cer": cer(pred, ref), "wer": wer(pred, ref)}
