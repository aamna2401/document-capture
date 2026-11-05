"""Evaluation metrics. OCR is scored against the dataset's word annotations, ignoring reading order."""
from collections import Counter

from .ocr import tokens


def word_prf(pred_tokens, gt_tokens):
    """Multiset precision / recall / F1 over tokens."""
    p, g = Counter(pred_tokens), Counter(gt_tokens)
    overlap = sum((p & g).values())
    prec = overlap / max(sum(p.values()), 1)
    rec = overlap / max(sum(g.values()), 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def numeric_recall(pred_tokens, gt_tokens):
    """Recall on tokens that contain a digit (prices, quantities, dates), the part of a receipt that matters most."""
    g = Counter(t for t in gt_tokens if any(c.isdigit() for c in t))
    p = Counter(t for t in pred_tokens if any(c.isdigit() for c in t))
    return sum((p & g).values()) / max(sum(g.values()), 1)


def score_text(pred_text, gt_tokens):
    pt = tokens(pred_text)
    prec, rec, f1 = word_prf(pt, gt_tokens)
    return {"precision": prec, "recall": rec, "f1": f1, "num_recall": numeric_recall(pt, gt_tokens)}
