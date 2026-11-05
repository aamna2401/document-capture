"""Text accuracy metrics for the evaluation harness."""
import re


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
    """Character error rate against the reference (whitespace-normalised)."""
    pred, ref = _norm(pred), _norm(ref)
    return _edit_distance(pred, ref) / max(len(ref), 1)


def wer(pred: str, ref: str) -> float:
    p, r = _norm(pred).split(), _norm(ref).split()
    return _edit_distance(p, r) / max(len(r), 1)
