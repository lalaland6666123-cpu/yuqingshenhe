"""Small dependency-free metrics used by benchmark runners."""
from __future__ import annotations

from collections import Counter
from math import sqrt


def precision_recall_f1(y_true, y_pred):
    true = Counter(y_true)
    pred = Counter(y_pred)
    labels = sorted(set(true) | set(pred))
    rows = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = pred[label] - tp
        fn = true[label] - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append((p, r, f))
    return {
        "macro_precision": sum(x[0] for x in rows) / len(rows) if rows else 0.0,
        "macro_recall": sum(x[1] for x in rows) / len(rows) if rows else 0.0,
        "macro_f1": sum(x[2] for x in rows) / len(rows) if rows else 0.0,
    }


def mae(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))


def set_f1(true_sets, pred_sets):
    tp = fp = fn = 0
    for truth, pred in zip(true_sets, pred_sets):
        truth, pred = set(truth), set(pred)
        tp += len(truth & pred)
        fp += len(pred - truth)
        fn += len(truth - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def cohens_kappa(y1, y2):
    labels = sorted(set(y1) | set(y2))
    n = max(1, len(y1))
    po = sum(a == b for a, b in zip(y1, y2)) / n
    p1 = {x: y1.count(x) / n for x in labels}
    p2 = {x: y2.count(x) / n for x in labels}
    pe = sum(p1[x] * p2[x] for x in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
