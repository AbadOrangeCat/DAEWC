"""Stateless label-budget and metric functions; used identically by all methods."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def stable_seed(seed, domain):
    return (int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16) + seed) % (2**32)


def sample_budget(rows, shots, seed, domain):
    assert all(r["split"] == "train" and r["domain"] == domain for r in rows)
    rng = np.random.default_rng(stable_seed(seed, domain))
    chosen = []
    for label in [0, 1]:
        pool = sorted([r for r in rows if r["label"] == label], key=lambda r: r["id"])
        if len(pool) < shots:
            raise ValueError(f"{domain}: requested {shots}/class, only {len(pool)} in class {label}")
        # One permutation makes shot budgets nested across K for each seed.
        chosen.extend(pool[i] for i in rng.permutation(len(pool))[:shots])
    assert len(chosen) == 2 * shots and len({r["id"] for r in chosen}) == 2 * shots
    return chosen


def metrics(labels, probabilities, threshold=.5):
    y, p = np.asarray(labels), np.asarray(probabilities)
    prediction = (p >= threshold).astype(int)
    return {"macro_f1": float(100 * f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)),
            "accuracy": float(100 * accuracy_score(y, prediction)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}


def source_threshold(labels, probabilities):
    """Source development only. Ties prefer .5, then the smaller threshold."""
    candidates = np.linspace(.05, .95, 91)
    return float(max(candidates, key=lambda t: (metrics(labels, probabilities, t)["macro_f1"], -abs(t-.5), -t)))


def retention(before, after):
    """Signed change in percentage points. Never clip improvements."""
    return float(after - before)


def feasible(delta, tolerance, absolute=False):
    # Both old absolute-band and new loss-only definitions are reported.
    return abs(delta) < tolerance if absolute else delta > -tolerance


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
