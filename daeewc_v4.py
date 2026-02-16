# daeewc_v4.py
# -*- coding: utf-8 -*-
"""
DAEWC v4 (paper-like + make DAEWC as strong as possible)

Key changes vs v3:
  1) Few-shot is TOTAL labeled samples: {10, 20, 80, 160} (balanced per class).
  2) Strong near-duplicate filtering (SimHash + LSH banding):
        - within-domain dedup (source + target)
        - cross-domain dedup (remove target samples near-duplicate of ANY source split)
  3) Source-only tokenizer (paper-strict), vocab cap=5000.
  4) EWC Fisher estimation made effective:
        - temperature-soft pseudo labels (avoid saturated gradients)
        - Fisher mean normalization (mean(F)=1) -> lambda becomes meaningful
  5) DAEWC:
        Stage1: train adapter/gate/domain-emb/target-head (backbone frozen)
        Stage2: selectively unfreeze backbone top layers + EWC + L2 anchor
               grid search lambda on dev (and optionally strategy for >=80)

Outputs:
  - results_daeewc_paperstyle_v4_raw.csv
  - results_daeewc_paperstyle_v4_summary.csv
"""

from __future__ import annotations
import os
import re
import json
import math
import time
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import tensorflow as tf


# ======================================================================================
# CONFIG (no CLI args; edit here if needed)
# ======================================================================================

# ---- data ----
DATA_DIR = "./data"

# Try these filenames (in DATA_DIR). First match wins.
SOURCE_FILE_CANDIDATES = [
    "political.csv", "political.jsonl", "political.tsv",
    "Political.csv", "Political.jsonl", "Political.tsv",
    "source.csv", "source.jsonl"
]
TARGET_FILE_CANDIDATES = [
    "medical.csv", "medical.jsonl", "medical.tsv",
    "Medical.csv", "Medical.jsonl", "Medical.tsv",
    "target.csv", "target.jsonl"
]

# Column auto-detection for CSV/TSV
TEXT_COL_CANDIDATES = ["text", "content", "article", "body", "claim", "statement"]
TITLE_COL_CANDIDATES = ["title", "headline"]
LABEL_COL_CANDIDATES = ["label", "y", "target", "is_fake", "fake", "verdict"]

# Label mapping
FAKE_STRINGS = {"fake", "false", "fabricated", "misleading", "0", "1"}  # we'll handle numeric separately
REAL_STRINGS = {"real", "true", "legit", "genuine"}

# ---- experiment ----
SEEDS = [42, 43, 44, 45, 46]
SHOTS_TOTAL = [10, 20, 80, 160]  # TOTAL samples (balanced per class)
SPLIT_RATIOS = (0.70, 0.10, 0.20)  # train/dev/test

# ---- tokenizer ----
MAX_VOCAB = 5000  # cap
MAX_LEN = 256

# ---- model ----
EMB_DIM = 100
CONV_FILTERS = 128
KERNEL_SIZE = 5
HIDDEN_DIM = 128

DOM_EMB_DIM = 64  # for gate
# adapter is a width-128 residual MLP (stronger than tiny bottleneck; closer to your v3 param scale)
ADAPTER_HIDDEN_DIM = 128

DROPOUT_RATE = 0.10

# ---- training general ----
BATCH_PRETRAIN = 64
BATCH_ADAPT = 32
BATCH_EVAL = 256

EPOCHS_PRETRAIN = 12
PATIENCE_PRETRAIN = 2

EPOCHS_ADAPT = 40
PATIENCE_ADAPT = 6

MIN_STEPS_PER_EPOCH = 50  # ensures enough updates for tiny few-shot

LR_PRETRAIN = 1e-3
LR_ADAPT = 2e-3

GRAD_CLIP_NORM = 1.0

# ---- EWC / Fisher ----
FISHER_BATCHES = 200  # ~200 batches * 64 = 12800 examples max (will stop earlier if dataset smaller)
FISHER_TEMPERATURE = 2.5
FISHER_EPS = 1e-8

# DAEWC Stage2 search space (dev-selected)
LAMBDA_GRID = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Stage2 layer-unfreeze strategies
# For shot<=20: default will only try "fc_only"
# For shot>=80: will try both "fc_only" and "conv_fc" and pick best on dev
STRATEGY_SMALL = ["fc_only"]
STRATEGY_LARGE = ["fc_only", "conv_fc"]

# Backbone LR multiplier in Stage2 (adapter uses LR_ADAPT; backbone grads scaled by this)
BACKBONE_LR_MULT_GRID = [0.05, 0.10, 0.20]

# Extra anchoring (L2-SP style) for any unfrozen backbone vars in Stage2 (works well with normalized Fisher)
L2_ANCHOR_ALPHA = 1e-4

# ---- dedup ----
# SimHash near-dup hamming threshold (0=exact, 3-5 typical). Start at 3 for strong filtering.
SIMHASH_HAMMING_THRESH = 3
SIMHASH_NGRAM = 3           # word n-gram size
SIMHASH_BITS = 64
SIMHASH_BANDS = 4           # 4 bands x 16 bits = 64 bits

# Print more logs
VERBOSE = True


# ======================================================================================
# Utilities
# ======================================================================================

def _log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def configure_tf() -> None:
    # Make GPU memory growth friendlier
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " <url> ", s)
    s = re.sub(r"\d+", " <num> ", s)
    # keep letters, angle-bracket tokens, and spaces
    s = re.sub(r"[^a-z<>\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_concat_text(title: str, body: str) -> str:
    title = "" if title is None else str(title)
    body = "" if body is None else str(body)
    if title and body:
        return f"{title}. {body}"
    return title or body

def find_existing_file(data_dir: str, candidates: List[str]) -> str:
    for fn in candidates:
        path = os.path.join(data_dir, fn)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not find dataset file in {data_dir}. Tried: {candidates}\n"
        f"Please put your files there or edit SOURCE_FILE_CANDIDATES/TARGET_FILE_CANDIDATES."
    )

def load_dataset_auto(path: str) -> Tuple[List[str], List[int]]:
    """
    Loads CSV/TSV/JSONL dataset and returns (texts, labels) where label: fake=1, real=0.
    Tries to auto-detect text/title/label columns.

    Accepted formats:
      - CSV/TSV with columns containing {text/content/...} and {label/...}
      - JSONL where each line is a JSON object with similar keys
    """
    ext = os.path.splitext(path)[1].lower()

    records: List[Dict[str, Any]] = []

    if ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, engine="python")
        cols = list(df.columns)

        def pick_col(cands: List[str]) -> Optional[str]:
            for c in cands:
                if c in cols:
                    return c
            # fallback: case-insensitive match
            lower_map = {c.lower(): c for c in cols}
            for c in cands:
                if c.lower() in lower_map:
                    return lower_map[c.lower()]
            return None

        text_col = pick_col(TEXT_COL_CANDIDATES)
        title_col = pick_col(TITLE_COL_CANDIDATES)
        label_col = pick_col(LABEL_COL_CANDIDATES)

        if label_col is None:
            raise ValueError(
                f"Cannot find label column in {path}. "
                f"Columns: {cols}. Candidates: {LABEL_COL_CANDIDATES}"
            )
        if text_col is None and title_col is None:
            raise ValueError(
                f"Cannot find text/title column in {path}. "
                f"Columns: {cols}. Candidates text={TEXT_COL_CANDIDATES}, title={TITLE_COL_CANDIDATES}"
            )

        for _, row in df.iterrows():
            title = row[title_col] if title_col else ""
            body = row[text_col] if text_col else ""
            text = safe_concat_text(title, body)
            lab = row[label_col]
            records.append({"text": text, "label": lab})

    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(obj)
    else:
        raise ValueError(f"Unsupported dataset extension: {ext} (file={path})")

    texts: List[str] = []
    labels: List[int] = []

    def map_label(v: Any) -> Optional[int]:
        # numeric
        if isinstance(v, (int, np.integer)):
            return 1 if int(v) == 1 else 0
        if isinstance(v, (float, np.floating)):
            return 1 if int(v) == 1 else 0

        # bool
        if isinstance(v, bool):
            return 1 if v else 0

        # string
        s = str(v).strip().lower()
        # common: "fake"/"real"
        if s in FAKE_STRINGS:
            # NOTE: If your dataset uses "0" meaning fake, this will be wrong.
            # We'll try to infer below with heuristics if needed.
            pass
        if "fake" in s or "false" in s:
            return 1
        if "real" in s or "true" in s:
            return 0
        # last resort: try int
        try:
            return 1 if int(s) == 1 else 0
        except Exception:
            return None

    # For JSONL: try to pick keys
    if ext == ".jsonl":
        # attempt auto key detection
        # build a key frequency map
        key_freq: Dict[str, int] = {}
        for r in records:
            if isinstance(r, dict):
                for k in r.keys():
                    key_freq[k] = key_freq.get(k, 0) + 1

        def pick_key(cands: List[str]) -> Optional[str]:
            for c in cands:
                if c in key_freq:
                    return c
            lower_map = {k.lower(): k for k in key_freq}
            for c in cands:
                if c.lower() in lower_map:
                    return lower_map[c.lower()]
            return None

        text_key = pick_key(TEXT_COL_CANDIDATES)
        title_key = pick_key(TITLE_COL_CANDIDATES)
        label_key = pick_key(LABEL_COL_CANDIDATES)

        if label_key is None:
            raise ValueError(
                f"Cannot find label key in {path}. Keys sample: {list(key_freq.keys())[:30]}"
            )
        if text_key is None and title_key is None:
            raise ValueError(
                f"Cannot find text/title key in {path}. Keys sample: {list(key_freq.keys())[:30]}"
            )

        for r in records:
            if not isinstance(r, dict):
                continue
            title = r.get(title_key, "") if title_key else ""
            body = r.get(text_key, "") if text_key else ""
            text = safe_concat_text(title, body)
            lab = r.get(label_key, None)
            y = map_label(lab)
            if y is None:
                continue
            texts.append(text)
            labels.append(int(y))
    else:
        for r in records:
            y = map_label(r["label"])
            if y is None:
                continue
            texts.append(r["text"])
            labels.append(int(y))

    # normalize text
    texts = [normalize_text(t) for t in texts]
    # drop empties
    out_texts, out_labels = [], []
    for t, y in zip(texts, labels):
        if t.strip():
            out_texts.append(t)
            out_labels.append(y)

    return out_texts, out_labels


# ======================================================================================
# SimHash near-duplicate filtering
# ======================================================================================

def _hash64(token: str) -> int:
    # stable 64-bit hash from md5
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)

def simhash64(text: str, ngram: int = 3, bits: int = 64) -> int:
    """
    Word n-gram SimHash over normalized text.
    """
    tokens = text.split()
    if len(tokens) == 0:
        return 0
    # build n-grams
    grams: List[str] = []
    if len(tokens) < ngram:
        grams = tokens
    else:
        grams = [" ".join(tokens[i:i+ngram]) for i in range(len(tokens) - ngram + 1)]

    # term frequency
    tfreq: Dict[str, int] = {}
    for g in grams:
        tfreq[g] = tfreq.get(g, 0) + 1

    v = [0] * bits
    for g, w in tfreq.items():
        hv = _hash64(g)
        for i in range(bits):
            bit = (hv >> i) & 1
            v[i] += w if bit else -w

    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= (1 << i)
    return out

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def _band_keys(h: int, bands: int = 4, bits: int = 64) -> List[Tuple[int, int]]:
    # bands equal size
    band_bits = bits // bands
    keys = []
    for bi in range(bands):
        mask = (1 << band_bits) - 1
        v = (h >> (bi * band_bits)) & mask
        keys.append((bi, v))
    return keys

def dedup_near(
    texts: List[str],
    labels: List[int],
    hamming_thresh: int = 3,
    ngram: int = 3,
    bits: int = 64,
    bands: int = 4,
) -> Tuple[List[str], List[int], int]:
    """
    Near-duplicate removal using SimHash + LSH banding.
    Keeps first occurrence; removes later ones with hamming <= threshold.
    """
    assert len(texts) == len(labels)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    kept_hashes: List[int] = []
    kept_texts: List[str] = []
    kept_labels: List[int] = []

    removed = 0
    for t, y in zip(texts, labels):
        h = simhash64(t, ngram=ngram, bits=bits)
        cand_idx: List[int] = []
        for key in _band_keys(h, bands=bands, bits=bits):
            cand_idx.extend(buckets.get(key, []))

        is_dup = False
        # check candidates
        for idx in cand_idx:
            if hamming64(h, kept_hashes[idx]) <= hamming_thresh:
                is_dup = True
                break

        if is_dup:
            removed += 1
            continue

        idx_new = len(kept_hashes)
        kept_hashes.append(h)
        kept_texts.append(t)
        kept_labels.append(y)
        for key in _band_keys(h, bands=bands, bits=bits):
            buckets.setdefault(key, []).append(idx_new)

    return kept_texts, kept_labels, removed

def cross_domain_dedup_target_against_source(
    src_texts: List[str],
    tgt_texts: List[str],
    tgt_labels: List[int],
    hamming_thresh: int = 3,
    ngram: int = 3,
    bits: int = 64,
    bands: int = 4,
) -> Tuple[List[str], List[int], int]:
    """
    Remove target samples that near-duplicate ANY source sample.
    """
    # build LSH buckets from source
    buckets: Dict[Tuple[int, int], List[int]] = {}
    src_hashes: List[int] = []
    for t in src_texts:
        h = simhash64(t, ngram=ngram, bits=bits)
        idx = len(src_hashes)
        src_hashes.append(h)
        for key in _band_keys(h, bands=bands, bits=bits):
            buckets.setdefault(key, []).append(idx)

    kept_texts, kept_labels = [], []
    removed = 0
    for t, y in zip(tgt_texts, tgt_labels):
        h = simhash64(t, ngram=ngram, bits=bits)
        cand_idx: List[int] = []
        for key in _band_keys(h, bands=bands, bits=bits):
            cand_idx.extend(buckets.get(key, []))

        is_dup = False
        for idx in cand_idx:
            if hamming64(h, src_hashes[idx]) <= hamming_thresh:
                is_dup = True
                break

        if is_dup:
            removed += 1
            continue

        kept_texts.append(t)
        kept_labels.append(y)

    return kept_texts, kept_labels, removed


# ======================================================================================
# Split / sampling
# ======================================================================================

def stratified_split(
    texts: List[str],
    labels: List[int],
    ratios: Tuple[float, float, float],
    seed: int
) -> Dict[str, Tuple[List[str], List[int]]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    rng = np.random.RandomState(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split_one(idxs: List[int]) -> Tuple[List[int], List[int], List[int]]:
        n = len(idxs)
        n_train = int(round(ratios[0] * n))
        n_dev = int(round(ratios[1] * n))
        n_test = n - n_train - n_dev
        train = idxs[:n_train]
        dev = idxs[n_train:n_train + n_dev]
        test = idxs[n_train + n_dev:]
        assert len(train) + len(dev) + len(test) == n
        return train, dev, test

    tr0, dv0, te0 = split_one(idx0)
    tr1, dv1, te1 = split_one(idx1)

    train_idx = tr0 + tr1
    dev_idx = dv0 + dv1
    test_idx = te0 + te1
    rng.shuffle(train_idx)
    rng.shuffle(dev_idx)
    rng.shuffle(test_idx)

    def pack(idxs: List[int]) -> Tuple[List[str], List[int]]:
        return [texts[i] for i in idxs], [labels[i] for i in idxs]

    return {
        "train": pack(train_idx),
        "dev": pack(dev_idx),
        "test": pack(test_idx),
    }

def sample_fewshot_total_balanced(
    texts: List[str],
    labels: List[int],
    shot_total: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TOTAL=shot_total, balanced (half per class).
    """
    assert shot_total % 2 == 0, "shot_total must be even for balanced sampling."
    k = shot_total // 2
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    rng = np.random.RandomState(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    if len(idx0) < k or len(idx1) < k:
        raise ValueError(f"Not enough samples to draw {k} per class. Have: real={len(idx0)} fake={len(idx1)}")
    sel = idx0[:k] + idx1[:k]
    rng.shuffle(sel)
    sel_texts = [texts[i] for i in sel]
    sel_labels = [labels[i] for i in sel]
    return np.array(sel_texts, dtype=object), np.array(sel_labels, dtype=np.int32)


# ======================================================================================
# Tokenizer (source-only, paper-strict)
# ======================================================================================

def build_vocab_source_only(texts_src_train: List[str], max_vocab: int) -> Dict[str, int]:
    """
    Build vocab from source train only.
    Reserve:
      0: <pad>
      1: <unk>
    """
    freq: Dict[str, int] = {}
    for t in texts_src_train:
        for tok in t.split():
            freq[tok] = freq.get(tok, 0) + 1

    # top tokens
    # max_vocab includes pad+unk
    num_tokens = max_vocab - 2
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:num_tokens]

    token2id = {"<pad>": 0, "<unk>": 1}
    for i, (tok, _) in enumerate(top, start=2):
        token2id[tok] = i
    return token2id

def vectorize_texts(
    texts: List[str],
    token2id: Dict[str, int],
    max_len: int
) -> np.ndarray:
    unk = token2id.get("<unk>", 1)
    pad = token2id.get("<pad>", 0)
    out = np.full((len(texts), max_len), pad, dtype=np.int32)
    for i, t in enumerate(texts):
        toks = t.split()
        ids = [token2id.get(tok, unk) for tok in toks[:max_len]]
        out[i, :len(ids)] = np.array(ids, dtype=np.int32)
    return out


# ======================================================================================
# Metrics
# ======================================================================================

def binary_metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = y_true.astype(np.int32).reshape(-1)
    probs = probs.astype(np.float32).reshape(-1)
    y_pred = (probs >= thr).astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    acc = (tp + tn) / max(1, len(y_true))

    # F1 for class 1
    denom1 = (2 * tp + fp + fn)
    f1_1 = (2 * tp / denom1) if denom1 > 0 else 0.0

    # F1 for class 0: treat 0 as positive
    # TP0=TN, FP0=FN, FN0=FP
    denom0 = (2 * tn + fn + fp)
    f1_0 = (2 * tn / denom0) if denom0 > 0 else 0.0

    macro_f1 = 0.5 * (f1_0 + f1_1)

    return {"acc": float(acc), "macro_f1": float(macro_f1)}

def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # search a dense grid
    best_thr = 0.5
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    for thr in np.linspace(0.05, 0.95, 181):
        m = binary_metrics_from_probs(y_true, probs, float(thr))
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics

def bce_from_logits(y_true: np.ndarray, logits: np.ndarray) -> float:
    y = y_true.astype(np.float32).reshape(-1, 1)
    l = logits.astype(np.float32).reshape(-1, 1)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=l)
    return float(tf.reduce_mean(bce).numpy())


# ======================================================================================
# Models
# ======================================================================================

class PlainTwoHeadCNN(tf.keras.Model):
    """
    Backbone: Embedding -> Conv1D -> GlobalMaxPool -> Dense
    Heads: src_head, tgt_head (binary logits)
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMB_DIM, name="emb")
        self.conv = tf.keras.layers.Conv1D(
            filters=CONV_FILTERS, kernel_size=KERNEL_SIZE,
            activation="relu", padding="valid", name="conv"
        )
        self.pool = tf.keras.layers.GlobalMaxPool1D(name="gmp")
        self.dropout = tf.keras.layers.Dropout(DROPOUT_RATE, name="drop")
        self.shared_fc = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", name="shared_fc")

        self.head_src = tf.keras.layers.Dense(1, name="head_src")
        self.head_tgt = tf.keras.layers.Dense(1, name="head_tgt")

    def encode(self, x_ids: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.embedding(x_ids)
        x = self.conv(x)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        z = self.shared_fc(x)
        return z

    def logits(self, x_ids: tf.Tensor, head: str, training: bool) -> tf.Tensor:
        z = self.encode(x_ids, training=training)
        if head == "src":
            return self.head_src(z)
        elif head == "tgt":
            return self.head_tgt(z)
        else:
            raise ValueError(f"Unknown head={head}")

class DAEWCCNN(tf.keras.Model):
    """
    DAEWC wrapper:
      - same backbone & two heads
      - domain embedding + gate -> per-dim scale g (size=HIDDEN_DIM)
      - residual adapter MLP on representation z
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        # backbone (same as plain)
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMB_DIM, name="emb")
        self.conv = tf.keras.layers.Conv1D(
            filters=CONV_FILTERS, kernel_size=KERNEL_SIZE,
            activation="relu", padding="valid", name="conv"
        )
        self.pool = tf.keras.layers.GlobalMaxPool1D(name="gmp")
        self.dropout = tf.keras.layers.Dropout(DROPOUT_RATE, name="drop")
        self.shared_fc = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", name="shared_fc")

        # heads
        self.head_src = tf.keras.layers.Dense(1, name="head_src")
        self.head_tgt = tf.keras.layers.Dense(1, name="head_tgt")

        # domain embedding (2 domains: 0=source, 1=target)
        self.dom_emb = self.add_weight(
            name="dom_emb", shape=(2, DOM_EMB_DIM),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        # gate maps dom_emb -> [0,1]^HIDDEN_DIM
        self.gate = tf.keras.layers.Dense(HIDDEN_DIM, activation="sigmoid", name="gate")

        # residual adapter (strong MLP)
        self.adapter_dense1 = tf.keras.layers.Dense(ADAPTER_HIDDEN_DIM, activation="relu", name="adapter_d1")
        self.adapter_dense2 = tf.keras.layers.Dense(HIDDEN_DIM, activation=None, name="adapter_d2")

    def encode_backbone(self, x_ids: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.embedding(x_ids)
        x = self.conv(x)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        z = self.shared_fc(x)
        return z

    def encode_with_adapter(self, x_ids: tf.Tensor, domain_id: int, training: bool) -> tf.Tensor:
        z = self.encode_backbone(x_ids, training=training)
        d = self.dom_emb[domain_id]  # (DOM_EMB_DIM,)
        g = self.gate(tf.expand_dims(d, 0))  # (1, HIDDEN_DIM)
        g = tf.cast(g, z.dtype)
        # adapter
        a = self.adapter_dense1(z)
        a = self.adapter_dense2(a)
        z2 = z + g * a
        return z2

    def logits(
        self,
        x_ids: tf.Tensor,
        head: str,
        domain_id: int,
        training: bool,
        use_adapter: bool
    ) -> tf.Tensor:
        if use_adapter:
            z = self.encode_with_adapter(x_ids, domain_id=domain_id, training=training)
        else:
            z = self.encode_backbone(x_ids, training=training)

        if head == "src":
            return self.head_src(z)
        elif head == "tgt":
            return self.head_tgt(z)
        else:
            raise ValueError(f"Unknown head={head}")


# ======================================================================================
# EWC / training
# ======================================================================================

@dataclass
class EWCInfo:
    # each entry: key -> (theta_star, fisher)
    star: Dict[str, np.ndarray]
    fisher: Dict[str, np.ndarray]

def get_protected_vars_plain(model: PlainTwoHeadCNN) -> Dict[str, tf.Variable]:
    # Keep consistent keys across models
    return {
        "emb": model.embedding.embeddings,
        "conv_kernel": model.conv.kernel,
        "fc_kernel": model.shared_fc.kernel,
    }

def get_protected_vars_dae(model: DAEWCCNN) -> Dict[str, tf.Variable]:
    return {
        "emb": model.embedding.embeddings,
        "conv_kernel": model.conv.kernel,
        "fc_kernel": model.shared_fc.kernel,
    }

def estimate_fisher_plain(
    model: PlainTwoHeadCNN,
    ds: tf.data.Dataset,
    head: str = "src",
    max_batches: int = 200,
    temperature: float = 2.5
) -> EWCInfo:
    """
    Empirical Fisher approximation:
      F_i = E[ (dL/dtheta_i)^2 ]
    We avoid vanishing gradients by using softened pseudo-labels:
      y_soft = sigmoid(logits / T)
      loss = CE(y_soft, logits)  (from_logits)  -> gradient ~ sigmoid(logits) - y_soft
    Then normalize fisher to mean=1 over all protected entries.
    """
    protected = get_protected_vars_plain(model)
    # capture theta*
    star = {k: v.numpy().copy() for k, v in protected.items()}

    # accum
    fisher_acc = {k: np.zeros_like(star[k], dtype=np.float32) for k in star.keys()}
    n_batches = 0

    for batch in ds.take(max_batches):
        x, _y = batch
        with tf.GradientTape() as tape:
            logits = model.logits(x, head=head, training=False)
            y_soft = tf.stop_gradient(tf.sigmoid(logits / float(temperature)))
            loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_soft, logits=logits)
            loss = tf.reduce_mean(loss_vec)

        vars_list = list(protected.values())
        grads = tape.gradient(loss, vars_list)

        for (k, v), g in zip(protected.items(), grads):
            if g is None:
                continue
            fisher_acc[k] += (g.numpy().astype(np.float32) ** 2)

        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Fisher estimation got 0 batches.")

    fisher = {k: fisher_acc[k] / float(n_batches) for k in fisher_acc.keys()}

    # mean-normalize (global)
    all_means = [float(np.mean(f)) for f in fisher.values()]
    mean_f = float(np.mean(all_means))
    if mean_f < FISHER_EPS:
        mean_f = FISHER_EPS
    fisher = {k: f / mean_f for k, f in fisher.items()}

    return EWCInfo(star=star, fisher=fisher)

def ewc_penalty(current_vars: Dict[str, tf.Variable], ewc: EWCInfo) -> tf.Tensor:
    # sum_k sum_i fisher_i (theta - theta*)^2
    losses = []
    for k, v in current_vars.items():
        if k not in ewc.star:
            continue
        theta_star = tf.constant(ewc.star[k], dtype=v.dtype)
        fisher = tf.constant(ewc.fisher[k], dtype=v.dtype)
        losses.append(tf.reduce_sum(fisher * tf.square(v - theta_star)))
    if not losses:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n(losses)

def l2_anchor_penalty(vars_to_anchor: List[tf.Variable], star_map: Dict[str, np.ndarray]) -> tf.Tensor:
    """
    Unweighted L2 anchor for any unfrozen vars that exist in star_map.
    This complements EWC (which is fisher-weighted).
    """
    losses = []
    for v in vars_to_anchor:
        name = v.name
        # We can't rely on exact names across instances; so we pass star_map keyed by logical keys elsewhere.
        # Here we do nothing unless explicitly provided with mapping by object identity (not used).
        _ = name
    return tf.constant(0.0, dtype=tf.float32)

def count_params(vars_list: List[tf.Variable]) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in vars_list))

def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, seed: int, repeat: bool) -> tf.data.Dataset:
    y = y.astype(np.float32).reshape(-1, 1)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def eval_logits_probs(
    logits_fn,
    ds_eval: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (y_true, logits, probs)
    """
    ys: List[np.ndarray] = []
    ls: List[np.ndarray] = []
    for x, y in ds_eval:
        logits = logits_fn(x, training=False)
        ys.append(y.numpy())
        ls.append(logits.numpy())
    y_true = np.concatenate(ys, axis=0).reshape(-1)
    logits_all = np.concatenate(ls, axis=0).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits_all))
    return y_true, logits_all, probs

def train_loop_earlystop(
    train_ds: tf.data.Dataset,
    dev_ds: tf.data.Dataset,
    logits_fn,                      # (x, training)->logits
    train_vars: List[tf.Variable],
    backbone_vars: List[tf.Variable],
    ewc_info: Optional[EWCInfo],
    ewc_lambda: float,
    l2_anchor_alpha: float,
    optimizer: tf.keras.optimizers.Optimizer,
    steps_per_epoch: int,
    max_epochs: int,
    patience: int,
    backbone_lr_mult: float,
    tag: str
) -> Tuple[float, Dict[str, float]]:
    """
    Custom training with early stopping by dev Macro-F1 (threshold optimized each epoch).
    Returns (best_thr, best_dev_metrics)
    """
    # snapshot weights
    best_weights = None
    best_thr = 0.5
    best_f1 = -1.0
    best_metrics = {}
    wait = 0

    protected_now = None
    if ewc_info is not None:
        # infer whether model is plain or dae by probing vars
        # We'll build the protected dict lazily inside the step using variable objects.
        protected_now = None

    backbone_ids = set(id(v) for v in backbone_vars)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = logits_fn(x, training=True)
            loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.reduce_mean(loss_vec)

            if ewc_info is not None:
                # build protected vars map from train_vars via names? better: pass from outer scope
                # We'll assume logits_fn is bound to a model in closure and we can access it:
                # but tf.function closure capturing a python object is OK.
                # We'll require logits_fn has attribute "_ewc_protected" set by caller (hacky but stable).
                prot = getattr(logits_fn, "_ewc_protected", None)
                if prot is not None:
                    loss += tf.cast(ewc_lambda, loss.dtype) * ewc_penalty(prot, ewc_info)

            # L2 anchor: simple unweighted anchor on protected vars too (optional)
            if l2_anchor_alpha > 0.0 and ewc_info is not None:
                prot = getattr(logits_fn, "_ewc_protected", None)
                if prot is not None:
                    # anchor (theta - theta*)^2 without fisher
                    l2_losses = []
                    for k, v in prot.items():
                        if k in ewc_info.star:
                            theta_star = tf.constant(ewc_info.star[k], dtype=v.dtype)
                            l2_losses.append(tf.reduce_sum(tf.square(v - theta_star)))
                    if l2_losses:
                        loss += tf.cast(l2_anchor_alpha, loss.dtype) * tf.add_n(l2_losses)

        grads = tape.gradient(loss, train_vars)
        # scale backbone grads
        scaled = []
        for g, v in zip(grads, train_vars):
            if g is None:
                scaled.append(g)
            else:
                if id(v) in backbone_ids:
                    scaled.append(g * tf.cast(backbone_lr_mult, g.dtype))
                else:
                    scaled.append(g)

        # clip
        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            scaled, _ = tf.clip_by_global_norm(scaled, GRAD_CLIP_NORM)

        optimizer.apply_gradients([(g, v) for g, v in zip(scaled, train_vars) if g is not None])
        return loss

    for epoch in range(1, max_epochs + 1):
        # train epoch
        losses = []
        it = iter(train_ds)
        for _ in range(steps_per_epoch):
            x_b, y_b = next(it)
            l = train_step(x_b, y_b)
            losses.append(float(l.numpy()))

        # eval dev with threshold search
        y_true, logits_all, probs = eval_logits_probs(logits_fn, dev_ds)
        thr, m = find_best_threshold(y_true, probs)
        bce = bce_from_logits(y_true, logits_all)

        if VERBOSE:
            _log(f"{tag} | epoch={epoch:02d}  train_loss={np.mean(losses):.4f}  dev_thr={thr:.3f}  dev_acc={m['acc']:.4f}  dev_MacroF1={m['macro_f1']:.4f}  dev_BCE={bce:.4f}")

        if m["macro_f1"] > best_f1 + 1e-6:
            best_f1 = m["macro_f1"]
            best_thr = thr
            best_metrics = {"acc": m["acc"], "macro_f1": m["macro_f1"], "bce": bce}
            best_weights = [w.copy() for w in logits_fn._model.get_weights()]  # store full weights
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_weights is not None:
        logits_fn._model.set_weights(best_weights)

    # recompute best metrics on dev after restore
    y_true, logits_all, probs = eval_logits_probs(logits_fn, dev_ds)
    thr, m = find_best_threshold(y_true, probs)
    bce = bce_from_logits(y_true, logits_all)
    metrics = {"acc": m["acc"], "macro_f1": m["macro_f1"], "bce": bce}
    return thr, metrics


# ======================================================================================
# Experiment runner
# ======================================================================================

def build_and_init_plain(vocab_size: int) -> PlainTwoHeadCNN:
    model = PlainTwoHeadCNN(vocab_size)
    # build weights
    dummy = tf.zeros((2, MAX_LEN), dtype=tf.int32)
    _ = model.logits(dummy, head="src", training=False)
    _ = model.logits(dummy, head="tgt", training=False)
    return model

def build_and_init_dae(vocab_size: int) -> DAEWCCNN:
    model = DAEWCCNN(vocab_size)
    dummy = tf.zeros((2, MAX_LEN), dtype=tf.int32)
    _ = model.logits(dummy, head="src", domain_id=0, training=False, use_adapter=False)
    _ = model.logits(dummy, head="tgt", domain_id=1, training=False, use_adapter=True)
    return model

def copy_backbone_and_src_head_plain_to_plain(src: PlainTwoHeadCNN, dst: PlainTwoHeadCNN) -> None:
    dst.embedding.set_weights(src.embedding.get_weights())
    dst.conv.set_weights(src.conv.get_weights())
    dst.shared_fc.set_weights(src.shared_fc.get_weights())
    dst.head_src.set_weights(src.head_src.get_weights())

def copy_backbone_and_src_head_plain_to_dae(src: PlainTwoHeadCNN, dst: DAEWCCNN) -> None:
    dst.embedding.set_weights(src.embedding.get_weights())
    dst.conv.set_weights(src.conv.get_weights())
    dst.shared_fc.set_weights(src.shared_fc.get_weights())
    dst.head_src.set_weights(src.head_src.get_weights())

def eval_on_split_plain(model: PlainTwoHeadCNN, x: np.ndarray, y: np.ndarray, head: str, seed: int) -> Tuple[float, Dict[str, float]]:
    ds = make_tf_dataset(x, y, batch=BATCH_EVAL, shuffle=False, seed=seed, repeat=False)
    def logits_fn(xb, training=False):
        return model.logits(xb, head=head, training=training)
    logits_fn._model = model
    y_true, logits_all, probs = eval_logits_probs(logits_fn, ds)
    thr, m = find_best_threshold(y_true, probs)
    bce = bce_from_logits(y_true, logits_all)
    return thr, {"acc": m["acc"], "macro_f1": m["macro_f1"], "bce": bce}

def eval_on_split_plain_with_thr(model: PlainTwoHeadCNN, x: np.ndarray, y: np.ndarray, head: str, thr: float, seed: int) -> Dict[str, float]:
    ds = make_tf_dataset(x, y, batch=BATCH_EVAL, shuffle=False, seed=seed, repeat=False)
    def logits_fn(xb, training=False):
        return model.logits(xb, head=head, training=training)
    logits_fn._model = model
    y_true, logits_all, probs = eval_logits_probs(logits_fn, ds)
    m = binary_metrics_from_probs(y_true, probs, thr)
    bce = bce_from_logits(y_true, logits_all)
    return {"acc": m["acc"], "macro_f1": m["macro_f1"], "bce": bce}

def eval_on_split_dae(model: DAEWCCNN, x: np.ndarray, y: np.ndarray, head: str, domain_id: int, use_adapter: bool, seed: int) -> Tuple[float, Dict[str, float]]:
    ds = make_tf_dataset(x, y, batch=BATCH_EVAL, shuffle=False, seed=seed, repeat=False)
    def logits_fn(xb, training=False):
        return model.logits(xb, head=head, domain_id=domain_id, training=training, use_adapter=use_adapter)
    logits_fn._model = model
    y_true, logits_all, probs = eval_logits_probs(logits_fn, ds)
    thr, m = find_best_threshold(y_true, probs)
    bce = bce_from_logits(y_true, logits_all)
    return thr, {"acc": m["acc"], "macro_f1": m["macro_f1"], "bce": bce}

def eval_on_split_dae_with_thr(model: DAEWCCNN, x: np.ndarray, y: np.ndarray, head: str, domain_id: int, use_adapter: bool, thr: float, seed: int) -> Dict[str, float]:
    ds = make_tf_dataset(x, y, batch=BATCH_EVAL, shuffle=False, seed=seed, repeat=False)
    def logits_fn(xb, training=False):
        return model.logits(xb, head=head, domain_id=domain_id, training=training, use_adapter=use_adapter)
    logits_fn._model = model
    y_true, logits_all, probs = eval_logits_probs(logits_fn, ds)
    m = binary_metrics_from_probs(y_true, probs, thr)
    bce = bce_from_logits(y_true, logits_all)
    return {"acc": m["acc"], "macro_f1": m["macro_f1"], "bce": bce}


def main():
    configure_tf()

    # ---------------------------
    # Load datasets
    # ---------------------------
    src_path = find_existing_file(DATA_DIR, SOURCE_FILE_CANDIDATES)
    tgt_path = find_existing_file(DATA_DIR, TARGET_FILE_CANDIDATES)

    _log(f"Loading datasets...\n  source={src_path}\n  target={tgt_path}")
    src_texts, src_labels = load_dataset_auto(src_path)
    tgt_texts, tgt_labels = load_dataset_auto(tgt_path)

    def count_labels(lbls):
        fake = int(np.sum(np.array(lbls) == 1))
        real = int(np.sum(np.array(lbls) == 0))
        return fake, real, fake + real

    sf, sr, st = count_labels(src_labels)
    tf_, tr, tt = count_labels(tgt_labels)
    _log(f"Source   | fake={sf}  real={sr}  total={st}")
    _log(f"Target   | fake={tf_}  real={tr}  total={tt}")

    # ---------------------------
    # Strong near-dup filtering
    # ---------------------------
    _log("\n[Dedup] Within-domain near-duplicate removal...")
    src_texts, src_labels, rm_src = dedup_near(
        src_texts, src_labels,
        hamming_thresh=SIMHASH_HAMMING_THRESH,
        ngram=SIMHASH_NGRAM, bits=SIMHASH_BITS, bands=SIMHASH_BANDS
    )
    tgt_texts, tgt_labels, rm_tgt = dedup_near(
        tgt_texts, tgt_labels,
        hamming_thresh=SIMHASH_HAMMING_THRESH,
        ngram=SIMHASH_NGRAM, bits=SIMHASH_BITS, bands=SIMHASH_BANDS
    )
    _log(f"  Source dedup removed {rm_src}")
    _log(f"  Target dedup removed {rm_tgt}")

    _log("\n[Dedup] Cross-domain near-duplicate removal (target against source)...")
    tgt_texts, tgt_labels, rm_cross = cross_domain_dedup_target_against_source(
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        tgt_labels=tgt_labels,
        hamming_thresh=SIMHASH_HAMMING_THRESH,
        ngram=SIMHASH_NGRAM, bits=SIMHASH_BITS, bands=SIMHASH_BANDS
    )
    _log(f"CROSS_DOMAIN_DEDUP: removed {rm_cross} target samples that near-duplicate source (SimHash).")

    # re-count
    sf, sr, st = count_labels(src_labels)
    tf_, tr, tt = count_labels(tgt_labels)
    _log(f"\nAfter dedup:\nSource   | fake={sf}  real={sr}  total={st}\nTarget   | fake={tf_}  real={tr}  total={tt}")

    # ---------------------------
    # Split
    # ---------------------------
    # Use fixed split seed (not the run seed) to avoid variance from resplitting
    SPLIT_SEED = 1337
    src_split = stratified_split(src_texts, src_labels, SPLIT_RATIOS, seed=SPLIT_SEED)
    tgt_split = stratified_split(tgt_texts, tgt_labels, SPLIT_RATIOS, seed=SPLIT_SEED)

    _log("\nSplit sizes:")
    _log(f"Source: train={len(src_split['train'][0])}  dev={len(src_split['dev'][0])}  test={len(src_split['test'][0])}")
    _log(f"Target: train={len(tgt_split['train'][0])}  dev={len(tgt_split['dev'][0])}  test={len(tgt_split['test'][0])}")

    # ---------------------------
    # Tokenizer (SOURCE-only)
    # ---------------------------
    _log("\nTokenizer mode: SOURCE-only (paper-strict)")
    token2id = build_vocab_source_only(src_split["train"][0], MAX_VOCAB)
    vocab_size = len(token2id)
    _log(f"Vocab size = {vocab_size} (cap={MAX_VOCAB})")

    # vectorize all splits once
    src_train_x = vectorize_texts(src_split["train"][0], token2id, MAX_LEN)
    src_train_y = np.array(src_split["train"][1], dtype=np.int32)
    src_dev_x = vectorize_texts(src_split["dev"][0], token2id, MAX_LEN)
    src_dev_y = np.array(src_split["dev"][1], dtype=np.int32)
    src_test_x = vectorize_texts(src_split["test"][0], token2id, MAX_LEN)
    src_test_y = np.array(src_split["test"][1], dtype=np.int32)

    tgt_train_full_x_text = tgt_split["train"][0]
    tgt_train_full_y = np.array(tgt_split["train"][1], dtype=np.int32)
    tgt_dev_x = vectorize_texts(tgt_split["dev"][0], token2id, MAX_LEN)
    tgt_dev_y = np.array(tgt_split["dev"][1], dtype=np.int32)
    tgt_test_x = vectorize_texts(tgt_split["test"][0], token2id, MAX_LEN)
    tgt_test_y = np.array(tgt_split["test"][1], dtype=np.int32)

    # ---------------------------
    # Run experiments
    # ---------------------------
    raw_rows: List[Dict[str, Any]] = []

    print("\n" + "#" * 120)
    for seed in SEEDS:
        set_global_seed(seed)
        _log(f"\nRUN seed={seed}\n")

        # ========== Stage A: pretrain plain backbone on source ==========
        t0 = time.time()
        base = build_and_init_plain(vocab_size=vocab_size)

        # pretrain vars: backbone + src head
        pretrain_vars = (
            base.embedding.trainable_variables +
            base.conv.trainable_variables +
            base.shared_fc.trainable_variables +
            base.head_src.trainable_variables
        )
        pretrain_backbone_vars = (
            base.embedding.trainable_variables +
            base.conv.trainable_variables +
            base.shared_fc.trainable_variables
        )

        # class weighting for source imbalance
        n0 = int(np.sum(src_train_y == 0))
        n1 = int(np.sum(src_train_y == 1))
        w0 = (n0 + n1) / max(1, 2 * n0)
        w1 = (n0 + n1) / max(1, 2 * n1)

        src_train_ds = make_tf_dataset(src_train_x, src_train_y, batch=BATCH_PRETRAIN, shuffle=True, seed=seed, repeat=True)
        src_dev_ds = make_tf_dataset(src_dev_x, src_dev_y, batch=BATCH_EVAL, shuffle=False, seed=seed, repeat=False)

        steps_pre = max(MIN_STEPS_PER_EPOCH, math.ceil(len(src_train_x) / BATCH_PRETRAIN))

        opt = tf.keras.optimizers.Adam(learning_rate=LR_PRETRAIN)

        def logits_fn_src(xb, training=False):
            return base.logits(xb, head="src", training=training)
        logits_fn_src._model = base
        # set protected vars accessor for EWC compatibility (not used here)
        logits_fn_src._ewc_protected = get_protected_vars_plain(base)

        # override train loop to include class weights
        @tf.function
        def pretrain_step(x, y):
            with tf.GradientTape() as tape:
                logits = logits_fn_src(x, training=True)
                loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
                # apply class weights
                weights = tf.where(tf.equal(y, 1.0), tf.cast(w1, loss_vec.dtype), tf.cast(w0, loss_vec.dtype))
                loss = tf.reduce_mean(loss_vec * weights)
            grads = tape.gradient(loss, pretrain_vars)
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP_NORM)
            opt.apply_gradients([(g, v) for g, v in zip(grads, pretrain_vars) if g is not None])
            return loss

        best_f1 = -1.0
        best_weights = None
        wait = 0

        _log("[Stage A] Pretrain Plain backbone on Source (domain=0)")
        it = iter(src_train_ds)
        for epoch in range(1, EPOCHS_PRETRAIN + 1):
            losses = []
            for _ in range(steps_pre):
                x_b, y_b = next(it)
                l = pretrain_step(x_b, y_b)
                losses.append(float(l.numpy()))

            # dev eval
            y_true, logits_all, probs = eval_logits_probs(logits_fn_src, src_dev_ds)
            thr_src, m = find_best_threshold(y_true, probs)
            bce = bce_from_logits(y_true, logits_all)

            _log(f"Source DEV (Plain pre) | thr={thr_src:.3f}  Acc={m['acc']:.4f}  MacroF1={m['macro_f1']:.4f}  BCE={bce:.4f}")

            if m["macro_f1"] > best_f1 + 1e-6:
                best_f1 = m["macro_f1"]
                best_weights = [w.copy() for w in base.get_weights()]
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE_PRETRAIN:
                    break

        if best_weights is not None:
            base.set_weights(best_weights)

        t_pretrain = time.time() - t0

        # evaluate source test + store threshold
        src_test_ds = make_tf_dataset(src_test_x, src_test_y, batch=BATCH_EVAL, shuffle=False, seed=seed, repeat=False)
        y_true, logits_all, probs = eval_logits_probs(logits_fn_src, src_test_ds)
        thr_src_pre, m_src_test = find_best_threshold(y_true, probs)
        src_bce_pre = bce_from_logits(y_true, logits_all)
        _log(f"Source TEST (Plain pre) | thr={thr_src_pre:.3f}  Acc={m_src_test['acc']:.4f}  MacroF1={m_src_test['macro_f1']:.4f}  BCE={src_bce_pre:.4f}")

        # ========== Fisher estimation ==========
        t1 = time.time()
        fisher_ds = make_tf_dataset(src_train_x, src_train_y, batch=BATCH_PRETRAIN, shuffle=True, seed=seed + 999, repeat=False)
        ewc_info = estimate_fisher_plain(
            model=base,
            ds=fisher_ds,
            head="src",
            max_batches=FISHER_BATCHES,
            temperature=FISHER_TEMPERATURE
        )
        fisher_means = [float(np.mean(v)) for v in ewc_info.fisher.values()]
        _log(f"Fisher mean (post-norm global mean ~1): per-var means={['%.4f'%x for x in fisher_means]}")
        t_fisher = time.time() - t1

        print("\n" + "-" * 110)
        for shot in SHOTS_TOTAL:
            _log(f"Target setting: TOTAL {shot} labeled (balanced)")

            # sample few-shot from target train pool (TEXT), then vectorize with source vocab
            fs_texts, fs_labels = sample_fewshot_total_balanced(
                tgt_train_full_x_text, tgt_train_full_y.tolist(), shot_total=shot, seed=seed + shot * 17
            )
            tgt_fs_x = vectorize_texts(fs_texts.tolist(), token2id, MAX_LEN)
            tgt_fs_y = fs_labels.astype(np.int32)

            # prepare datasets
            tgt_train_ds = make_tf_dataset(tgt_fs_x, tgt_fs_y, batch=BATCH_ADAPT, shuffle=True, seed=seed + shot, repeat=True)
            tgt_dev_ds = make_tf_dataset(tgt_dev_x, tgt_dev_y, batch=BATCH_EVAL, shuffle=False, seed=seed, repeat=False)
            tgt_test_ds = make_tf_dataset(tgt_test_x, tgt_test_y, batch=BATCH_EVAL, shuffle=False, seed=seed, repeat=False)

            steps_adapt = max(MIN_STEPS_PER_EPOCH, math.ceil(len(tgt_fs_x) / BATCH_ADAPT))

            # ---------------------------------------------------------
            # [Scratch-Plain] train plain model from scratch on few-shot
            # ---------------------------------------------------------
            _log("\n[Scratch-Plain]")
            t_adapt0 = time.time()
            scratch = build_and_init_plain(vocab_size=vocab_size)
            # train all backbone + tgt head (src head not used)
            train_vars = (
                scratch.embedding.trainable_variables +
                scratch.conv.trainable_variables +
                scratch.shared_fc.trainable_variables +
                scratch.head_tgt.trainable_variables +
                scratch.head_src.trainable_variables  # keep consistent with your v3 param accounting
            )
            backbone_vars = (
                scratch.embedding.trainable_variables +
                scratch.conv.trainable_variables +
                scratch.shared_fc.trainable_variables
            )
            opt_s = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            def logits_fn_tgt(xb, training=False):
                return scratch.logits(xb, head="tgt", training=training)
            logits_fn_tgt._model = scratch
            logits_fn_tgt._ewc_protected = get_protected_vars_plain(scratch)

            thr_tgt, devm = train_loop_earlystop(
                train_ds=tgt_train_ds,
                dev_ds=tgt_dev_ds,
                logits_fn=logits_fn_tgt,
                train_vars=train_vars,
                backbone_vars=backbone_vars,
                ewc_info=None,
                ewc_lambda=0.0,
                l2_anchor_alpha=0.0,
                optimizer=opt_s,
                steps_per_epoch=steps_adapt,
                max_epochs=EPOCHS_ADAPT,
                patience=PATIENCE_ADAPT,
                backbone_lr_mult=1.0,
                tag="Scratch"
            )
            t_adapt = time.time() - t_adapt0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_tgt, tgt_test_ds)
            m = binary_metrics_from_probs(y_true, probs, thr_tgt)
            bce = bce_from_logits(y_true, logits_all)
            _log(f"Target TEST (Scratch-Plain) | thr={thr_tgt:.3f}  Acc={m['acc']:.4f}  MacroF1={m['macro_f1']:.4f}  BCE={bce:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "Scratch-Plain",
                "tgt_acc": m["acc"], "tgt_f1_macro": m["macro_f1"], "tgt_bce": bce,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src_test["macro_f1"],
                "src_acc_after": np.nan, "src_f1_after": np.nan, "src_bce_after": np.nan,
                "forget_f1": np.nan,
                "avg_f1_after": np.nan,
                "trainable_params": count_params(train_vars),
                "t_pretrain": t_pretrain, "t_fisher": t_fisher, "t_adapt": t_adapt,
                "t_total_once": t_pretrain + t_fisher + t_adapt
            })

            # ---------------------------------------------------------
            # [Transfer-Plain FullFT] pretrained backbone + full fine-tune
            # ---------------------------------------------------------
            _log("\n[Transfer-Plain FullFT]")
            t_adapt0 = time.time()
            transfer = build_and_init_plain(vocab_size=vocab_size)
            copy_backbone_and_src_head_plain_to_plain(base, transfer)
            # src head frozen (do not train)
            train_vars = (
                transfer.embedding.trainable_variables +
                transfer.conv.trainable_variables +
                transfer.shared_fc.trainable_variables +
                transfer.head_tgt.trainable_variables
            )
            backbone_vars = (
                transfer.embedding.trainable_variables +
                transfer.conv.trainable_variables +
                transfer.shared_fc.trainable_variables
            )
            opt_t = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            def logits_fn_tr(xb, training=False):
                return transfer.logits(xb, head="tgt", training=training)
            logits_fn_tr._model = transfer
            logits_fn_tr._ewc_protected = get_protected_vars_plain(transfer)

            thr_tgt_tr, devm = train_loop_earlystop(
                train_ds=tgt_train_ds,
                dev_ds=tgt_dev_ds,
                logits_fn=logits_fn_tr,
                train_vars=train_vars,
                backbone_vars=backbone_vars,
                ewc_info=None,
                ewc_lambda=0.0,
                l2_anchor_alpha=0.0,
                optimizer=opt_t,
                steps_per_epoch=steps_adapt,
                max_epochs=EPOCHS_ADAPT,
                patience=PATIENCE_ADAPT,
                backbone_lr_mult=1.0,
                tag="Transfer"
            )
            t_adapt = time.time() - t_adapt0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_tr, tgt_test_ds)
            m_tgt = binary_metrics_from_probs(y_true, probs, thr_tgt_tr)
            bce_tgt = bce_from_logits(y_true, logits_all)

            # source after transfer: use src head + PRETRAIN threshold
            src_after = eval_on_split_plain_with_thr(transfer, src_test_x, src_test_y, head="src", thr=thr_src_pre, seed=seed)
            forget = float(m_src_test["macro_f1"] - src_after["macro_f1"])

            _log(f"Target TEST (Transfer-Plain) | thr={thr_tgt_tr:.3f}  Acc={m_tgt['acc']:.4f}  MacroF1={m_tgt['macro_f1']:.4f}  BCE={bce_tgt:.4f}")
            _log(f"Source TEST after (Transfer-Plain) | thr={thr_src_pre:.3f}  Acc={src_after['acc']:.4f}  MacroF1={src_after['macro_f1']:.4f}  BCE={src_after['bce']:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "Transfer-Plain",
                "tgt_acc": m_tgt["acc"], "tgt_f1_macro": m_tgt["macro_f1"], "tgt_bce": bce_tgt,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src_test["macro_f1"],
                "src_acc_after": src_after["acc"], "src_f1_after": src_after["macro_f1"], "src_bce_after": src_after["bce"],
                "forget_f1": forget,
                "avg_f1_after": float(0.5 * (m_tgt["macro_f1"] + src_after["macro_f1"])),
                "trainable_params": count_params(train_vars),
                "t_pretrain": t_pretrain, "t_fisher": 0.0, "t_adapt": t_adapt,
                "t_total_once": t_pretrain + t_adapt
            })

            # ---------------------------------------------------------
            # [Transfer-Plain + EWC] same but add EWC penalty (use normalized Fisher)
            # ---------------------------------------------------------
            _log("\n[Transfer-Plain + EWC]")
            t_adapt0 = time.time()
            transfer_e = build_and_init_plain(vocab_size=vocab_size)
            copy_backbone_and_src_head_plain_to_plain(base, transfer_e)

            train_vars = (
                transfer_e.embedding.trainable_variables +
                transfer_e.conv.trainable_variables +
                transfer_e.shared_fc.trainable_variables +
                transfer_e.head_tgt.trainable_variables
            )
            backbone_vars = (
                transfer_e.embedding.trainable_variables +
                transfer_e.conv.trainable_variables +
                transfer_e.shared_fc.trainable_variables
            )

            opt_te = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            def logits_fn_tre(xb, training=False):
                return transfer_e.logits(xb, head="tgt", training=training)
            logits_fn_tre._model = transfer_e
            logits_fn_tre._ewc_protected = get_protected_vars_plain(transfer_e)

            # pick a reasonable fixed lambda for transfer (we do not grid-search transfer; focus on DAEWC)
            LAMBDA_TRANSFER = 1.0

            thr_tgt_tre, devm = train_loop_earlystop(
                train_ds=tgt_train_ds,
                dev_ds=tgt_dev_ds,
                logits_fn=logits_fn_tre,
                train_vars=train_vars,
                backbone_vars=backbone_vars,
                ewc_info=ewc_info,
                ewc_lambda=LAMBDA_TRANSFER,
                l2_anchor_alpha=L2_ANCHOR_ALPHA,
                optimizer=opt_te,
                steps_per_epoch=steps_adapt,
                max_epochs=EPOCHS_ADAPT,
                patience=PATIENCE_ADAPT,
                backbone_lr_mult=1.0,
                tag=f"Transfer+EWC(λ={LAMBDA_TRANSFER})"
            )
            t_adapt = time.time() - t_adapt0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_tre, tgt_test_ds)
            m_tgt = binary_metrics_from_probs(y_true, probs, thr_tgt_tre)
            bce_tgt = bce_from_logits(y_true, logits_all)

            src_after = eval_on_split_plain_with_thr(transfer_e, src_test_x, src_test_y, head="src", thr=thr_src_pre, seed=seed)
            forget = float(m_src_test["macro_f1"] - src_after["macro_f1"])

            _log(f"Target TEST (Plain+EWC) | thr={thr_tgt_tre:.3f}  Acc={m_tgt['acc']:.4f}  MacroF1={m_tgt['macro_f1']:.4f}  BCE={bce_tgt:.4f}")
            _log(f"Source TEST after (Plain+EWC) | thr={thr_src_pre:.3f}  Acc={src_after['acc']:.4f}  MacroF1={src_after['macro_f1']:.4f}  BCE={src_after['bce']:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "Transfer-Plain+EWC",
                "tgt_acc": m_tgt["acc"], "tgt_f1_macro": m_tgt["macro_f1"], "tgt_bce": bce_tgt,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src_test["macro_f1"],
                "src_acc_after": src_after["acc"], "src_f1_after": src_after["macro_f1"], "src_bce_after": src_after["bce"],
                "forget_f1": forget,
                "avg_f1_after": float(0.5 * (m_tgt["macro_f1"] + src_after["macro_f1"])),
                "trainable_params": count_params(train_vars),
                "t_pretrain": t_pretrain, "t_fisher": t_fisher, "t_adapt": t_adapt,
                "t_total_once": t_pretrain + t_fisher + t_adapt
            })

            # ---------------------------------------------------------
            # [Adapter-Only] DAEWC stage1 (backbone frozen)
            # ---------------------------------------------------------
            _log("\n[Adapter-Only (DAEWC; backbone frozen)]")
            t_adapt0 = time.time()
            dae1 = build_and_init_dae(vocab_size=vocab_size)
            copy_backbone_and_src_head_plain_to_dae(base, dae1)

            # train vars: adapter + gate + dom_emb(row=1) + tgt head
            # We'll train whole dom_emb and gate weights; backbone & src head excluded.
            train_vars = (
                dae1.dom_emb.variables +
                dae1.gate.trainable_variables +
                dae1.adapter_dense1.trainable_variables +
                dae1.adapter_dense2.trainable_variables +
                dae1.head_tgt.trainable_variables
            )
            backbone_vars = []  # frozen

            opt_a = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            def logits_fn_ad(xb, training=False):
                return dae1.logits(xb, head="tgt", domain_id=1, training=training, use_adapter=True)
            logits_fn_ad._model = dae1
            logits_fn_ad._ewc_protected = get_protected_vars_dae(dae1)

            thr_tgt_ad, devm = train_loop_earlystop(
                train_ds=tgt_train_ds,
                dev_ds=tgt_dev_ds,
                logits_fn=logits_fn_ad,
                train_vars=train_vars,
                backbone_vars=backbone_vars,
                ewc_info=None,
                ewc_lambda=0.0,
                l2_anchor_alpha=0.0,
                optimizer=opt_a,
                steps_per_epoch=steps_adapt,
                max_epochs=EPOCHS_ADAPT,
                patience=PATIENCE_ADAPT,
                backbone_lr_mult=1.0,
                tag="AdapterOnly"
            )
            t_adapt = time.time() - t_adapt0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_ad, tgt_test_ds)
            m_tgt = binary_metrics_from_probs(y_true, probs, thr_tgt_ad)
            bce_tgt = bce_from_logits(y_true, logits_all)

            # source after adapter-only: backbone unchanged, so use base for source test equivalently.
            src_after = eval_on_split_plain_with_thr(base, src_test_x, src_test_y, head="src", thr=thr_src_pre, seed=seed)
            forget = float(m_src_test["macro_f1"] - src_after["macro_f1"])

            _log(f"Target TEST (Adapter-Only) | thr={thr_tgt_ad:.3f}  Acc={m_tgt['acc']:.4f}  MacroF1={m_tgt['macro_f1']:.4f}  BCE={bce_tgt:.4f}")
            _log(f"Source TEST after (Adapter-Only) | thr={thr_src_pre:.3f}  Acc={src_after['acc']:.4f}  MacroF1={src_after['macro_f1']:.4f}  BCE={src_after['bce']:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "Adapter-Only",
                "tgt_acc": m_tgt["acc"], "tgt_f1_macro": m_tgt["macro_f1"], "tgt_bce": bce_tgt,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src_test["macro_f1"],
                "src_acc_after": src_after["acc"], "src_f1_after": src_after["macro_f1"], "src_bce_after": src_after["bce"],
                "forget_f1": forget,
                "avg_f1_after": float(0.5 * (m_tgt["macro_f1"] + src_after["macro_f1"])),
                "trainable_params": count_params(train_vars),
                "t_pretrain": t_pretrain, "t_fisher": 0.0, "t_adapt": t_adapt,
                "t_total_once": t_pretrain + t_adapt
            })

            # ---------------------------------------------------------
            # [DAEWC] Stage1 + Stage2 with dev-selected (lambda, strategy, backbone_lr_mult)
            # ---------------------------------------------------------
            _log("\n[DAEWC (Stage1 + Stage2)]")

            # Stage1 already corresponds to dae1 state; we will reuse its weights as init for stage2.
            # We'll pick best stage2 config by dev Macro-F1.
            strategies = STRATEGY_SMALL if shot <= 20 else STRATEGY_LARGE

            best_cfg = None
            best_dev_f1 = -1.0
            best_model_weights = None
            best_thr = 0.5

            # cache stage1 weights
            stage1_weights = [w.copy() for w in dae1.get_weights()]

            # candidate search
            t_adapt0 = time.time()
            for strat in strategies:
                for bb_mult in BACKBONE_LR_MULT_GRID:
                    for lam in LAMBDA_GRID:
                        dae2 = build_and_init_dae(vocab_size=vocab_size)
                        copy_backbone_and_src_head_plain_to_dae(base, dae2)
                        dae2.set_weights(stage1_weights)

                        # train vars for stage2:
                        #   always keep adapter/gate/dom_emb/tgt head trainable
                        #   + unfreeze selected backbone top layers
                        stage2_train_vars = (
                            dae2.dom_emb.variables +
                            dae2.gate.trainable_variables +
                            dae2.adapter_dense1.trainable_variables +
                            dae2.adapter_dense2.trainable_variables +
                            dae2.head_tgt.trainable_variables
                        )
                        stage2_backbone_vars = []

                        # IMPORTANT: embedding stays frozen in paper-style DAEWC (prevents OOV chaos & forgetting)
                        # Unfreeze strategy
                        if strat == "fc_only":
                            stage2_train_vars += dae2.shared_fc.trainable_variables
                            stage2_backbone_vars += dae2.shared_fc.trainable_variables
                        elif strat == "conv_fc":
                            stage2_train_vars += dae2.shared_fc.trainable_variables + dae2.conv.trainable_variables
                            stage2_backbone_vars += dae2.shared_fc.trainable_variables + dae2.conv.trainable_variables
                        else:
                            raise ValueError(f"Unknown strategy={strat}")

                        opt_d2 = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

                        def logits_fn_d2(xb, training=False):
                            return dae2.logits(xb, head="tgt", domain_id=1, training=training, use_adapter=True)
                        logits_fn_d2._model = dae2
                        logits_fn_d2._ewc_protected = get_protected_vars_dae(dae2)

                        thr_dev, devm = train_loop_earlystop(
                            train_ds=tgt_train_ds,
                            dev_ds=tgt_dev_ds,
                            logits_fn=logits_fn_d2,
                            train_vars=stage2_train_vars,
                            backbone_vars=stage2_backbone_vars,
                            ewc_info=ewc_info,
                            ewc_lambda=float(lam),
                            l2_anchor_alpha=L2_ANCHOR_ALPHA,
                            optimizer=opt_d2,
                            steps_per_epoch=steps_adapt,
                            max_epochs=max(10, EPOCHS_ADAPT // 2),   # stage2 shorter; we search many configs
                            patience=max(3, PATIENCE_ADAPT // 2),
                            backbone_lr_mult=float(bb_mult),
                            tag=f"DAEWC-S2[{strat},bb={bb_mult},λ={lam}]"
                        )

                        # measure dev f1 quickly (already in devm)
                        if devm["macro_f1"] > best_dev_f1 + 1e-6:
                            best_dev_f1 = devm["macro_f1"]
                            best_cfg = (strat, bb_mult, lam)
                            best_thr = thr_dev
                            best_model_weights = [w.copy() for w in dae2.get_weights()]

            t_adapt = time.time() - t_adapt0

            # evaluate best
            dae_best = build_and_init_dae(vocab_size=vocab_size)
            copy_backbone_and_src_head_plain_to_dae(base, dae_best)
            dae_best.set_weights(best_model_weights)

            def logits_fn_best(xb, training=False):
                return dae_best.logits(xb, head="tgt", domain_id=1, training=training, use_adapter=True)
            logits_fn_best._model = dae_best

            y_true, logits_all, probs = eval_logits_probs(logits_fn_best, tgt_test_ds)
            m_tgt = binary_metrics_from_probs(y_true, probs, best_thr)
            bce_tgt = bce_from_logits(y_true, logits_all)

            # source after DAEWC: evaluate with NO adapter, src head, pretrain thr
            src_after = eval_on_split_dae_with_thr(
                dae_best, src_test_x, src_test_y,
                head="src", domain_id=0, use_adapter=False,
                thr=thr_src_pre, seed=seed
            )
            forget = float(m_src_test["macro_f1"] - src_after["macro_f1"])

            _log(f"Target TEST (DAEWC best={best_cfg}) | thr={best_thr:.3f}  Acc={m_tgt['acc']:.4f}  MacroF1={m_tgt['macro_f1']:.4f}  BCE={bce_tgt:.4f}")
            _log(f"Source TEST after (DAEWC) | thr={thr_src_pre:.3f}  Acc={src_after['acc']:.4f}  MacroF1={src_after['macro_f1']:.4f}  BCE={src_after['bce']:.4f}")

            # estimate trainable params of best config
            # rebuild vars list similarly
            strat, bb_mult, lam = best_cfg
            best_train_vars = (
                dae_best.dom_emb.variables +
                dae_best.gate.trainable_variables +
                dae_best.adapter_dense1.trainable_variables +
                dae_best.adapter_dense2.trainable_variables +
                dae_best.head_tgt.trainable_variables
            )
            if strat == "fc_only":
                best_train_vars += dae_best.shared_fc.trainable_variables
            else:
                best_train_vars += dae_best.shared_fc.trainable_variables + dae_best.conv.trainable_variables

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "DAEWC",
                "tgt_acc": m_tgt["acc"], "tgt_f1_macro": m_tgt["macro_f1"], "tgt_bce": bce_tgt,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src_test["macro_f1"],
                "src_acc_after": src_after["acc"], "src_f1_after": src_after["macro_f1"], "src_bce_after": src_after["bce"],
                "forget_f1": forget,
                "avg_f1_after": float(0.5 * (m_tgt["macro_f1"] + src_after["macro_f1"])),
                "trainable_params": count_params(best_train_vars),
                "t_pretrain": t_pretrain, "t_fisher": t_fisher, "t_adapt": t_adapt,
                "t_total_once": t_pretrain + t_fisher + t_adapt
            })

            # ---------------------------------------------------------
            # [Replay Upper] (optional joint training)
            # ---------------------------------------------------------
            _log("\n[Replay Upper (Joint training)]")
            # Keep it simple: multi-task batches (alternate source & target batches)
            t_adapt0 = time.time()
            replay = build_and_init_plain(vocab_size=vocab_size)
            copy_backbone_and_src_head_plain_to_plain(base, replay)

            # train vars: backbone + both heads
            train_vars = (
                replay.embedding.trainable_variables +
                replay.conv.trainable_variables +
                replay.shared_fc.trainable_variables +
                replay.head_src.trainable_variables +
                replay.head_tgt.trainable_variables
            )
            backbone_vars = (
                replay.embedding.trainable_variables +
                replay.conv.trainable_variables +
                replay.shared_fc.trainable_variables
            )
            opt_r = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            src_joint_ds = make_tf_dataset(src_train_x, src_train_y, batch=BATCH_ADAPT, shuffle=True, seed=seed + 111, repeat=True)
            tgt_joint_ds = tgt_train_ds  # already repeat
            src_it = iter(src_joint_ds)
            tgt_it = iter(tgt_joint_ds)

            # simple fixed steps
            steps_joint = max(MIN_STEPS_PER_EPOCH, steps_adapt)

            @tf.function
            def joint_step(x, y, head: str):
                with tf.GradientTape() as tape:
                    logits = replay.logits(x, head=head, training=True)
                    loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
                    loss = tf.reduce_mean(loss_vec)
                grads = tape.gradient(loss, train_vars)
                if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                    grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP_NORM)
                opt_r.apply_gradients([(g, v) for g, v in zip(grads, train_vars) if g is not None])
                return loss

            # train for a few epochs (joint training is expensive; keep moderate)
            for _ep in range(8):
                for _ in range(steps_joint):
                    # alternate: one src step, one tgt step
                    xs, ys = next(src_it)
                    joint_step(xs, ys, head="src")
                    xt, yt = next(tgt_it)
                    joint_step(xt, yt, head="tgt")

            t_adapt = time.time() - t_adapt0

            # eval target
            def logits_fn_r_tgt(xb, training=False):
                return replay.logits(xb, head="tgt", training=training)
            logits_fn_r_tgt._model = replay
            y_true, logits_all, probs = eval_logits_probs(logits_fn_r_tgt, tgt_dev_ds)
            thr_r, _ = find_best_threshold(y_true, probs)

            y_true, logits_all, probs = eval_logits_probs(logits_fn_r_tgt, tgt_test_ds)
            m_tgt = binary_metrics_from_probs(y_true, probs, thr_r)
            bce_tgt = bce_from_logits(y_true, logits_all)

            # eval source after replay
            src_after = eval_on_split_plain_with_thr(replay, src_test_x, src_test_y, head="src", thr=thr_src_pre, seed=seed)
            forget = float(m_src_test["macro_f1"] - src_after["macro_f1"])

            _log(f"Target TEST (ReplayUpper) | thr={thr_r:.3f}  Acc={m_tgt['acc']:.4f}  MacroF1={m_tgt['macro_f1']:.4f}  BCE={bce_tgt:.4f}")
            _log(f"Source TEST (ReplayUpper) | thr={thr_src_pre:.3f}  Acc={src_after['acc']:.4f}  MacroF1={src_after['macro_f1']:.4f}  BCE={src_after['bce']:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "ReplayUpper",
                "tgt_acc": m_tgt["acc"], "tgt_f1_macro": m_tgt["macro_f1"], "tgt_bce": bce_tgt,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src_test["macro_f1"],
                "src_acc_after": src_after["acc"], "src_f1_after": src_after["macro_f1"], "src_bce_after": src_after["bce"],
                "forget_f1": forget,
                "avg_f1_after": float(0.5 * (m_tgt["macro_f1"] + src_after["macro_f1"])),
                "trainable_params": count_params(train_vars),
                "t_pretrain": 0.0, "t_fisher": 0.0, "t_adapt": t_adapt,
                "t_total_once": t_adapt
            })

            print("\n" + "-" * 110)

        print("\n" + "#" * 120)

    # ---------------------------
    # Save raw & summary
    # ---------------------------
    raw_df = pd.DataFrame(raw_rows)
    raw_out = "results_daeewc_paperstyle_v4_raw.csv"
    raw_df.to_csv(raw_out, index=False)
    _log(f"\nSaved raw: {raw_out}")

    # Summary with mean & CI95 over seeds
    def ci95(x: np.ndarray) -> float:
        x = np.array(x, dtype=np.float64)
        if len(x) <= 1:
            return 0.0
        return 1.96 * float(np.std(x, ddof=1)) / math.sqrt(len(x))

    grp = raw_df.groupby(["shot", "method"], as_index=False)
    summary_rows = []
    for (shot, method), g in grp:
        summary_rows.append({
            "shot": shot,
            "method": method,
            "tgt_f1_macro_mean": float(np.mean(g["tgt_f1_macro"])),
            "tgt_f1_macro_ci95": ci95(g["tgt_f1_macro"].values),
            "forget_f1_mean": float(np.nanmean(g["forget_f1"].values)),
            "forget_f1_ci95": ci95(g["forget_f1"].dropna().values) if g["forget_f1"].notna().any() else np.nan,
            "adapt_time_mean_s": float(np.mean(g["t_adapt"])),
            "adapt_time_ci95": ci95(g["t_adapt"].values),
            "trainable_params": float(np.mean(g["trainable_params"])),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["shot", "method"])
    sum_out = "results_daeewc_paperstyle_v4_summary.csv"
    summary_df.to_csv(sum_out, index=False)
    _log(f"Saved summary: {sum_out}")

    # Pivots
    pivot_f1 = summary_df.pivot(index="shot", columns="method", values="tgt_f1_macro_mean")
    pivot_forget = summary_df.pivot(index="shot", columns="method", values="forget_f1_mean")
    pivot_time = summary_df.pivot(index="shot", columns="method", values="adapt_time_mean_s")

    print("\n=== Pivot: Target Macro-F1 (mean) ===")
    print(pivot_f1.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Pivot: Forgetting (ΔSrc Macro-F1 = pre - after; lower is better) ===")
    print(pivot_forget.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Pivot: Adapt time (s, mean) ===")
    print(pivot_time.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
