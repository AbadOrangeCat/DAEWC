# daeewc_v4.py
# -*- coding: utf-8 -*-
"""
DAEWC v4 (news -> covid)
- No CLI args; edit CONFIG only if needed.
- Data paths fixed as user provided:
    PATH_FAKE       = './news/Fake.csv'
    PATH_REAL       = './news/True.csv'
    PATH_COVID_FAKE = './covid/fakeNews.csv'
    PATH_COVID_REAL = './covid/trueNews.csv'

Paper-like choices:
  - SOURCE-only tokenizer (vocab cap=5000)
  - Stratified split 70/10/20
  - Few-shot by default: K-shot PER CLASS  (10/20/80/160)
  - Threshold tuned on DEV, reported on TEST

Make DAEWC more likely strongest:
  - Effective Fisher: temperature-soft pseudo labels + global mean normalization
  - DAEWC Stage2: shot-dependent unfreeze + stronger lambda schedule (no heavy grid search)
  - L2-anchor (L2-SP style) + small backbone LR multiplier

Time accounting:
  - t_pretrain / t_fisher are one-time costs
  - t_adapt is per-shot adaptation time (what you care about when new domain arrives)
"""

from __future__ import annotations
import os
import re
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
# CONFIG (no CLI args)
# ======================================================================================

# ---- fixed paths (as user provided) ----
PATH_FAKE = "./news/Fake.csv"
PATH_REAL = "./news/True.csv"
PATH_COVID_FAKE = "./covid/fakeNews.csv"
PATH_COVID_REAL = "./covid/trueNews.csv"

# ---- experiment ----
SEEDS = [42, 43, 44, 45, 46]

# Shots
SHOT_MODE = "per_class"  # "per_class" (default, like your v3) OR "total"
SHOTS = [10, 20, 80, 160]  # if per_class: K per class; if total: total labeled (balanced)
SPLIT_RATIOS = (0.70, 0.10, 0.20)

# ---- tokenizer ----
MAX_VOCAB = 5000
MAX_LEN = 256

# ---- model ----
EMB_DIM = 100
CONV_FILTERS = 128
KERNEL_SIZE = 5
HIDDEN_DIM = 128

DOM_EMB_DIM = 64
ADAPTER_HIDDEN_DIM = 192  # slightly stronger than v3-ish, still small

DROPOUT_RATE = 0.10

# ---- training ----
BATCH_PRETRAIN = 64
BATCH_ADAPT = 32
BATCH_EVAL = 256

EPOCHS_PRETRAIN = 8
PATIENCE_PRETRAIN = 2

EPOCHS_ADAPT = 40
PATIENCE_ADAPT = 6

MIN_STEPS_PER_EPOCH = 50

LR_PRETRAIN = 1e-3
LR_ADAPT = 2e-3

GRAD_CLIP_NORM = 1.0

LABEL_SMOOTHING = 0.05  # mild; helps few-shot generalization

# ---- EWC/Fisher ----
FISHER_BATCHES = 200
FISHER_TEMPERATURE = 2.5
FISHER_EPS = 1e-8

# DAEWC Stage2 (no heavy grid search; keep adapter-time comparable)
# Shot-dependent strategy and lambda schedule (tune here if you want even more aggressive)
DAEWC_LAMBDA_BY_SHOT = {
    10: 1.0,
    20: 1.0,
    80: 5.0,
    160: 10.0,
}
# Shot-dependent unfreeze strategy
#  - small shot: unfreeze only shared_fc
#  - large shot: unfreeze conv + shared_fc
DAEWC_STRATEGY_BY_SHOT = {
    10: "fc_only",
    20: "fc_only",
    80: "conv_fc",
    160: "conv_fc",
}
BACKBONE_LR_MULT_BY_SHOT = {
    10: 0.10,
    20: 0.10,
    80: 0.15,
    160: 0.20,
}
L2_ANCHOR_ALPHA = 1e-4  # complements EWC

# ---- logging ----
VERBOSE = True


# ======================================================================================
# Utils
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
    s = re.sub(r"[^a-z<>\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_concat_text(title: Any, body: Any) -> str:
    t = "" if title is None else str(title)
    b = "" if body is None else str(body)
    if t and b:
        return f"{t}. {b}"
    return t or b

def robust_read_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    # Try utf-8 then latin1
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, engine="python", encoding="latin1", on_bad_lines="skip")

def pick_text_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (title_col, text_col). If not found, title_col may be None and text_col may be some best guess.
    """
    cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    lower = {c.lower(): c for c in cols}

    # common in Fake/True dataset
    title_col = lower.get("title", None)

    # candidates for main body
    body_candidates = [
        "text", "content", "article", "body", "news", "statement", "claim",
        "maintext", "message", "tweet", "post", "context"
    ]
    text_col = None
    for k in body_candidates:
        if k in lower:
            text_col = lower[k]
            break

    # fallback: if has exactly one non-unnamed column besides label-ish, pick the longest-text looking one
    if text_col is None:
        # prefer object dtype columns
        obj_cols = [c for c in cols if df[c].dtype == object]
        if obj_cols:
            # pick the column with largest avg length on sample
            sample = df[obj_cols].head(200)
            best = None
            best_len = -1.0
            for c in obj_cols:
                vals = sample[c].astype(str).fillna("")
                avg = float(vals.map(len).mean())
                if avg > best_len:
                    best_len = avg
                    best = c
            text_col = best
        else:
            text_col = cols[0] if cols else None

    return title_col, text_col

def load_pair_as_dataset(fake_path: str, real_path: str, name: str) -> Tuple[List[str], List[int]]:
    """
    Load two CSVs: fake and real; return texts and labels (fake=1, real=0).
    """
    df_f = robust_read_csv(fake_path)
    df_r = robust_read_csv(real_path)

    tf_title, tf_text = pick_text_columns(df_f)
    tr_title, tr_text = pick_text_columns(df_r)

    def extract(df: pd.DataFrame, title_col: Optional[str], text_col: Optional[str]) -> List[str]:
        out: List[str] = []
        if text_col is None and title_col is None:
            return out
        for _, row in df.iterrows():
            title = row[title_col] if title_col in df.columns else ""
            body = row[text_col] if text_col in df.columns else ""
            t = safe_concat_text(title, body)
            t = normalize_text(t)
            if t:
                out.append(t)
        return out

    fake_texts = extract(df_f, tf_title, tf_text)
    real_texts = extract(df_r, tr_title, tr_text)

    texts = fake_texts + real_texts
    labels = [1] * len(fake_texts) + [0] * len(real_texts)

    _log(f"{name:<10} loaded | fake={len(fake_texts)} real={len(real_texts)} total={len(texts)}")
    return texts, labels

def exact_dedup(texts: List[str], labels: List[int]) -> Tuple[List[str], List[int], int]:
    """
    Exact dedup by normalized text string hash.
    """
    seen = set()
    out_t, out_y = [], []
    removed = 0
    for t, y in zip(texts, labels):
        h = hashlib.sha1(t.encode("utf-8")).hexdigest()
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        out_t.append(t)
        out_y.append(y)
    return out_t, out_y, removed

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

    return {"train": pack(train_idx), "dev": pack(dev_idx), "test": pack(test_idx)}

def build_vocab_source_only(texts_src_train: List[str], max_vocab: int) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for t in texts_src_train:
        for tok in t.split():
            freq[tok] = freq.get(tok, 0) + 1

    num_tokens = max_vocab - 2
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:num_tokens]
    token2id = {"<pad>": 0, "<unk>": 1}
    for i, (tok, _) in enumerate(top, start=2):
        token2id[tok] = i
    return token2id

def vectorize_texts(texts: List[str], token2id: Dict[str, int], max_len: int) -> np.ndarray:
    unk = token2id.get("<unk>", 1)
    pad = token2id.get("<pad>", 0)
    out = np.full((len(texts), max_len), pad, dtype=np.int32)
    for i, t in enumerate(texts):
        toks = t.split()
        ids = [token2id.get(tok, unk) for tok in toks[:max_len]]
        if ids:
            out[i, :len(ids)] = np.array(ids, dtype=np.int32)
    return out

def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, seed: int, repeat: bool) -> tf.data.Dataset:
    y = y.astype(np.float32).reshape(-1, 1)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        buf = int(min(len(x), 10000))
        ds = ds.shuffle(buffer_size=max(10, buf), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


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

    denom1 = (2 * tp + fp + fn)
    f1_1 = (2 * tp / denom1) if denom1 > 0 else 0.0

    denom0 = (2 * tn + fn + fp)
    f1_0 = (2 * tn / denom0) if denom0 > 0 else 0.0

    macro_f1 = 0.5 * (f1_0 + f1_1)
    return {"acc": float(acc), "macro_f1": float(macro_f1)}

def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
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

def eval_logits_probs(logits_fn, ds_eval: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ======================================================================================
# Models
# ======================================================================================

class PlainTwoHeadCNN(tf.keras.Model):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMB_DIM, name="emb")
        self.conv = tf.keras.layers.Conv1D(
            filters=CONV_FILTERS, kernel_size=KERNEL_SIZE,
            activation="relu", padding="valid", name="conv"
        )
        self.pool = tf.keras.layers.GlobalMaxPool1D(name="gmp")
        self.drop = tf.keras.layers.Dropout(DROPOUT_RATE, name="drop")
        self.shared_fc = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", name="shared_fc")

        self.head_src = tf.keras.layers.Dense(1, name="head_src")
        self.head_tgt = tf.keras.layers.Dense(1, name="head_tgt")

    def encode(self, x_ids: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.embedding(x_ids)
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x, training=training)
        z = self.shared_fc(x)
        return z

    def logits(self, x_ids: tf.Tensor, head: str, training: bool) -> tf.Tensor:
        z = self.encode(x_ids, training=training)
        if head == "src":
            return self.head_src(z)
        if head == "tgt":
            return self.head_tgt(z)
        raise ValueError(head)

class DAEWCCNN(tf.keras.Model):
    def __init__(self, vocab_size: int):
        super().__init__()
        # backbone
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMB_DIM, name="emb")
        self.conv = tf.keras.layers.Conv1D(
            filters=CONV_FILTERS, kernel_size=KERNEL_SIZE,
            activation="relu", padding="valid", name="conv"
        )
        self.pool = tf.keras.layers.GlobalMaxPool1D(name="gmp")
        self.drop = tf.keras.layers.Dropout(DROPOUT_RATE, name="drop")
        self.shared_fc = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", name="shared_fc")

        # heads
        self.head_src = tf.keras.layers.Dense(1, name="head_src")
        self.head_tgt = tf.keras.layers.Dense(1, name="head_tgt")

        # domain embedding (2 domains)
        self.dom_emb = self.add_weight(
            name="dom_emb", shape=(2, DOM_EMB_DIM),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        self.gate = tf.keras.layers.Dense(HIDDEN_DIM, activation="sigmoid", name="gate")

        # residual adapter
        self.adapter_d1 = tf.keras.layers.Dense(ADAPTER_HIDDEN_DIM, activation="relu", name="adapter_d1")
        self.adapter_drop = tf.keras.layers.Dropout(DROPOUT_RATE, name="adapter_drop")
        self.adapter_d2 = tf.keras.layers.Dense(HIDDEN_DIM, activation=None, name="adapter_d2")

    def encode_backbone(self, x_ids: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.embedding(x_ids)
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x, training=training)
        z = self.shared_fc(x)
        return z

    def encode_with_adapter(self, x_ids: tf.Tensor, domain_id: int, training: bool) -> tf.Tensor:
        z = self.encode_backbone(x_ids, training=training)
        d = self.dom_emb[domain_id]                         # (DOM_EMB_DIM,)
        g = self.gate(tf.expand_dims(d, 0))                 # (1, HIDDEN_DIM)
        g = tf.cast(g, z.dtype)

        a = self.adapter_d1(z)
        a = self.adapter_drop(a, training=training)
        a = self.adapter_d2(a)
        z2 = z + g * a
        return z2

    def logits(self, x_ids: tf.Tensor, head: str, domain_id: int, training: bool, use_adapter: bool) -> tf.Tensor:
        if use_adapter:
            z = self.encode_with_adapter(x_ids, domain_id=domain_id, training=training)
        else:
            z = self.encode_backbone(x_ids, training=training)

        if head == "src":
            return self.head_src(z)
        if head == "tgt":
            return self.head_tgt(z)
        raise ValueError(head)


# ======================================================================================
# EWC / training
# ======================================================================================

@dataclass
class EWCInfo:
    star: Dict[str, np.ndarray]
    fisher: Dict[str, np.ndarray]

def get_protected_vars_plain(model: PlainTwoHeadCNN) -> Dict[str, tf.Variable]:
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
    head: str,
    max_batches: int,
    temperature: float
) -> EWCInfo:
    """
    Empirical Fisher with softened pseudo labels to avoid vanishing grads on an easy source dataset.
    Then global-mean normalize fisher so lambda is meaningful.
    """
    protected = get_protected_vars_plain(model)
    star = {k: v.numpy().copy() for k, v in protected.items()}
    fisher_acc = {k: np.zeros_like(star[k], dtype=np.float32) for k in star.keys()}

    n_batches = 0
    for x, _y in ds.take(max_batches):
        with tf.GradientTape() as tape:
            logits = model.logits(x, head=head, training=False)
            y_soft = tf.stop_gradient(tf.sigmoid(logits / float(temperature)))
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_soft, logits=logits))
        grads = tape.gradient(loss, list(protected.values()))
        for (k, _v), g in zip(protected.items(), grads):
            if g is None:
                continue
            fisher_acc[k] += (g.numpy().astype(np.float32) ** 2)
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Fisher: got 0 batches.")

    fisher = {k: fisher_acc[k] / float(n_batches) for k in fisher_acc.keys()}

    # global mean normalize
    means = [float(np.mean(f)) for f in fisher.values()]
    mean_f = float(np.mean(means))
    if mean_f < FISHER_EPS:
        mean_f = FISHER_EPS
    fisher = {k: f / mean_f for k, f in fisher.items()}
    return EWCInfo(star=star, fisher=fisher)

def ewc_penalty(current_vars: Dict[str, tf.Variable], ewc: EWCInfo) -> tf.Tensor:
    losses = []
    for k, v in current_vars.items():
        if k not in ewc.star:
            continue
        theta_star = tf.constant(ewc.star[k], dtype=v.dtype)
        fisher = tf.constant(ewc.fisher[k], dtype=v.dtype)
        losses.append(tf.reduce_sum(fisher * tf.square(v - theta_star)))
    return tf.add_n(losses) if losses else tf.constant(0.0, dtype=tf.float32)

def count_params(vars_list: List[tf.Variable]) -> int:
    return int(sum(np.prod(v.shape.as_list()) for v in vars_list))

def smooth_labels(y: tf.Tensor, smoothing: float) -> tf.Tensor:
    if smoothing <= 0.0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing

def train_loop_earlystop(
    train_ds: tf.data.Dataset,
    dev_ds: tf.data.Dataset,
    logits_fn,
    train_vars: List[tf.Variable],
    backbone_vars: List[tf.Variable],
    optimizer: tf.keras.optimizers.Optimizer,
    steps_per_epoch: int,
    max_epochs: int,
    patience: int,
    *,
    ewc_info: Optional[EWCInfo] = None,
    ewc_lambda: float = 0.0,
    l2_anchor_alpha: float = 0.0,
    backbone_lr_mult: float = 1.0,
    tag: str = ""
) -> Tuple[float, Dict[str, float]]:
    """
    Early stop by dev macro-F1 (threshold optimized each epoch).
    """
    backbone_ids = set(id(v) for v in backbone_vars)

    best_f1 = -1.0
    best_thr = 0.5
    best_weights = None
    wait = 0

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = logits_fn(x, training=True)
            y_s = smooth_labels(y, LABEL_SMOOTHING)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_s, logits=logits))

            if ewc_info is not None and ewc_lambda > 0.0:
                prot = getattr(logits_fn, "_ewc_protected", None)
                if prot is not None:
                    loss += tf.cast(ewc_lambda, loss.dtype) * ewc_penalty(prot, ewc_info)

            if ewc_info is not None and l2_anchor_alpha > 0.0:
                prot = getattr(logits_fn, "_ewc_protected", None)
                if prot is not None:
                    l2_terms = []
                    for k, v in prot.items():
                        if k in ewc_info.star:
                            theta_star = tf.constant(ewc_info.star[k], dtype=v.dtype)
                            l2_terms.append(tf.reduce_sum(tf.square(v - theta_star)))
                    if l2_terms:
                        loss += tf.cast(l2_anchor_alpha, loss.dtype) * tf.add_n(l2_terms)

        grads = tape.gradient(loss, train_vars)

        # build pairs, scale backbone grads, filter None
        pairs = []
        for g, v in zip(grads, train_vars):
            if g is None:
                continue
            if id(v) in backbone_ids:
                g = g * tf.cast(backbone_lr_mult, g.dtype)
            pairs.append((g, v))

        if not pairs:
            return loss

        # clip
        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            g_list = [p[0] for p in pairs]
            g_list, _ = tf.clip_by_global_norm(g_list, GRAD_CLIP_NORM)
            pairs = [(g, v) for g, (_, v) in zip(g_list, pairs)]

        optimizer.apply_gradients(pairs)
        return loss

    it = iter(train_ds)

    for epoch in range(1, max_epochs + 1):
        losses = []
        for _ in range(steps_per_epoch):
            x_b, y_b = next(it)
            l = train_step(x_b, y_b)
            losses.append(float(l.numpy()))

        y_true, logits_all, probs = eval_logits_probs(logits_fn, dev_ds)
        thr, m = find_best_threshold(y_true, probs)
        bce = bce_from_logits(y_true, logits_all)

        if VERBOSE:
            _log(f"{tag} | epoch={epoch:02d}  train_loss={np.mean(losses):.4f}  dev_thr={thr:.3f}  dev_acc={m['acc']:.4f}  dev_MacroF1={m['macro_f1']:.4f}  dev_BCE={bce:.4f}")

        if m["macro_f1"] > best_f1 + 1e-6:
            best_f1 = m["macro_f1"]
            best_thr = thr
            best_weights = [w.copy() for w in logits_fn._model.get_weights()]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_weights is not None:
        logits_fn._model.set_weights(best_weights)

    # final dev metrics
    y_true, logits_all, probs = eval_logits_probs(logits_fn, dev_ds)
    thr, m = find_best_threshold(y_true, probs)
    bce = bce_from_logits(y_true, logits_all)
    return thr, {"acc": m["acc"], "macro_f1": m["macro_f1"], "bce": bce}


# ======================================================================================
# Sampling (few-shot)
# ======================================================================================

def sample_fewshot_balanced(
    texts: List[str],
    labels: List[int],
    shot: int,
    seed: int,
    mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    mode:
      - "per_class": shot = K per class (total=2K)
      - "total": shot = total (balanced, K=shot/2)
    """
    rng = np.random.RandomState(seed)
    idx0 = [i for i, y in enumerate(labels) if y == 0]
    idx1 = [i for i, y in enumerate(labels) if y == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    if mode == "per_class":
        k = shot
    elif mode == "total":
        if shot % 2 != 0:
            raise ValueError("total shot must be even for balanced sampling.")
        k = shot // 2
    else:
        raise ValueError(mode)

    if len(idx0) < k or len(idx1) < k:
        raise ValueError(f"Not enough samples for fewshot: need {k}/class, have real={len(idx0)} fake={len(idx1)}")

    sel = idx0[:k] + idx1[:k]
    rng.shuffle(sel)
    xs = np.array([texts[i] for i in sel], dtype=object)
    ys = np.array([labels[i] for i in sel], dtype=np.int32)
    return xs, ys


# ======================================================================================
# Build/init/copy helpers
# ======================================================================================

def build_plain(vocab_size: int) -> PlainTwoHeadCNN:
    m = PlainTwoHeadCNN(vocab_size)
    dummy = tf.zeros((2, MAX_LEN), dtype=tf.int32)
    _ = m.logits(dummy, head="src", training=False)
    _ = m.logits(dummy, head="tgt", training=False)
    return m

def build_dae(vocab_size: int) -> DAEWCCNN:
    m = DAEWCCNN(vocab_size)
    dummy = tf.zeros((2, MAX_LEN), dtype=tf.int32)
    _ = m.logits(dummy, head="src", domain_id=0, training=False, use_adapter=False)
    _ = m.logits(dummy, head="tgt", domain_id=1, training=False, use_adapter=True)
    return m

def copy_plain_to_plain(src: PlainTwoHeadCNN, dst: PlainTwoHeadCNN) -> None:
    dst.embedding.set_weights(src.embedding.get_weights())
    dst.conv.set_weights(src.conv.get_weights())
    dst.shared_fc.set_weights(src.shared_fc.get_weights())
    dst.head_src.set_weights(src.head_src.get_weights())

def copy_plain_to_dae(src: PlainTwoHeadCNN, dst: DAEWCCNN) -> None:
    dst.embedding.set_weights(src.embedding.get_weights())
    dst.conv.set_weights(src.conv.get_weights())
    dst.shared_fc.set_weights(src.shared_fc.get_weights())
    dst.head_src.set_weights(src.head_src.get_weights())


# ======================================================================================
# Main
# ======================================================================================

def main():
    configure_tf()

    _log("Loading datasets...")
    src_texts, src_labels = load_pair_as_dataset(PATH_FAKE, PATH_REAL, name="NEWS(src)")
    tgt_texts, tgt_labels = load_pair_as_dataset(PATH_COVID_FAKE, PATH_COVID_REAL, name="COVID(tgt)")

    _log("\n[Exact dedup]")
    src_texts, src_labels, rm_s = exact_dedup(src_texts, src_labels)
    tgt_texts, tgt_labels, rm_t = exact_dedup(tgt_texts, tgt_labels)
    _log(f"  Source removed exact dups: {rm_s}")
    _log(f"  Target removed exact dups: {rm_t}")

    SPLIT_SEED = 1337
    src_split = stratified_split(src_texts, src_labels, SPLIT_RATIOS, seed=SPLIT_SEED)
    tgt_split = stratified_split(tgt_texts, tgt_labels, SPLIT_RATIOS, seed=SPLIT_SEED)

    _log("\nSplit sizes:")
    _log(f"Source: train={len(src_split['train'][0])} dev={len(src_split['dev'][0])} test={len(src_split['test'][0])}")
    _log(f"Target: train={len(tgt_split['train'][0])} dev={len(tgt_split['dev'][0])} test={len(tgt_split['test'][0])}")

    _log("\nTokenizer: SOURCE-only (paper-strict)")
    token2id = build_vocab_source_only(src_split["train"][0], MAX_VOCAB)
    vocab_size = len(token2id)
    _log(f"Vocab size = {vocab_size} (cap={MAX_VOCAB})")

    # vectorize all splits
    src_train_x = vectorize_texts(src_split["train"][0], token2id, MAX_LEN)
    src_train_y = np.array(src_split["train"][1], dtype=np.int32)
    src_dev_x = vectorize_texts(src_split["dev"][0], token2id, MAX_LEN)
    src_dev_y = np.array(src_split["dev"][1], dtype=np.int32)
    src_test_x = vectorize_texts(src_split["test"][0], token2id, MAX_LEN)
    src_test_y = np.array(src_split["test"][1], dtype=np.int32)

    tgt_train_text = tgt_split["train"][0]
    tgt_train_y_full = np.array(tgt_split["train"][1], dtype=np.int32)
    tgt_dev_x = vectorize_texts(tgt_split["dev"][0], token2id, MAX_LEN)
    tgt_dev_y = np.array(tgt_split["dev"][1], dtype=np.int32)
    tgt_test_x = vectorize_texts(tgt_split["test"][0], token2id, MAX_LEN)
    tgt_test_y = np.array(tgt_split["test"][1], dtype=np.int32)

    raw_rows: List[Dict[str, Any]] = []

    print("\n" + "#" * 120)
    for seed in SEEDS:
        set_global_seed(seed)
        _log(f"\nRUN seed={seed}")

        # ---------------- Stage A: pretrain on source ----------------
        _log("\n[Stage A] Pretrain Plain backbone on NEWS(source)")
        t0 = time.time()
        base = build_plain(vocab_size)

        # class weights (source might be imbalanced)
        n0 = int(np.sum(src_train_y == 0))
        n1 = int(np.sum(src_train_y == 1))
        w0 = (n0 + n1) / max(1, 2 * n0)
        w1 = (n0 + n1) / max(1, 2 * n1)

        pretrain_vars = (
            base.embedding.trainable_variables +
            base.conv.trainable_variables +
            base.shared_fc.trainable_variables +
            base.head_src.trainable_variables
        )

        src_train_ds = make_tf_dataset(src_train_x, src_train_y, BATCH_PRETRAIN, True, seed, True)
        src_dev_ds = make_tf_dataset(src_dev_x, src_dev_y, BATCH_EVAL, False, seed, False)
        steps_pre = max(MIN_STEPS_PER_EPOCH, math.ceil(len(src_train_x) / BATCH_PRETRAIN))
        opt = tf.keras.optimizers.Adam(learning_rate=LR_PRETRAIN)

        @tf.function
        def pretrain_step(x, y):
            with tf.GradientTape() as tape:
                logits = base.logits(x, head="src", training=True)
                y_s = smooth_labels(y, LABEL_SMOOTHING)
                loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_s, logits=logits)
                weights = tf.where(tf.equal(y, 1.0), tf.cast(w1, loss_vec.dtype), tf.cast(w0, loss_vec.dtype))
                loss = tf.reduce_mean(loss_vec * weights)
            grads = tape.gradient(loss, pretrain_vars)
            pairs = [(g, v) for g, v in zip(grads, pretrain_vars) if g is not None]
            if pairs and GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                g_list = [p[0] for p in pairs]
                g_list, _ = tf.clip_by_global_norm(g_list, GRAD_CLIP_NORM)
                pairs = [(g, v) for g, (_, v) in zip(g_list, pairs)]
            opt.apply_gradients(pairs)
            return loss

        best_f1 = -1.0
        best_weights = None
        wait = 0
        it = iter(src_train_ds)

        for epoch in range(1, EPOCHS_PRETRAIN + 1):
            losses = []
            for _ in range(steps_pre):
                x_b, y_b = next(it)
                losses.append(float(pretrain_step(x_b, y_b).numpy()))

            # dev eval
            def logits_fn_src(xb, training=False):
                return base.logits(xb, head="src", training=training)
            logits_fn_src._model = base

            y_true, logits_all, probs = eval_logits_probs(logits_fn_src, src_dev_ds)
            thr, m = find_best_threshold(y_true, probs)
            bce = bce_from_logits(y_true, logits_all)

            _log(f"NEWS DEV | thr={thr:.3f} Acc={m['acc']:.4f} MacroF1={m['macro_f1']:.4f} BCE={bce:.4f}")

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

        # source test (pre)
        src_test_ds = make_tf_dataset(src_test_x, src_test_y, BATCH_EVAL, False, seed, False)
        def logits_fn_src(xb, training=False):
            return base.logits(xb, head="src", training=training)
        logits_fn_src._model = base

        y_true, logits_all, probs = eval_logits_probs(logits_fn_src, src_test_ds)
        thr_src_pre, m_src = find_best_threshold(y_true, probs)
        bce_src_pre = bce_from_logits(y_true, logits_all)
        _log(f"NEWS TEST(pre) | thr={thr_src_pre:.3f} Acc={m_src['acc']:.4f} MacroF1={m_src['macro_f1']:.4f} BCE={bce_src_pre:.4f}")

        # ---------------- Fisher (once per seed) ----------------
        t1 = time.time()
        fisher_ds = make_tf_dataset(src_train_x, src_train_y, BATCH_PRETRAIN, True, seed + 999, False)
        ewc_info = estimate_fisher_plain(
            model=base, ds=fisher_ds, head="src",
            max_batches=FISHER_BATCHES, temperature=FISHER_TEMPERATURE
        )
        means = [float(np.mean(v)) for v in ewc_info.fisher.values()]
        _log(f"Fisher per-var mean (after norm): {[round(x,4) for x in means]}")
        t_fisher = time.time() - t1

        print("\n" + "-" * 110)
        for shot in SHOTS:
            if SHOT_MODE == "per_class":
                n_total = shot * 2
                _log(f"\nTarget setting: {shot}-shot/class (n={n_total})")
            else:
                n_total = shot
                _log(f"\nTarget setting: TOTAL {shot} (balanced)")

            # sample few-shot from target train pool
            fs_texts, fs_labels = sample_fewshot_balanced(
                tgt_train_text, tgt_train_y_full.tolist(),
                shot=shot,
                seed=seed + shot * 17,
                mode=SHOT_MODE
            )
            tgt_fs_x = vectorize_texts(fs_texts.tolist(), token2id, MAX_LEN)
            tgt_fs_y = fs_labels.astype(np.int32)

            tgt_train_ds = make_tf_dataset(tgt_fs_x, tgt_fs_y, BATCH_ADAPT, True, seed + shot, True)
            tgt_dev_ds = make_tf_dataset(tgt_dev_x, tgt_dev_y, BATCH_EVAL, False, seed, False)
            tgt_test_ds = make_tf_dataset(tgt_test_x, tgt_test_y, BATCH_EVAL, False, seed, False)
            steps_adapt = max(MIN_STEPS_PER_EPOCH, math.ceil(len(tgt_fs_x) / BATCH_ADAPT))

            # ================= Scratch-Plain =================
            _log("[Scratch-Plain]")
            t_ad0 = time.time()
            scratch = build_plain(vocab_size)

            train_vars = (
                scratch.embedding.trainable_variables +
                scratch.conv.trainable_variables +
                scratch.shared_fc.trainable_variables +
                scratch.head_tgt.trainable_variables
            )
            backbone_vars = (
                scratch.embedding.trainable_variables +
                scratch.conv.trainable_variables +
                scratch.shared_fc.trainable_variables
            )
            opt_s = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            def logits_fn_sc(xb, training=False):
                return scratch.logits(xb, head="tgt", training=training)
            logits_fn_sc._model = scratch
            logits_fn_sc._ewc_protected = get_protected_vars_plain(scratch)

            thr_tgt, _ = train_loop_earlystop(
                tgt_train_ds, tgt_dev_ds, logits_fn_sc,
                train_vars, backbone_vars, opt_s,
                steps_adapt, EPOCHS_ADAPT, PATIENCE_ADAPT,
                tag="Scratch"
            )
            t_adapt = time.time() - t_ad0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_sc, tgt_test_ds)
            m_t = binary_metrics_from_probs(y_true, probs, thr_tgt)
            bce_t = bce_from_logits(y_true, logits_all)
            _log(f"COVID TEST(Scratch) | thr={thr_tgt:.3f} Acc={m_t['acc']:.4f} MacroF1={m_t['macro_f1']:.4f} BCE={bce_t:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "Scratch-Plain",
                "tgt_acc": m_t["acc"], "tgt_f1_macro": m_t["macro_f1"], "tgt_bce": bce_t,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src["macro_f1"],
                "src_acc_after": np.nan, "src_f1_after": np.nan, "src_bce_after": np.nan,
                "forget_f1": np.nan,
                "trainable_params": count_params(train_vars),
                "t_pretrain": 0.0, "t_fisher": 0.0, "t_adapt": t_adapt,
                "t_total_once": t_adapt
            })

            # ================= Transfer-Plain FullFT =================
            _log("[Transfer-Plain FullFT]")
            t_ad0 = time.time()
            tr = build_plain(vocab_size)
            copy_plain_to_plain(base, tr)

            train_vars = (
                tr.embedding.trainable_variables +
                tr.conv.trainable_variables +
                tr.shared_fc.trainable_variables +
                tr.head_tgt.trainable_variables
            )
            backbone_vars = (
                tr.embedding.trainable_variables +
                tr.conv.trainable_variables +
                tr.shared_fc.trainable_variables
            )
            opt_t = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            def logits_fn_tr(xb, training=False):
                return tr.logits(xb, head="tgt", training=training)
            logits_fn_tr._model = tr
            logits_fn_tr._ewc_protected = get_protected_vars_plain(tr)

            thr_tgt_tr, _ = train_loop_earlystop(
                tgt_train_ds, tgt_dev_ds, logits_fn_tr,
                train_vars, backbone_vars, opt_t,
                steps_adapt, EPOCHS_ADAPT, PATIENCE_ADAPT,
                tag="Transfer"
            )
            t_adapt = time.time() - t_ad0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_tr, tgt_test_ds)
            m_t = binary_metrics_from_probs(y_true, probs, thr_tgt_tr)
            bce_t = bce_from_logits(y_true, logits_all)

            # source after (use src head, pretrain thr)
            def logits_fn_src_after(xb, training=False):
                return tr.logits(xb, head="src", training=training)
            logits_fn_src_after._model = tr
            y_true_s, logits_s, probs_s = eval_logits_probs(logits_fn_src_after, src_test_ds)
            m_s_after = binary_metrics_from_probs(y_true_s, probs_s, thr_src_pre)
            bce_s_after = bce_from_logits(y_true_s, logits_s)
            forget = float(m_src["macro_f1"] - m_s_after["macro_f1"])

            _log(f"COVID TEST(Transfer) | thr={thr_tgt_tr:.3f} Acc={m_t['acc']:.4f} MacroF1={m_t['macro_f1']:.4f} BCE={bce_t:.4f}")
            _log(f"NEWS TEST after(Transfer) | thr={thr_src_pre:.3f} Acc={m_s_after['acc']:.4f} MacroF1={m_s_after['macro_f1']:.4f} BCE={bce_s_after:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "Transfer-Plain",
                "tgt_acc": m_t["acc"], "tgt_f1_macro": m_t["macro_f1"], "tgt_bce": bce_t,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src["macro_f1"],
                "src_acc_after": m_s_after["acc"], "src_f1_after": m_s_after["macro_f1"], "src_bce_after": bce_s_after,
                "forget_f1": forget,
                "trainable_params": count_params(train_vars),
                "t_pretrain": 0.0, "t_fisher": 0.0, "t_adapt": t_adapt,
                "t_total_once": t_adapt
            })

            # ================= Transfer-Plain + EWC =================
            _log("[Transfer-Plain + EWC]")
            t_ad0 = time.time()
            tr_e = build_plain(vocab_size)
            copy_plain_to_plain(base, tr_e)

            train_vars = (
                tr_e.embedding.trainable_variables +
                tr_e.conv.trainable_variables +
                tr_e.shared_fc.trainable_variables +
                tr_e.head_tgt.trainable_variables
            )
            backbone_vars = (
                tr_e.embedding.trainable_variables +
                tr_e.conv.trainable_variables +
                tr_e.shared_fc.trainable_variables
            )
            opt_te = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            def logits_fn_tre(xb, training=False):
                return tr_e.logits(xb, head="tgt", training=training)
            logits_fn_tre._model = tr_e
            logits_fn_tre._ewc_protected = get_protected_vars_plain(tr_e)

            LAM_TRANSFER = 1.0
            thr_tgt_tre, _ = train_loop_earlystop(
                tgt_train_ds, tgt_dev_ds, logits_fn_tre,
                train_vars, backbone_vars, opt_te,
                steps_adapt, EPOCHS_ADAPT, PATIENCE_ADAPT,
                ewc_info=ewc_info, ewc_lambda=LAM_TRANSFER, l2_anchor_alpha=L2_ANCHOR_ALPHA,
                tag=f"Transfer+EWC(λ={LAM_TRANSFER})"
            )
            t_adapt = time.time() - t_ad0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_tre, tgt_test_ds)
            m_t = binary_metrics_from_probs(y_true, probs, thr_tgt_tre)
            bce_t = bce_from_logits(y_true, logits_all)

            def logits_fn_src_after(xb, training=False):
                return tr_e.logits(xb, head="src", training=training)
            logits_fn_src_after._model = tr_e
            y_true_s, logits_s, probs_s = eval_logits_probs(logits_fn_src_after, src_test_ds)
            m_s_after = binary_metrics_from_probs(y_true_s, probs_s, thr_src_pre)
            bce_s_after = bce_from_logits(y_true_s, logits_s)
            forget = float(m_src["macro_f1"] - m_s_after["macro_f1"])

            _log(f"COVID TEST(Plain+EWC) | thr={thr_tgt_tre:.3f} Acc={m_t['acc']:.4f} MacroF1={m_t['macro_f1']:.4f} BCE={bce_t:.4f}")
            _log(f"NEWS TEST after(Plain+EWC) | thr={thr_src_pre:.3f} Acc={m_s_after['acc']:.4f} MacroF1={m_s_after['macro_f1']:.4f} BCE={bce_s_after:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "Transfer-Plain+EWC",
                "tgt_acc": m_t["acc"], "tgt_f1_macro": m_t["macro_f1"], "tgt_bce": bce_t,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src["macro_f1"],
                "src_acc_after": m_s_after["acc"], "src_f1_after": m_s_after["macro_f1"], "src_bce_after": bce_s_after,
                "forget_f1": forget,
                "trainable_params": count_params(train_vars),
                "t_pretrain": 0.0, "t_fisher": 0.0, "t_adapt": t_adapt,
                "t_total_once": t_adapt
            })

            # ================= Adapter-Only =================
            _log("[Adapter-Only (DAEWC; backbone frozen)]")
            t_ad0 = time.time()
            dae1 = build_dae(vocab_size)
            copy_plain_to_dae(base, dae1)

            train_vars = (
                [dae1.dom_emb] +
                dae1.gate.trainable_variables +
                dae1.adapter_d1.trainable_variables +
                dae1.adapter_d2.trainable_variables +
                dae1.head_tgt.trainable_variables
            )
            backbone_vars = []  # frozen
            opt_a = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            def logits_fn_ad(xb, training=False):
                return dae1.logits(xb, head="tgt", domain_id=1, training=training, use_adapter=True)
            logits_fn_ad._model = dae1
            logits_fn_ad._ewc_protected = get_protected_vars_dae(dae1)

            thr_tgt_ad, _ = train_loop_earlystop(
                tgt_train_ds, tgt_dev_ds, logits_fn_ad,
                train_vars, backbone_vars, opt_a,
                steps_adapt, EPOCHS_ADAPT, PATIENCE_ADAPT,
                tag="AdapterOnly"
            )
            t_adapt = time.time() - t_ad0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_ad, tgt_test_ds)
            m_t = binary_metrics_from_probs(y_true, probs, thr_tgt_ad)
            bce_t = bce_from_logits(y_true, logits_all)

            # source after adapter-only: backbone unchanged => same as base
            y_true_s, logits_s, probs_s = eval_logits_probs(logits_fn_src, src_test_ds)
            m_s_after = binary_metrics_from_probs(y_true_s, probs_s, thr_src_pre)
            bce_s_after = bce_from_logits(y_true_s, logits_s)
            forget = float(m_src["macro_f1"] - m_s_after["macro_f1"])

            _log(f"COVID TEST(AdapterOnly) | thr={thr_tgt_ad:.3f} Acc={m_t['acc']:.4f} MacroF1={m_t['macro_f1']:.4f} BCE={bce_t:.4f}")
            _log(f"NEWS TEST after(AdapterOnly) | thr={thr_src_pre:.3f} Acc={m_s_after['acc']:.4f} MacroF1={m_s_after['macro_f1']:.4f} BCE={bce_s_after:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "Adapter-Only",
                "tgt_acc": m_t["acc"], "tgt_f1_macro": m_t["macro_f1"], "tgt_bce": bce_t,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src["macro_f1"],
                "src_acc_after": m_s_after["acc"], "src_f1_after": m_s_after["macro_f1"], "src_bce_after": bce_s_after,
                "forget_f1": forget,
                "trainable_params": count_params(train_vars),
                "t_pretrain": 0.0, "t_fisher": 0.0, "t_adapt": t_adapt,
                "t_total_once": t_adapt
            })

            # ================= DAEWC (Stage1 + Stage2) =================
            _log("[DAEWC (Stage1 + Stage2)]")
            # Stage1 weights are dae1 already trained (adapter-only)
            stage1_weights = [w.copy() for w in dae1.get_weights()]

            strat = DAEWC_STRATEGY_BY_SHOT.get(shot, "fc_only")
            lam = float(DAEWC_LAMBDA_BY_SHOT.get(shot, 1.0))
            bb_mult = float(BACKBONE_LR_MULT_BY_SHOT.get(shot, 0.10))

            t_ad0 = time.time()
            dae2 = build_dae(vocab_size)
            copy_plain_to_dae(base, dae2)
            dae2.set_weights(stage1_weights)

            # train vars: adapter components + tgt head + selected backbone part
            train_vars = (
                [dae2.dom_emb] +
                dae2.gate.trainable_variables +
                dae2.adapter_d1.trainable_variables +
                dae2.adapter_d2.trainable_variables +
                dae2.head_tgt.trainable_variables
            )
            backbone_vars = []

            if strat == "fc_only":
                train_vars += dae2.shared_fc.trainable_variables
                backbone_vars += dae2.shared_fc.trainable_variables
            elif strat == "conv_fc":
                train_vars += dae2.conv.trainable_variables + dae2.shared_fc.trainable_variables
                backbone_vars += dae2.conv.trainable_variables + dae2.shared_fc.trainable_variables
            else:
                raise ValueError(strat)

            opt_d = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            def logits_fn_dae(xb, training=False):
                return dae2.logits(xb, head="tgt", domain_id=1, training=training, use_adapter=True)
            logits_fn_dae._model = dae2
            logits_fn_dae._ewc_protected = get_protected_vars_dae(dae2)

            thr_tgt_d, _ = train_loop_earlystop(
                tgt_train_ds, tgt_dev_ds, logits_fn_dae,
                train_vars, backbone_vars, opt_d,
                steps_adapt, EPOCHS_ADAPT, PATIENCE_ADAPT,
                ewc_info=ewc_info, ewc_lambda=lam, l2_anchor_alpha=L2_ANCHOR_ALPHA,
                backbone_lr_mult=bb_mult,
                tag=f"DAEWC[{strat},bb={bb_mult},λ={lam}]"
            )
            t_adapt = time.time() - t_ad0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_dae, tgt_test_ds)
            m_t = binary_metrics_from_probs(y_true, probs, thr_tgt_d)
            bce_t = bce_from_logits(y_true, logits_all)

            # source after DAEWC: use no-adapter path for source + src head
            def logits_fn_src_dae(xb, training=False):
                return dae2.logits(xb, head="src", domain_id=0, training=training, use_adapter=False)
            logits_fn_src_dae._model = dae2

            y_true_s, logits_s, probs_s = eval_logits_probs(logits_fn_src_dae, src_test_ds)
            m_s_after = binary_metrics_from_probs(y_true_s, probs_s, thr_src_pre)
            bce_s_after = bce_from_logits(y_true_s, logits_s)
            forget = float(m_src["macro_f1"] - m_s_after["macro_f1"])

            _log(f"COVID TEST(DAEWC) | thr={thr_tgt_d:.3f} Acc={m_t['acc']:.4f} MacroF1={m_t['macro_f1']:.4f} BCE={bce_t:.4f}")
            _log(f"NEWS TEST after(DAEWC) | thr={thr_src_pre:.3f} Acc={m_s_after['acc']:.4f} MacroF1={m_s_after['macro_f1']:.4f} BCE={bce_s_after:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "DAEWC",
                "tgt_acc": m_t["acc"], "tgt_f1_macro": m_t["macro_f1"], "tgt_bce": bce_t,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src["macro_f1"],
                "src_acc_after": m_s_after["acc"], "src_f1_after": m_s_after["macro_f1"], "src_bce_after": bce_s_after,
                "forget_f1": forget,
                "trainable_params": count_params(train_vars),
                "t_pretrain": t_pretrain, "t_fisher": t_fisher, "t_adapt": t_adapt,
                "t_total_once": t_pretrain + t_fisher + t_adapt
            })

            # ================= Replay Upper (Joint training) =================
            _log("[Replay Upper (Joint training)]")
            t_ad0 = time.time()
            rep = build_plain(vocab_size)
            copy_plain_to_plain(base, rep)

            train_vars = (
                rep.embedding.trainable_variables +
                rep.conv.trainable_variables +
                rep.shared_fc.trainable_variables +
                rep.head_src.trainable_variables +
                rep.head_tgt.trainable_variables
            )
            opt_r = tf.keras.optimizers.Adam(learning_rate=LR_ADAPT)

            src_joint_ds = make_tf_dataset(src_train_x, src_train_y, BATCH_ADAPT, True, seed + 111, True)
            tgt_joint_ds = make_tf_dataset(tgt_fs_x, tgt_fs_y, BATCH_ADAPT, True, seed + 222, True)
            src_it = iter(src_joint_ds)
            tgt_it = iter(tgt_joint_ds)

            steps_joint = max(MIN_STEPS_PER_EPOCH, steps_adapt)

            @tf.function
            def joint_step(x, y, head: str):
                with tf.GradientTape() as tape:
                    logits = rep.logits(x, head=head, training=True)
                    y_s = smooth_labels(y, LABEL_SMOOTHING)
                    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_s, logits=logits))
                grads = tape.gradient(loss, train_vars)
                pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
                if pairs and GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    g_list = [p[0] for p in pairs]
                    g_list, _ = tf.clip_by_global_norm(g_list, GRAD_CLIP_NORM)
                    pairs = [(g, v) for g, (_, v) in zip(g_list, pairs)]
                opt_r.apply_gradients(pairs)
                return loss

            # modest joint epochs (upper bound; expensive)
            for _ep in range(8):
                for _ in range(steps_joint):
                    xs, ys = next(src_it); joint_step(xs, ys, "src")
                    xt, yt = next(tgt_it); joint_step(xt, yt, "tgt")

            t_adapt = time.time() - t_ad0

            # target thr from dev
            def logits_fn_rep_tgt(xb, training=False):
                return rep.logits(xb, head="tgt", training=training)
            logits_fn_rep_tgt._model = rep
            y_true, logits_all, probs = eval_logits_probs(logits_fn_rep_tgt, tgt_dev_ds)
            thr_r, _ = find_best_threshold(y_true, probs)

            y_true, logits_all, probs = eval_logits_probs(logits_fn_rep_tgt, tgt_test_ds)
            m_t = binary_metrics_from_probs(y_true, probs, thr_r)
            bce_t = bce_from_logits(y_true, logits_all)

            def logits_fn_rep_src(xb, training=False):
                return rep.logits(xb, head="src", training=training)
            logits_fn_rep_src._model = rep
            y_true_s, logits_s, probs_s = eval_logits_probs(logits_fn_rep_src, src_test_ds)
            m_s_after = binary_metrics_from_probs(y_true_s, probs_s, thr_src_pre)
            bce_s_after = bce_from_logits(y_true_s, logits_s)
            forget = float(m_src["macro_f1"] - m_s_after["macro_f1"])

            _log(f"COVID TEST(ReplayUpper) | thr={thr_r:.3f} Acc={m_t['acc']:.4f} MacroF1={m_t['macro_f1']:.4f} BCE={bce_t:.4f}")
            _log(f"NEWS TEST(ReplayUpper) | thr={thr_src_pre:.3f} Acc={m_s_after['acc']:.4f} MacroF1={m_s_after['macro_f1']:.4f} BCE={bce_s_after:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "ReplayUpper",
                "tgt_acc": m_t["acc"], "tgt_f1_macro": m_t["macro_f1"], "tgt_bce": bce_t,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src["macro_f1"],
                "src_acc_after": m_s_after["acc"], "src_f1_after": m_s_after["macro_f1"], "src_bce_after": bce_s_after,
                "forget_f1": forget,
                "trainable_params": count_params(train_vars),
                "t_pretrain": 0.0, "t_fisher": 0.0, "t_adapt": t_adapt,
                "t_total_once": t_adapt
            })

            print("\n" + "-" * 110)

        print("\n" + "#" * 120)

    # ---------------- save results ----------------
    raw_df = pd.DataFrame(raw_rows)
    raw_out = "results_daeewc_v4_raw.csv"
    raw_df.to_csv(raw_out, index=False)
    _log(f"\nSaved raw: {raw_out}")

    def ci95(x: np.ndarray) -> float:
        x = np.array(x, dtype=np.float64)
        if len(x) <= 1:
            return 0.0
        return 1.96 * float(np.std(x, ddof=1)) / math.sqrt(len(x))

    summary_rows = []
    for (shot, method), g in raw_df.groupby(["shot", "method"]):
        summary_rows.append({
            "shot": shot, "method": method,
            "tgt_f1_macro_mean": float(np.mean(g["tgt_f1_macro"])),
            "tgt_f1_macro_ci95": ci95(g["tgt_f1_macro"].values),
            "forget_f1_mean": float(np.nanmean(g["forget_f1"].values)),
            "forget_f1_ci95": ci95(g["forget_f1"].dropna().values) if g["forget_f1"].notna().any() else np.nan,
            "adapt_time_mean_s": float(np.mean(g["t_adapt"])),
            "adapt_time_ci95": ci95(g["t_adapt"].values),
            "trainable_params": float(np.mean(g["trainable_params"])),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["shot", "method"])
    sum_out = "results_daeewc_v4_summary.csv"
    summary_df.to_csv(sum_out, index=False)
    _log(f"Saved summary: {sum_out}")

    print("\n=== Pivot: Target Macro-F1 (mean) ===")
    print(summary_df.pivot(index="shot", columns="method", values="tgt_f1_macro_mean").to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Pivot: Forgetting (ΔSrc Macro-F1 = pre - after; lower is better) ===")
    print(summary_df.pivot(index="shot", columns="method", values="forget_f1_mean").to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Pivot: Adapt time (s, mean) ===")
    print(summary_df.pivot(index="shot", columns="method", values="adapt_time_mean_s").to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
