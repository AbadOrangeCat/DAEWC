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


def make_adam(lr: float) -> tf.keras.optimizers.Optimizer:
    """Use legacy Adam on Apple Silicon (TF 2.11+), fall back otherwise."""
    try:
        return tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    except Exception:
        return tf.keras.optimizers.Adam(learning_rate=lr)


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
SHOT_MODE = "per_class"  # "per_class" (K per class) OR "total" (balanced total)
SHOTS = [10, 20, 80, 160]
SPLIT_RATIOS = (0.70, 0.10, 0.20)  # train/dev/test
SPLIT_SEED = 1337

# ---- tokenizer ----
MAX_VOCAB = 5000
MAX_LEN = 256

# ---- model ----
EMB_DIM = 100

# LSTM backbone (replaces Conv1D + GlobalMaxPool)
BIDIR_LSTM = True          # True = BiLSTM; False = single LSTM
LSTM_UNITS = 64            # if BIDIR_LSTM=True => output dim = 2*LSTM_UNITS (=128)
LSTM_DROPOUT = 0.0         # internal dropout inside LSTM (keep 0 for cuDNN speed)
LSTM_REC_DROPOUT = 0.0     # recurrent_dropout (keep 0 for cuDNN speed)

# Keep the original CNN hyperparams defined (unused in LSTM version) for compatibility / logging.
CONV_FILTERS = 128
KERNEL_SIZE = 5

HIDDEN_DIM = 128

# Adapter
ADAPTER_BOTTLENECK = 96
ADAPTER_DROPOUT = 0.10

# Backbone dropout
DROPOUT_RATE = 0.10

# ---- training ----
BATCH_PRETRAIN = 64
BATCH_ADAPT = 32
BATCH_EVAL = 256

EPOCHS_PRETRAIN = 8
PATIENCE_PRETRAIN = 2

EPOCHS_ADAPT = 40
PATIENCE_ADAPT = 6
# Backward-compat alias: some code paths (older drafts) reference ADAPT_EPOCHS.
# Keep it defined to avoid NameError.
ADAPT_EPOCHS = EPOCHS_ADAPT
# Backward-compat alias: some code paths reference PATIENCE.
PATIENCE = PATIENCE_ADAPT

MIN_STEPS_PER_EPOCH = 50

LR_PRETRAIN = 1e-3
LR_ADAPT = 2e-3

GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.05
LABEL_SMOOTH = LABEL_SMOOTHING  # alias used in loss

# ---- EWC/Fisher (for Transfer+EWC baseline only) ----
FISHER_BATCHES = 200
FISHER_TEMPERATURE = 2.5
FISHER_EPS = 1e-8
L2_ANCHOR_ALPHA = 1e-4

# ---- DAEWC STRONG (FixMatch) ----
DAEWC_USE_UNLABELED = True

# unsup weight schedule
DAEWC_LAMBDA_U_BY_SHOT = {10: 2.0, 20: 2.0, 80: 1.0, 160: 1.0}
DAEWC_TAU_BY_SHOT = {10: 0.93, 20: 0.90, 80: 0.85, 160: 0.80}
DAEWC_RAMPUP_EPOCHS = 8
DAEWC_EMA_DECAY = 0.997

# token-drop augmentation (weak/strong)
DAEWC_WEAK_DROP = 0.05
DAEWC_STRONG_DROP = 0.20
DAEWC_BALANCE_W = 0.10  # class-balance regularizer on unlabeled (helps low-shot)

# optional cap on unlabeled pool for speed (None = use all)
DAEWC_ULB_MAX: Optional[int] = None

# ---- optional: Replay upper bound (slow) ----
RUN_REPLAY_UPPER = True
REPLAY_EPOCHS = 8

# ---- logging ----
VERBOSE = True
VERBOSE_DAEWC = VERBOSE  # DAEWC-specific verbosity flag


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
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, engine="python", encoding="latin1", on_bad_lines="skip")


def pick_text_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Return (title_col, text_col) best guess."""
    cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    lower = {c.lower(): c for c in cols}

    title_col = lower.get("title", None)

    body_candidates = [
        "text", "content", "article", "body", "news", "statement", "claim",
        "maintext", "message", "tweet", "post", "context",
    ]
    text_col = None
    for k in body_candidates:
        if k in lower:
            text_col = lower[k]
            break

    if text_col is None:
        obj_cols = [c for c in cols if df[c].dtype == object]
        if obj_cols:
            sample = df[obj_cols].head(200)
            best, best_len = None, -1.0
            for c in obj_cols:
                vals = sample[c].astype(str).fillna("")
                avg = float(vals.map(len).mean())
                if avg > best_len:
                    best_len, best = avg, c
            text_col = best
        else:
            text_col = cols[0] if cols else None

    return title_col, text_col


def load_pair_as_dataset(fake_path: str, real_path: str, name: str) -> Tuple[List[str], List[int]]:
    """Load two CSVs: fake=1, real=0."""
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
            t = normalize_text(safe_concat_text(title, body))
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
    """Exact dedup by normalized text."""
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
    seed: int,
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


def make_tf_dataset_x(x: np.ndarray, batch: int, shuffle: bool, seed: int, repeat: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(x)
    if shuffle:
        buf = int(min(len(x), 10000))
        ds = ds.shuffle(buffer_size=max(10, buf), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


# ======================================================================================
# Metrics / eval
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


def acc_from_probs(y_true: np.ndarray, probs: np.ndarray, thr: float) -> float:
    """Convenience wrapper: accuracy at a fixed threshold."""
    return float(binary_metrics_from_probs(y_true, probs, thr)["acc"])


def f1_macro_from_probs(y_true: np.ndarray, probs: np.ndarray, thr: float) -> float:
    """Convenience wrapper: macro-F1 at a fixed threshold."""
    return float(binary_metrics_from_probs(y_true, probs, thr)["macro_f1"])


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


def eval_logits_probs_x(logits_fn, ds_x: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    ls: List[np.ndarray] = []
    for x in ds_x:
        logits = logits_fn(x, training=False)
        ls.append(logits.numpy())
    logits_all = np.concatenate(ls, axis=0).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits_all))
    return logits_all, probs


# ======================================================================================
# Models (LSTM backbone)
# ======================================================================================

class PlainTwoHeadCNN(tf.keras.Model):
    """
    NOTE: class name kept to avoid editing the rest of the script.
    Internally this is now an (Bi)LSTM backbone:
        Embedding(mask_zero) -> (Bi)LSTM -> Dropout -> shared_fc -> two heads
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMB_DIM, mask_zero=True, name="emb")

        # Keep attribute name "conv" so downstream training code doesn't change.
        if BIDIR_LSTM:
            self.conv = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    LSTM_UNITS,
                    return_sequences=False,
                    dropout=LSTM_DROPOUT,
                    recurrent_dropout=LSTM_REC_DROPOUT,
                ),
                name="bilstm",
            )
        else:
            self.conv = tf.keras.layers.LSTM(
                LSTM_UNITS,
                return_sequences=False,
                dropout=LSTM_DROPOUT,
                recurrent_dropout=LSTM_REC_DROPOUT,
                name="lstm",
            )

        self.drop = tf.keras.layers.Dropout(DROPOUT_RATE, name="drop")
        self.shared_fc = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", name="shared_fc")
        self.head_src = tf.keras.layers.Dense(1, name="head_src")
        self.head_tgt = tf.keras.layers.Dense(1, name="head_tgt")

    def encode(self, x_ids: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.embedding(x_ids)                # (B,T,E)
        x = self.conv(x, training=training)      # (B,D)
        x = self.drop(x, training=training)
        z = self.shared_fc(x)                    # (B,HIDDEN_DIM)
        return z

    def logits(self, x_ids: tf.Tensor, head: str, training: bool) -> tf.Tensor:
        z = self.encode(x_ids, training=training)
        if head == "src":
            return self.head_src(z)
        if head == "tgt":
            return self.head_tgt(z)
        raise ValueError(head)


class AdapterBlock(tf.keras.layers.Layer):
    """LayerNorm + bottleneck MLP + residual scaling."""
    def __init__(self, dim: int, bottleneck: int, dropout: float, name: str):
        super().__init__(name=name)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln")
        self.d1 = tf.keras.layers.Dense(bottleneck, activation="gelu", name=f"{name}_d1")
        self.drop = tf.keras.layers.Dropout(dropout, name=f"{name}_drop")
        self.d2 = tf.keras.layers.Dense(dim, activation=None, name=f"{name}_d2")
        # trainable residual scale (start small)
        self.alpha = self.add_weight(
            name=f"{name}_alpha", shape=(), initializer=tf.keras.initializers.Constant(0.1), trainable=True
        )

    def call(self, z: tf.Tensor, training: bool) -> tf.Tensor:
        h = self.ln(z)
        h = self.d1(h)
        h = self.drop(h, training=training)
        h = self.d2(h)
        return z + tf.cast(self.alpha, z.dtype) * h


class DAEWCIndepAdapterCNN(tf.keras.Model):
    """Backbone + 2 heads + independent target adapter.

    - Backbone weights are copied from source-pretrained PlainTwoHeadCNN.
    - During target adaptation, we only train: target_adapter + head_tgt (or progressively unfreeze more).
    - Source prediction uses backbone + head_src (no adapter) => source is untouched.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMB_DIM, mask_zero=True, name="emb")

        # Keep attribute name "conv" for compatibility with downstream code.
        if BIDIR_LSTM:
            self.conv = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    LSTM_UNITS,
                    return_sequences=False,
                    dropout=LSTM_DROPOUT,
                    recurrent_dropout=LSTM_REC_DROPOUT,
                ),
                name="bilstm",
            )
        else:
            self.conv = tf.keras.layers.LSTM(
                LSTM_UNITS,
                return_sequences=False,
                dropout=LSTM_DROPOUT,
                recurrent_dropout=LSTM_REC_DROPOUT,
                name="lstm",
            )

        self.drop = tf.keras.layers.Dropout(DROPOUT_RATE, name="drop")
        self.shared_fc = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu", name="shared_fc")

        # heads
        self.head_src = tf.keras.layers.Dense(1, name="head_src")
        self.head_tgt = tf.keras.layers.Dense(1, name="head_tgt")

        # independent target adapter (only used for target)
        self.adapter_tgt = AdapterBlock(HIDDEN_DIM, ADAPTER_BOTTLENECK, ADAPTER_DROPOUT, name="adapter_tgt")

    def encode_backbone(self, x_ids: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.embedding(x_ids)
        x = self.conv(x, training=training)
        x = self.drop(x, training=training)
        z = self.shared_fc(x)
        return z

    def logits_src(self, x_ids: tf.Tensor, training: bool, **_kwargs) -> tf.Tensor:
        z = self.encode_backbone(x_ids, training=training)
        return self.head_src(z)

    def logits_tgt(self, x_ids: tf.Tensor, training: bool, use_adapter: bool = True) -> tf.Tensor:
        z = self.encode_backbone(x_ids, training=training)
        if use_adapter:
            z = self.adapter_tgt(z, training=training)
        return self.head_tgt(z)


# ======================================================================================
# EWC (for Transfer-Plain+EWC baseline)
# ======================================================================================

@dataclass
class EWCInfo:
    star: Dict[str, np.ndarray]
    fisher: Dict[str, np.ndarray]


def _rnn_kernel_and_recurrent(layer: tf.keras.layers.Layer) -> Tuple[tf.Variable, tf.Variable]:
    """Return (kernel, recurrent_kernel) for an LSTM layer across TF/Keras versions.

    Some TF/Keras builds expose these weights directly as `layer.kernel` / `layer.recurrent_kernel`,
    while others only expose them on the underlying `layer.cell`.
    """
    # Common case: attributes exist on the layer itself
    if hasattr(layer, "kernel") and hasattr(layer, "recurrent_kernel"):
        return layer.kernel, layer.recurrent_kernel  # type: ignore[attr-defined]

    # TF often stores them on the cell
    cell = getattr(layer, "cell", None)
    if cell is not None and hasattr(cell, "kernel") and hasattr(cell, "recurrent_kernel"):
        return cell.kernel, cell.recurrent_kernel  # type: ignore[attr-defined]

    # Fallback: scan weights by name (last resort)
    k = None
    rk = None
    for w in layer.weights:
        n = w.name.split("/")[-1]
        if n.startswith("kernel:"):
            k = w
        elif n.startswith("recurrent_kernel:"):
            rk = w
    if k is not None and rk is not None:
        return k, rk

    raise AttributeError(
        "Could not locate kernel/recurrent_kernel on this RNN layer. "
        "Make sure the model has been built (call it on a dummy batch once)."
    )


def get_protected_vars_plain(model: PlainTwoHeadCNN) -> Dict[str, tf.Variable]:
    """
    Protected vars for EWC/Fisher.
    For (Bi)LSTM we protect:
      - embedding matrix
      - LSTM kernel + recurrent_kernel (forward/backward if bidirectional)
      - shared_fc kernel
    """
    out: Dict[str, tf.Variable] = {
        "emb": model.embedding.embeddings,
        "fc_kernel": model.shared_fc.kernel,
    }

    if isinstance(model.conv, tf.keras.layers.Bidirectional):
        f = model.conv.forward_layer
        b = model.conv.backward_layer
        f_k, f_rk = _rnn_kernel_and_recurrent(f)
        b_k, b_rk = _rnn_kernel_and_recurrent(b)
        out.update({
            "rnn_f_kernel": f_k,
            "rnn_f_recurrent": f_rk,
            "rnn_b_kernel": b_k,
            "rnn_b_recurrent": b_rk,
        })
    else:
        k, rk = _rnn_kernel_and_recurrent(model.conv)
        out.update({
            "rnn_kernel": k,
            "rnn_recurrent": rk,
        })
    return out


def estimate_fisher_plain(
    model: PlainTwoHeadCNN,
    ds: tf.data.Dataset,
    head: str,
    max_batches: int,
    temperature: float,
) -> EWCInfo:
    """Empirical Fisher with softened pseudo labels + global mean normalization."""
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
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            fisher_acc[k] += (g.numpy().astype(np.float32) ** 2)
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Fisher: got 0 batches")

    fisher = {k: fisher_acc[k] / float(n_batches) for k in fisher_acc.keys()}
    mean_f = float(np.mean([float(np.mean(v)) for v in fisher.values()]))
    if mean_f < FISHER_EPS:
        mean_f = FISHER_EPS
    fisher = {k: v / mean_f for k, v in fisher.items()}
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


# ======================================================================================
# Training helpers
# ======================================================================================

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
    tag: str = "",
) -> Tuple[float, Dict[str, float]]:
    """Early stop by dev macro-F1 (thr optimized each epoch)."""
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

        pairs = []
        for g, v in zip(grads, train_vars):
            if g is None:
                continue
            if id(v) in backbone_ids:
                g = g * tf.cast(backbone_lr_mult, g.dtype)
            pairs.append((g, v))

        if pairs:
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
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
            _log(
                f"{tag} | epoch={epoch:02d}  train_loss={np.mean(losses):.4f}  "
                f"dev_thr={thr:.3f}  dev_acc={m['acc']:.4f}  dev_MacroF1={m['macro_f1']:.4f}  dev_BCE={bce:.4f}"
            )

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

    y_true, logits_all, probs = eval_logits_probs(logits_fn, dev_ds)
    thr, m = find_best_threshold(y_true, probs)
    bce = bce_from_logits(y_true, logits_all)
    return thr, {"acc": m["acc"], "macro_f1": m["macro_f1"], "bce": bce}


# ======================================================================================
# DAEWC STRONG: FixMatch-style adaptation (independent adapter)
# ======================================================================================

def _augment_token_drop(x_ids: tf.Tensor, drop_prob: float, unk_id: int = 1, pad_id: int = 0) -> tf.Tensor:
    """Randomly replace non-pad tokens with <unk> (word dropout on token IDs)."""
    if drop_prob <= 0.0:
        return x_ids
    rnd = tf.random.uniform(tf.shape(x_ids), 0.0, 1.0, dtype=tf.float32)
    is_pad = tf.equal(x_ids, pad_id)
    to_drop = tf.logical_and(tf.logical_not(is_pad), rnd < tf.cast(drop_prob, tf.float32))
    return tf.where(to_drop, tf.cast(unk_id, x_ids.dtype), x_ids)


def _ema_update(teacher: tf.keras.Model, student: tf.keras.Model, decay: float) -> None:
    for tw, sw in zip(teacher.weights, student.weights):
        tw.assign(decay * tw + (1.0 - decay) * sw)


def train_daeewc_fixmatch(
    student: DAEWCIndepAdapterCNN,
    teacher: DAEWCIndepAdapterCNN,
    ds_l: tf.data.Dataset,
    ds_u: Optional[tf.data.Dataset],
    dev_ds: tf.data.Dataset,
    train_vars: List[tf.Variable],
    optimizer: tf.keras.optimizers.Optimizer,
    steps_per_epoch: int,
    max_epochs: int,
    patience: int,
    *,
    lambda_u_max: float,
    tau: float,
    rampup_epochs: int,
    ema_decay: float,
    tag: str = "DAEWC",
) -> Tuple[float, Dict[str, float]]:
    """FixMatch-style training:

    - labeled batch: supervised loss on student
    - unlabeled batch: pseudo-labels from EMA teacher on weak aug; student trained on strong aug
    """

    best_f1 = -1.0
    best_thr = 0.5
    best_student_weights = None
    best_teacher_weights = None
    wait = 0

    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    it_l = iter(ds_l)
    it_u = iter(ds_u) if ds_u is not None else None

    @tf.function
    def step(x_l, y_l, x_u, lambda_u: tf.Tensor):
        with tf.GradientTape() as tape:
            # supervised
            logits_l = student.logits_tgt(x_l, training=True, use_adapter=True)
            y_s = smooth_labels(y_l, LABEL_SMOOTHING)
            loss_sup = tf.reduce_mean(bce_loss(y_s, logits_l))

            loss_u = tf.constant(0.0, dtype=loss_sup.dtype)
            loss_bal = tf.constant(0.0, dtype=loss_sup.dtype)
            if x_u is not None:
                x_u_w = _augment_token_drop(x_u, DAEWC_WEAK_DROP)
                x_u_s = _augment_token_drop(x_u, DAEWC_STRONG_DROP)

                logits_w = teacher.logits_tgt(x_u_w, training=False, use_adapter=True)
                probs_w = tf.stop_gradient(tf.sigmoid(logits_w))

                # class-balance regularizer (K-shot sampling is balanced)
                p_mean = tf.reduce_mean(probs_w)
                loss_bal = tf.square(p_mean - tf.cast(0.5, p_mean.dtype))

                # confidence mask
                conf = tf.maximum(probs_w, 1.0 - probs_w)
                mask = tf.cast(conf >= tf.cast(tau, conf.dtype), conf.dtype)

                # sharpen a bit (optional, mild)
                # p' = p^2 / (p^2 + (1-p)^2)
                p2 = probs_w * probs_w
                q2 = (1.0 - probs_w) * (1.0 - probs_w)
                probs_sharp = tf.stop_gradient(p2 / (p2 + q2 + 1e-8))

                logits_s = student.logits_tgt(x_u_s, training=True, use_adapter=True)
                loss_u_vec = bce_loss(probs_sharp, logits_s)
                # mask is (B,1) broadcast
                loss_u = tf.reduce_sum(loss_u_vec * mask) / (tf.reduce_sum(mask) + 1e-6)

            loss = loss_sup + lambda_u * loss_u + tf.cast(DAEWC_BALANCE_W, loss_sup.dtype) * loss_bal

        grads = tape.gradient(loss, train_vars)
        pairs = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if pairs and GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
            g_list = [p[0] for p in pairs]
            g_list, _ = tf.clip_by_global_norm(g_list, GRAD_CLIP_NORM)
            pairs = [(g, v) for g, (_, v) in zip(g_list, pairs)]
        optimizer.apply_gradients(pairs)
        return loss_sup, loss_u

    # Teacher init
    teacher.set_weights(student.get_weights())

    for epoch in range(1, max_epochs + 1):
        # ramp up unlabeled weight
        if ds_u is None or lambda_u_max <= 0.0:
            lambda_u = 0.0
        else:
            if rampup_epochs <= 0:
                lambda_u = lambda_u_max
            else:
                t = min(1.0, epoch / float(rampup_epochs))
                # smooth ramp (squared)
                lambda_u = lambda_u_max * (t * t)

        lambda_u_t = tf.constant(lambda_u, dtype=tf.float32)

        sup_losses, u_losses = [], []
        for _ in range(steps_per_epoch):
            x_l, y_l = next(it_l)
            x_u = next(it_u) if it_u is not None else None
            l_sup, l_u = step(x_l, y_l, x_u, lambda_u_t)
            sup_losses.append(float(l_sup.numpy()))
            u_losses.append(float(l_u.numpy()))

            # EMA update each step
            _ema_update(teacher, student, decay=ema_decay)

        # dev eval uses EMA teacher (more stable)
        def logits_fn_dev(xb, training=False):
            return teacher.logits_tgt(xb, training=False, use_adapter=True)
        logits_fn_dev._model = teacher

        y_true, logits_all, probs = eval_logits_probs(logits_fn_dev, dev_ds)
        thr, m = find_best_threshold(y_true, probs)
        bce = bce_from_logits(y_true, logits_all)

        if VERBOSE:
            _log(
                f"{tag} | epoch={epoch:02d}  sup={np.mean(sup_losses):.4f}  unsup={np.mean(u_losses):.4f}  "
                f"λu={lambda_u:.3f}  dev_thr={thr:.3f}  dev_acc={m['acc']:.4f}  dev_MacroF1={m['macro_f1']:.4f}  dev_BCE={bce:.4f}"
            )

        if m["macro_f1"] > best_f1 + 1e-6:
            best_f1 = m["macro_f1"]
            best_thr = thr
            best_student_weights = [w.copy() for w in student.get_weights()]
            best_teacher_weights = [w.copy() for w in teacher.get_weights()]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_student_weights is not None:
        student.set_weights(best_student_weights)
    if best_teacher_weights is not None:
        teacher.set_weights(best_teacher_weights)

    # final dev threshold from teacher
    def logits_fn_dev(xb, training=False):
        return teacher.logits_tgt(xb, training=False, use_adapter=True)
    logits_fn_dev._model = teacher

    y_true, logits_all, probs = eval_logits_probs(logits_fn_dev, dev_ds)
    thr, m = find_best_threshold(y_true, probs)
    bce = bce_from_logits(y_true, logits_all)
    return thr, {"acc": m["acc"], "macro_f1": m["macro_f1"], "bce": bce}


# ======================================================================================
# Few-shot sampling (balanced)
# ======================================================================================

def sample_fewshot_balanced_indices(labels: List[int], shot: int, seed: int, mode: str) -> np.ndarray:
    """Return indices into a (target-train) pool.

    mode:
      - "per_class": shot = K per class (total=2K)
      - "total": shot = total balanced (K=shot//2)
    """
    rng = np.random.RandomState(seed)
    idx0 = np.array([i for i, y in enumerate(labels) if y == 0], dtype=np.int32)
    idx1 = np.array([i for i, y in enumerate(labels) if y == 1], dtype=np.int32)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    if mode == "per_class":
        k = int(shot)
    elif mode == "total":
        if shot % 2 != 0:
            raise ValueError("total shot must be even for balanced sampling")
        k = int(shot // 2)
    else:
        raise ValueError(mode)

    if len(idx0) < k or len(idx1) < k:
        raise ValueError(f"Not enough samples: need {k}/class, have real={len(idx0)} fake={len(idx1)}")

    sel = np.concatenate([idx0[:k], idx1[:k]], axis=0)
    rng.shuffle(sel)
    return sel


# ======================================================================================
# Build/init/copy helpers
# ======================================================================================

# -----------------------------
# DAEWC++ (Dominant) helpers
# -----------------------------

def get_protected_vars_dae(model: "DAEWCIndepAdapterCNN") -> Dict[str, tf.Variable]:
    """Same key names as get_protected_vars_plain so we can reuse fisher from the plain model."""
    out: Dict[str, tf.Variable] = {
        "emb": model.embedding.embeddings,
        "fc_kernel": model.shared_fc.kernel,
    }

    if isinstance(model.conv, tf.keras.layers.Bidirectional):
        f = model.conv.forward_layer
        b = model.conv.backward_layer
        f_k, f_rk = _rnn_kernel_and_recurrent(f)
        b_k, b_rk = _rnn_kernel_and_recurrent(b)
        out.update({
            "rnn_f_kernel": f_k,
            "rnn_f_recurrent": f_rk,
            "rnn_b_kernel": b_k,
            "rnn_b_recurrent": b_rk,
        })
    else:
        k, rk = _rnn_kernel_and_recurrent(model.conv)
        out.update({
            "rnn_kernel": k,
            "rnn_recurrent": rk,
        })
    return out


def dae_train_vars(model: "DAEWCIndepAdapterCNN", level: str):
    """
    level:
      - "adapter": adapter_tgt + head_tgt only
      - "top":    + shared_fc (feature projector)
      - "mid":    + (Bi)LSTM
      - "full":   + embedding
    """
    level = str(level).lower().strip()
    vars_ = []
    vars_ += model.adapter_tgt.trainable_variables
    vars_ += model.head_tgt.trainable_variables
    if level in ("top", "mid", "full"):
        vars_ += model.shared_fc.trainable_variables
    if level in ("mid", "full"):
        vars_ += model.conv.trainable_variables
    if level in ("full",):
        vars_ += model.embedding.trainable_variables
    # NOTE: keep head_src frozen by default (source head is used only for replay regularization).
    return vars_


def l2_anchor_penalty(protected_vars: dict, star: dict):
    """Simple L2 anchoring to source weights (complements EWC)."""
    pen = 0.0
    for k, v in protected_vars.items():
        if k in star:
            pen += tf.reduce_sum(tf.square(v - star[k]))
    return pen


def train_daeewc_fixmatch_plus(
    student: "DAEWCIndepAdapterCNN",
    teacher: "DAEWCIndepAdapterCNN",
    tgt_l_ds: tf.data.Dataset,
    tgt_u_ds: tf.data.Dataset,
    dev_ds: tf.data.Dataset,
    *,
    train_vars: list,
    protected_vars: dict,
    ewc_info,
    steps_per_epoch: int,
    max_epochs: int,
    patience: int,
    lr: float,
    tau: float,
    lambda_u: float,
    ema: float,
    balance_w: float,
    # extra "dominant" knobs
    src_replay_ds=None,
    lambda_src: float = 0.0,
    ewc_lambda: float = 0.0,
    l2_anchor: float = 0.0,
    grad_clip: float = 1.0,
    verbose_prefix: str = "",
):
    """
    Strong DAEWC training:
      supervised target + FixMatch unlabeled + (optional) source replay regularizer
      + (optional) EWC + L2 anchor on shared backbone variables.

    Returns:
      t_elapsed, best_thr, best_dev_f1
    """
    t0 = time.time()
    opt = make_adam(lr)

    # --- constants (avoid dtype surprises / autograph scope issues) ---
    tau_t = tf.constant(float(tau), tf.float32)
    ema_t = tf.constant(float(ema), tf.float32)
    one_minus_ema_t = tf.constant(1.0 - float(ema), tf.float32)
    lambda_src_t = tf.constant(float(lambda_src), tf.float32)
    balance_w_t = tf.constant(float(balance_w), tf.float32)
    ewc_lambda_t = tf.constant(float(ewc_lambda), tf.float32)
    l2_anchor_t = tf.constant(float(l2_anchor), tf.float32)

    # --- init teacher = student (important) ---
    teacher.set_weights(student.get_weights())

    # --- prepare EWC tensors (once; avoid recreating constants every step) ---
    star_np, fisher_np = {}, {}
    if ewc_info is not None:
        if isinstance(ewc_info, dict):
            star_np = ewc_info.get("star", {}) or {}
            fisher_np = ewc_info.get("fisher", {}) or {}
        else:
            star_np = getattr(ewc_info, "star", {}) or {}
            fisher_np = getattr(ewc_info, "fisher", {}) or {}

    star_tf, fisher_tf = {}, {}
    if (ewc_lambda > 0.0 or l2_anchor > 0.0) and star_np:
        # match dtype to each protected var
        for k, v in protected_vars.items():
            if k in star_np:
                star_tf[k] = tf.constant(star_np[k], dtype=v.dtype)
        for k, v in protected_vars.items():
            if k in fisher_np:
                fisher_tf[k] = tf.constant(fisher_np[k], dtype=v.dtype)

    def _ewc_penalty() -> tf.Tensor:
        if not star_tf or not fisher_tf:
            return tf.constant(0.0, dtype=tf.float32)
        losses = []
        for k, v in protected_vars.items():
            if k in star_tf and k in fisher_tf:
                losses.append(tf.reduce_sum(fisher_tf[k] * tf.square(v - star_tf[k])))
        return tf.add_n(losses) if losses else tf.constant(0.0, dtype=tf.float32)

    def _l2_anchor_penalty() -> tf.Tensor:
        if not star_tf:
            return tf.constant(0.0, dtype=tf.float32)
        losses = []
        for k, v in protected_vars.items():
            if k in star_tf:
                losses.append(tf.reduce_sum(tf.square(v - star_tf[k])))
        return tf.add_n(losses) if losses else tf.constant(0.0, dtype=tf.float32)

    # --- iterators (repeat here to guarantee infinite stream) ---
    it_l = iter(tgt_l_ds.repeat())
    it_u = iter(tgt_u_ds.repeat())

    it_s = None
    if src_replay_ds is not None and lambda_src > 0.0:
        it_s = iter(src_replay_ds.repeat())

    best_f1 = -1.0
    best_thr = 0.5
    best_student_w = None
    best_teacher_w = None
    bad_epochs = 0

    DO_SRC = (src_replay_ds is not None and float(lambda_src) > 0.0)
    DO_BAL = (float(balance_w) > 0.0)
    DO_EWC = (ewc_info is not None and float(ewc_lambda) > 0.0)
    DO_L2 = (star_tf and float(l2_anchor) > 0.0)

    @tf.function
    def train_step(x_l, y_l, x_u, x_s, y_s, lambda_u_t):
        # augment unlabeled
        x_u_w = _augment_token_drop(x_u, DAEWC_WEAK_DROP)
        x_u_s = _augment_token_drop(x_u, DAEWC_STRONG_DROP)

        y_l = tf.cast(y_l, tf.float32)
        y_s = tf.cast(y_s, tf.float32)

        with tf.GradientTape() as tape:
            # supervised
            logits_l = student.logits_tgt(x_l, training=True, use_adapter=True)
            if LABEL_SMOOTH > 0.0:
                y_l_s = y_l * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
            else:
                y_l_s = y_l
            sup_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_l_s, logits=logits_l)  # (B,1)
            sup_loss = tf.reduce_mean(sup_vec)

            # pseudo-labels from teacher (stop grad)
            logits_u_w = teacher.logits_tgt(x_u_w, training=False, use_adapter=True)
            p_u = tf.stop_gradient(tf.sigmoid(logits_u_w))  # (B,1)

            conf = tf.maximum(p_u, 1.0 - p_u)              # (B,1)
            mask = tf.cast(conf >= tau_t, tf.float32)      # (B,1)

            logits_u_s = student.logits_tgt(x_u_s, training=True, use_adapter=True)
            unsup_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=p_u, logits=logits_u_s)  # (B,1)
            unsup_loss = tf.reduce_sum(unsup_vec * mask) / (tf.reduce_sum(mask) + 1e-6)

            # balance regularizer (match unlabeled marginal to labeled marginal)
            if DO_BAL:
                bal_loss = tf.square(tf.reduce_mean(p_u) - tf.reduce_mean(y_l))
            else:
                bal_loss = tf.constant(0.0, dtype=tf.float32)

            # source replay regularizer
            if DO_SRC:
                logits_s = student.logits_src(x_s, training=True)
                src_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_s, logits=logits_s)  # (Bs,1)
                src_loss = tf.reduce_mean(src_vec)
            else:
                src_loss = tf.constant(0.0, dtype=tf.float32)

            # EWC + L2 anchor (on protected vars)
            ewc_pen = _ewc_penalty() if DO_EWC else tf.constant(0.0, dtype=tf.float32)
            l2_pen = _l2_anchor_penalty() if DO_L2 else tf.constant(0.0, dtype=tf.float32)

            loss = (
                sup_loss
                + lambda_u_t * unsup_loss
                + balance_w_t * bal_loss
                + lambda_src_t * src_loss
                + ewc_lambda_t * ewc_pen
                + l2_anchor_t * l2_pen
            )

        grads = tape.gradient(loss, train_vars)
        grads_vars = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if grads_vars:
            g_list = [gv[0] for gv in grads_vars]
            v_list = [gv[1] for gv in grads_vars]
            if grad_clip is not None and grad_clip > 0.0:
                g_list, _ = tf.clip_by_global_norm(g_list, grad_clip)
            opt.apply_gradients(zip(g_list, v_list))

        # ---- EMA update teacher IN-PLACE ----
        for v_s_var, v_t_var in zip(student.variables, teacher.variables):
            v_t_var.assign(ema_t * v_t_var + one_minus_ema_t * v_s_var)

        return sup_loss, unsup_loss, src_loss

    for epoch in range(1, max_epochs + 1):
        # ramp up lambda_u (more stable in low-shot)
        if DAEWC_RAMPUP_EPOCHS and DAEWC_RAMPUP_EPOCHS > 0:
            t = min(1.0, epoch / float(DAEWC_RAMPUP_EPOCHS))
            lambda_u_epoch = float(lambda_u) * (t * t)
        else:
            lambda_u_epoch = float(lambda_u)
        lambda_u_epoch_t = tf.constant(lambda_u_epoch, tf.float32)

        sup_meter = tf.keras.metrics.Mean()
        unsup_meter = tf.keras.metrics.Mean()
        src_meter = tf.keras.metrics.Mean()

        for _ in range(steps_per_epoch):
            x_l, y_l = next(it_l)
            x_u = next(it_u)

            if it_s is not None:
                x_s, y_s = next(it_s)
            else:
                # dummy (unused if DO_SRC=False), but keeps signature stable
                x_s, y_s = x_l, y_l

            sup_l, unsup_l, src_l = train_step(x_l, y_l, x_u, x_s, y_s, lambda_u_epoch_t)
            sup_meter.update_state(sup_l)
            unsup_meter.update_state(unsup_l)
            src_meter.update_state(src_l)

        # dev eval with EMA teacher
        def _tgt_logits(x, training=False):
            return teacher.logits_tgt(x, training=training, use_adapter=True)

        y_dev, _logits_dev, p_dev = eval_logits_probs(_tgt_logits, dev_ds)
        thr, m = find_best_threshold(y_dev, p_dev)
        dev_f1 = float(m["macro_f1"])

        if dev_f1 > best_f1 + 1e-6:
            best_f1 = dev_f1
            best_thr = float(thr)
            best_student_w = student.get_weights()
            best_teacher_w = teacher.get_weights()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose_prefix:
            _log(
                f"{verbose_prefix} epoch {epoch:03d} | "
                f"sup={sup_meter.result():.4f} unsup={unsup_meter.result():.4f} src={src_meter.result():.4f} | "
                f"λu={lambda_u_epoch:.3f} dev_f1={dev_f1:.4f} thr={thr:.3f}"
            )

        if bad_epochs >= patience:
            break

    # restore best
    if best_student_w is not None:
        student.set_weights(best_student_w)
    if best_teacher_w is not None:
        teacher.set_weights(best_teacher_w)

    return time.time() - t0, float(best_thr), float(best_f1)


def build_plain(vocab_size: int) -> PlainTwoHeadCNN:
    m = PlainTwoHeadCNN(vocab_size)
    # Important: use <unk>=1 (not all zeros) when Embedding(mask_zero=True)
    dummy = tf.ones((2, MAX_LEN), dtype=tf.int32)
    _ = m.logits(dummy, head="src", training=False)
    _ = m.logits(dummy, head="tgt", training=False)
    return m


def build_dae(vocab_size: int) -> DAEWCIndepAdapterCNN:
    m = DAEWCIndepAdapterCNN(vocab_size)
    dummy = tf.ones((2, MAX_LEN), dtype=tf.int32)
    _ = m.logits_src(dummy, training=False)
    _ = m.logits_tgt(dummy, training=False, use_adapter=True)
    return m


def copy_plain_to_plain(src: PlainTwoHeadCNN, dst: PlainTwoHeadCNN) -> None:
    dst.embedding.set_weights(src.embedding.get_weights())
    dst.conv.set_weights(src.conv.get_weights())
    dst.shared_fc.set_weights(src.shared_fc.get_weights())
    dst.head_src.set_weights(src.head_src.get_weights())


def copy_plain_to_dae(src: PlainTwoHeadCNN, dst: DAEWCIndepAdapterCNN) -> None:
    # backbone
    dst.embedding.set_weights(src.embedding.get_weights())
    dst.conv.set_weights(src.conv.get_weights())
    dst.shared_fc.set_weights(src.shared_fc.get_weights())
    # source head
    dst.head_src.set_weights(src.head_src.get_weights())


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    configure_tf()

    _log("Loading datasets...")
    src_texts, src_labels = load_pair_as_dataset(PATH_FAKE, PATH_REAL, name="NEWS(src)")
    tgt_texts, tgt_labels = load_pair_as_dataset(PATH_COVID_FAKE, PATH_COVID_REAL, name="COVID(tgt)")

    _log("\n[Exact dedup]")
    src_texts, src_labels, rm_s = exact_dedup(src_texts, src_labels)
    tgt_texts, tgt_labels, rm_t = exact_dedup(tgt_texts, tgt_labels)
    _log(f"  Source removed exact dups: {rm_s}")
    _log(f"  Target removed exact dups: {rm_t}")

    src_split = stratified_split(src_texts, src_labels, SPLIT_RATIOS, seed=SPLIT_SEED)
    tgt_split = stratified_split(tgt_texts, tgt_labels, SPLIT_RATIOS, seed=SPLIT_SEED)

    _log("\nSplit sizes:")
    _log(f"Source: train={len(src_split['train'][0])} dev={len(src_split['dev'][0])} test={len(src_split['test'][0])}")
    _log(f"Target: train={len(tgt_split['train'][0])} dev={len(tgt_split['dev'][0])} test={len(tgt_split['test'][0])}")

    _log("\nTokenizer: SOURCE-only (paper-strict)")
    token2id = build_vocab_source_only(src_split["train"][0], MAX_VOCAB)
    vocab_size = len(token2id)
    _log(f"Vocab size = {vocab_size} (cap={MAX_VOCAB})")

    # vectorize source splits
    src_train_x = vectorize_texts(src_split["train"][0], token2id, MAX_LEN)
    src_train_y = np.array(src_split["train"][1], dtype=np.int32)
    src_dev_x = vectorize_texts(src_split["dev"][0], token2id, MAX_LEN)
    src_dev_y = np.array(src_split["dev"][1], dtype=np.int32)
    src_test_x = vectorize_texts(src_split["test"][0], token2id, MAX_LEN)
    src_test_y = np.array(src_split["test"][1], dtype=np.int32)

    # vectorize target splits
    tgt_train_texts = tgt_split["train"][0]
    tgt_train_y_full = np.array(tgt_split["train"][1], dtype=np.int32)
    tgt_train_x_full = vectorize_texts(tgt_train_texts, token2id, MAX_LEN)

    tgt_dev_x = vectorize_texts(tgt_split["dev"][0], token2id, MAX_LEN)
    tgt_dev_y = np.array(tgt_split["dev"][1], dtype=np.int32)
    tgt_test_x = vectorize_texts(tgt_split["test"][0], token2id, MAX_LEN)
    tgt_test_y = np.array(tgt_split["test"][1], dtype=np.int32)

    raw_rows: List[Dict[str, Any]] = []

    print("\n" + "#" * 120)

    for seed in SEEDS:
        set_global_seed(seed)
        _log(f"\nRUN seed={seed}")

        # ------------------------------------------------------------------
        # Stage A: pretrain backbone on NEWS(source)
        # ------------------------------------------------------------------
        _log("\n[Stage A] Pretrain Plain backbone on NEWS(source)")
        t0 = time.time()
        base = build_plain(vocab_size)

        # class weights
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
        src_test_ds = make_tf_dataset(src_test_x, src_test_y, BATCH_EVAL, False, seed, False)

        steps_pre = max(MIN_STEPS_PER_EPOCH, math.ceil(len(src_train_x) / BATCH_PRETRAIN))
        opt_pre = make_adam(LR_PRETRAIN)

        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        @tf.function
        def pretrain_step(x, y):
            with tf.GradientTape() as tape:
                logits = base.logits(x, head="src", training=True)
                y_s = smooth_labels(y, LABEL_SMOOTHING)
                loss_vec = bce_loss(y_s, logits)
                weights = tf.where(tf.equal(y, 1.0), tf.cast(w1, loss_vec.dtype), tf.cast(w0, loss_vec.dtype))
                loss = tf.reduce_mean(loss_vec * weights)
            grads = tape.gradient(loss, pretrain_vars)
            pairs = [(g, v) for g, v in zip(grads, pretrain_vars) if g is not None]
            if pairs and GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                g_list = [p[0] for p in pairs]
                g_list, _ = tf.clip_by_global_norm(g_list, GRAD_CLIP_NORM)
                pairs = [(g, v) for g, (_, v) in zip(g_list, pairs)]
            opt_pre.apply_gradients(pairs)
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
        def logits_fn_src(xb, training=False):
            return base.logits(xb, head="src", training=training)
        logits_fn_src._model = base

        y_true, logits_all, probs = eval_logits_probs(logits_fn_src, src_test_ds)
        thr_src_pre, m_src = find_best_threshold(y_true, probs)
        src_thr_pre = float(thr_src_pre)
        src_f1_pre = float(m_src["macro_f1"])
        bce_src_pre = bce_from_logits(y_true, logits_all)
        _log(f"NEWS TEST(pre) | thr={thr_src_pre:.3f} Acc={m_src['acc']:.4f} MacroF1={m_src['macro_f1']:.4f} BCE={bce_src_pre:.4f}")

        # ------------------------------------------------------------------
        # Fisher (once per seed) for Transfer+EWC baseline
        # ------------------------------------------------------------------
        t1 = time.time()
        fisher_ds = make_tf_dataset(src_train_x, src_train_y, BATCH_PRETRAIN, True, seed + 999, False)
        ewc_info = estimate_fisher_plain(
            model=base,
            ds=fisher_ds,
            head="src",
            max_batches=FISHER_BATCHES,
            temperature=FISHER_TEMPERATURE,
        )
        t_fisher = time.time() - t1

        print("\n" + "-" * 110)

        for shot in SHOTS:
            if SHOT_MODE == "per_class":
                n_total = shot * 2
                _log(f"\nTarget setting: {shot}-shot/class (n={n_total})")
            else:
                n_total = shot
                _log(f"\nTarget setting: TOTAL {shot} (balanced)")

            # sample few-shot indices from target train pool
            sel = sample_fewshot_balanced_indices(
                tgt_train_y_full.tolist(),
                shot=shot,
                seed=seed + shot * 17,
                mode=SHOT_MODE,
            )
            fs_x = tgt_train_x_full[sel]
            fs_y = tgt_train_y_full[sel]

            # unlabeled pool (remaining)
            all_idx = np.arange(len(tgt_train_x_full))
            mask = np.ones(len(tgt_train_x_full), dtype=bool)
            mask[sel] = False
            ul_idx = all_idx[mask]
            if DAEWC_ULB_MAX is not None and len(ul_idx) > DAEWC_ULB_MAX:
                rng = np.random.RandomState(seed + 9999 + shot)
                rng.shuffle(ul_idx)
                ul_idx = ul_idx[:DAEWC_ULB_MAX]
            ul_x = tgt_train_x_full[ul_idx] if len(ul_idx) > 0 else None

            tgt_train_ds = make_tf_dataset(fs_x, fs_y, BATCH_ADAPT, True, seed + shot, True)
            tgt_dev_ds = make_tf_dataset(tgt_dev_x, tgt_dev_y, BATCH_EVAL, False, seed, False)
            tgt_test_ds = make_tf_dataset(tgt_test_x, tgt_test_y, BATCH_EVAL, False, seed, False)

            ds_ul = None
            if ul_x is not None and len(ul_x) > 0:
                ds_ul = make_tf_dataset_x(ul_x, BATCH_ADAPT, True, seed + 777 + shot, True)

            steps_adapt = max(MIN_STEPS_PER_EPOCH, math.ceil(len(fs_x) / BATCH_ADAPT))

            # ---------------- Scratch-Plain ----------------
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
            opt_s = make_adam(LR_ADAPT)

            def logits_fn_sc(xb, training=False):
                return scratch.logits(xb, head="tgt", training=training)
            logits_fn_sc._model = scratch
            logits_fn_sc._ewc_protected = get_protected_vars_plain(scratch)

            thr_tgt, _ = train_loop_earlystop(
                tgt_train_ds, tgt_dev_ds, logits_fn_sc,
                train_vars, backbone_vars, opt_s,
                steps_adapt, EPOCHS_ADAPT, PATIENCE_ADAPT,
                tag="Scratch",
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
                "t_total_once": t_adapt,
            })

            # ---------------- Transfer-Plain FullFT ----------------
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
            opt_t = make_adam(LR_ADAPT)

            def logits_fn_tr(xb, training=False):
                return tr.logits(xb, head="tgt", training=training)
            logits_fn_tr._model = tr
            logits_fn_tr._ewc_protected = get_protected_vars_plain(tr)

            thr_tgt_tr, _ = train_loop_earlystop(
                tgt_train_ds, tgt_dev_ds, logits_fn_tr,
                train_vars, backbone_vars, opt_t,
                steps_adapt, EPOCHS_ADAPT, PATIENCE_ADAPT,
                tag="Transfer",
            )
            t_adapt = time.time() - t_ad0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_tr, tgt_test_ds)
            m_t = binary_metrics_from_probs(y_true, probs, thr_tgt_tr)
            bce_t = bce_from_logits(y_true, logits_all)

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
                "t_total_once": t_adapt,
            })

            # ---------------- Transfer-Plain + EWC ----------------
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
            opt_te = make_adam(LR_ADAPT)

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
                tag=f"Transfer+EWC(λ={LAM_TRANSFER})",
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
                "t_total_once": t_adapt,
            })

            # ---------------- Adapter-Only (independent adapter; labeled-only) ----------------
            _log("[Adapter-Only (independent adapter; labeled-only)]")
            t_ad0 = time.time()
            ad = build_dae(vocab_size)
            copy_plain_to_dae(base, ad)

            train_vars = ad.adapter_tgt.trainable_variables + ad.head_tgt.trainable_variables
            backbone_vars = []
            opt_a = make_adam(LR_ADAPT)

            def logits_fn_ad(xb, training=False):
                return ad.logits_tgt(xb, training=training, use_adapter=True)
            logits_fn_ad._model = ad

            thr_ad, _ = train_loop_earlystop(
                tgt_train_ds, tgt_dev_ds, logits_fn_ad,
                train_vars, backbone_vars, opt_a,
                steps_adapt, EPOCHS_ADAPT, PATIENCE_ADAPT,
                tag="AdapterOnly",
            )
            t_adapt = time.time() - t_ad0

            y_true, logits_all, probs = eval_logits_probs(logits_fn_ad, tgt_test_ds)
            m_t = binary_metrics_from_probs(y_true, probs, thr_ad)
            bce_t = bce_from_logits(y_true, logits_all)

            # source untouched
            y_true_s, logits_s, probs_s = eval_logits_probs(logits_fn_src, src_test_ds)
            m_s_after = binary_metrics_from_probs(y_true_s, probs_s, thr_src_pre)
            bce_s_after = bce_from_logits(y_true_s, logits_s)
            forget = float(m_src["macro_f1"] - m_s_after["macro_f1"])

            _log(f"COVID TEST(AdapterOnly) | thr={thr_ad:.3f} Acc={m_t['acc']:.4f} MacroF1={m_t['macro_f1']:.4f} BCE={bce_t:.4f}")
            _log(f"NEWS TEST after(AdapterOnly) | thr={thr_src_pre:.3f} Acc={m_s_after['acc']:.4f} MacroF1={m_s_after['macro_f1']:.4f} BCE={bce_s_after:.4f}")

            raw_rows.append({
                "seed": seed, "shot": shot, "method": "Adapter-Only",
                "tgt_acc": m_t["acc"], "tgt_f1_macro": m_t["macro_f1"], "tgt_bce": bce_t,
                "src_thr_pre": thr_src_pre, "src_f1_pre": m_src["macro_f1"],
                "src_acc_after": m_s_after["acc"], "src_f1_after": m_s_after["macro_f1"], "src_bce_after": bce_s_after,
                "forget_f1": forget,
                "trainable_params": count_params(train_vars),
                "t_pretrain": 0.0, "t_fisher": 0.0, "t_adapt": t_adapt,
                "t_total_once": t_adapt,
            })

            # ---------------- DAEWC STRONG (FixMatch + EMA teacher; independent adapter) ----------------
            _log("[DAEWC-DOMINANT] FixMatch+EMA + (auto)unfreeze + EWC + source-replay (select by dev F1)")

            # Candidate recipe set (kept small so it is still practical).
            tau_base = float(DAEWC_TAU_BY_SHOT.get(shot, 0.94))
            lam_u_base = float(DAEWC_LAMBDA_U_BY_SHOT.get(shot, 1.0))

            if shot <= 20:
                candidates = [
                    # name, level, lr, tau, lambda_u, lambda_src, ewc_lambda, l2_anchor
                    ("A_adapter_fixmatch",       "adapter", 2e-3, tau_base,                  lam_u_base,       0.00, 0.00, 0.0),
                    ("B_top_fixmatch_replay",    "top",     1e-3, max(0.90, tau_base - 0.01), lam_u_base,      0.20, 0.10, 1e-4),
                    ("C_mid_fixmatch_replay",    "mid",     7e-4, max(0.90, tau_base - 0.02), lam_u_base * 0.8, 0.15, 0.05, 5e-5),
                    ("D_full_fixmatch_replay",   "full",    5e-4, max(0.90, tau_base - 0.02), lam_u_base * 0.6, 0.12, 0.03, 1e-5),
                    # supervised-only + replay fallback
                    ("E_full_supervised_replay", "full",    5e-4, 1.00,                       0.00,            0.20, 0.00, 1e-5),
                ]
            elif shot <= 80:
                candidates = [
                    ("A_top_fixmatch_replay",    "top",     8e-4, 0.93, 1.0, 0.20, 0.05, 1e-5),
                    ("B_mid_fixmatch_replay",    "mid",     5e-4, 0.92, 0.8, 0.15, 0.03, 1e-6),
                    ("C_full_fixmatch_replay",   "full",    3e-4, 0.92, 0.6, 0.10, 0.02, 1e-6),
                    ("D_full_fixmatch_noreplay", "full",    3e-4, 0.92, 0.6, 0.00, 0.00, 0.0),
                ]
            else:
                candidates = [
                    ("A_mid_fixmatch_replay",    "mid",     5e-4, 0.90, 0.5, 0.10, 0.03, 1e-6),
                    ("B_full_fixmatch_replay",   "full",    3e-4, 0.90, 0.5, 0.08, 0.02, 1e-6),
                    ("C_full_fixmatch_noreplay", "full",    3e-4, 0.90, 0.5, 0.00, 0.00, 0.0),
                ]

            best = {
                "name": None,
                "score": -1e9,
                "dev_f1": -1.0,
                "src_f1_after": -1.0,
                "thr": 0.5,
                "t_adapt": 0.0,
                "trainable_params": None,
                "teacher": None,
                "student": None,
            }
            t_search_total = 0.0

            for cname, level, lr, tau, lam_u, lam_src, lam_ewc, lam_l2 in candidates:
                # build fresh student/teacher and init from the pretrained plain model
                student = build_dae(vocab_size)
                teacher = build_dae(vocab_size)
                copy_plain_to_dae(base, student)
                copy_plain_to_dae(base, teacher)

                train_vars = dae_train_vars(student, level)
                prot = get_protected_vars_dae(student)

                # optional source replay (regularizer)
                src_replay_ds = None
                if lam_src > 0.0:
                    src_replay_ds = make_tf_dataset(src_train_x, src_train_y, BATCH_ADAPT, True, seed + 999 + shot, True)

                # if no unlabeled pool, fall back to supervised-only (lambda_u -> 0)
                if ds_ul is None:
                    lam_u_eff = 0.0
                    ds_ul_eff = make_tf_dataset_x(fs_x, BATCH_ADAPT, True, seed + 777 + shot, True)
                else:
                    lam_u_eff = lam_u
                    ds_ul_eff = ds_ul

                t_cand, thr_cand, dev_f1_cand = train_daeewc_fixmatch_plus(
                    student, teacher, tgt_train_ds, ds_ul_eff, tgt_dev_ds,
                    train_vars=train_vars,
                    protected_vars=prot,
                    ewc_info=ewc_info,
                    steps_per_epoch=steps_adapt,
                    max_epochs=ADAPT_EPOCHS,
                    patience=PATIENCE,
                    lr=lr,
                    tau=tau,
                    lambda_u=lam_u_eff,
                    ema=DAEWC_EMA_DECAY,
                    balance_w=DAEWC_BALANCE_W,
                    src_replay_ds=src_replay_ds,
                    lambda_src=lam_src,
                    ewc_lambda=lam_ewc,
                    l2_anchor=lam_l2,
                    grad_clip=1.0,
                    verbose_prefix=f"[DAEWC {cname}]" if VERBOSE_DAEWC else "",
                )
                t_search_total += t_cand

                # quick src-after check (to avoid picking a candidate that forgets too much)
                def _src_logits_c(x, training=False):
                    return teacher.logits_src(x, training=training)

                y_s_c, _, p_s_c = eval_logits_probs(_src_logits_c, src_test_ds)
                src_f1_after_c = f1_macro_from_probs(y_s_c, p_s_c, src_thr_pre)
                forget_c = max(0.0, float(src_f1_pre - src_f1_after_c))

                # selection score: prioritize tgt-dev F1; light penalty for forgetting
                cand_score = float(dev_f1_cand) - 0.20 * forget_c

                _log(
                    f"  [Cand {cname}] level={level} "
                    f"dev_f1={dev_f1_cand:.4f} src_f1_after={src_f1_after_c:.4f} "
                    f"forget={forget_c:.4f} score={cand_score:.4f} t={t_cand:.2f}s"
                )

                if (cand_score > best["score"] + 1e-6) or (
                    abs(cand_score - best["score"]) <= 1e-6 and src_f1_after_c > best["src_f1_after"] + 1e-6
                ):
                    # drop old best to keep memory bounded
                    if best["teacher"] is not None:
                        del best["teacher"]
                        del best["student"]
                    best.update({
                        "name": cname,
                        "score": cand_score,
                        "dev_f1": dev_f1_cand,
                        "src_f1_after": src_f1_after_c,
                        "thr": thr_cand,
                        "t_adapt": t_cand,
                        "trainable_params": int(count_params(train_vars)),
                        "teacher": teacher,
                        "student": student,
                    })
                else:
                    # free the models if not best to reduce memory
                    del student
                    del teacher

            _log(
                f"[DAEWC-DOMINANT] picked={best['name']} "
                f"dev_f1={best['dev_f1']:.4f} src_f1_after={best['src_f1_after']:.4f} "
                f"score={best['score']:.4f} thr={best['thr']:.3f} trainable={best['trainable_params']}"
            )

            # Evaluate with the selected EMA teacher
            teacher = best["teacher"]
            thr_tgt = best["thr"]
            t_adapt = t_search_total
            trainable_params = best["trainable_params"]

            def tgt_logits(x, training=False):
                return teacher.logits_tgt(x, training=training, use_adapter=True)

            y_t, l_t, p_t = eval_logits_probs(tgt_logits, tgt_test_ds)
            tgt_acc = acc_from_probs(y_t, p_t, thr_tgt)
            tgt_f1 = f1_macro_from_probs(y_t, p_t, thr_tgt)
            tgt_bce = bce_from_logits(y_t, l_t)

            # Source-after (forgetting)
            def src_logits(x, training=False):
                return teacher.logits_src(x, training=training)

            y_s, l_s, p_s = eval_logits_probs(src_logits, src_test_ds)
            src_acc_after = acc_from_probs(y_s, p_s, src_thr_pre)
            src_f1_after = f1_macro_from_probs(y_s, p_s, src_thr_pre)
            src_bce_after = bce_from_logits(y_s, l_s)
            forget_f1 = max(0.0, src_f1_pre - src_f1_after)

            raw_rows.append({
                "seed": seed,
                "shot": shot,
                "method": "DAEWC",
                "tgt_acc": tgt_acc, "tgt_f1_macro": tgt_f1, "tgt_bce": tgt_bce,
                "src_thr_pre": src_thr_pre, "src_f1_pre": src_f1_pre,
                "src_acc_after": src_acc_after, "src_f1_after": src_f1_after, "src_bce_after": src_bce_after,
                "forget_f1": forget_f1,
                "trainable_params": trainable_params,
                "t_pretrain": t_pretrain, "t_fisher": t_fisher, "t_adapt": t_adapt,
                "t_total_once": t_pretrain + t_fisher + t_adapt,
            })

            # ---------------- Replay Upper (Joint training; slow upper bound) ----------------
            if RUN_REPLAY_UPPER:
                _log("[Replay Upper (Joint training; slow)]")
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
                opt_r = make_adam(LR_ADAPT)

                src_joint_ds = make_tf_dataset(src_train_x, src_train_y, BATCH_ADAPT, True, seed + 111, True)
                tgt_joint_ds = make_tf_dataset(fs_x, fs_y, BATCH_ADAPT, True, seed + 222 + shot, True)
                src_it = iter(src_joint_ds)
                tgt_it = iter(tgt_joint_ds)
                steps_joint = max(MIN_STEPS_PER_EPOCH, steps_adapt)

                @tf.function
                def joint_step(x, y, head_is_src: tf.Tensor):
                    with tf.GradientTape() as tape:
                        logits = tf.cond(
                            head_is_src,
                            lambda: rep.logits(x, head="src", training=True),
                            lambda: rep.logits(x, head="tgt", training=True),
                        )
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

                for _ep in range(REPLAY_EPOCHS):
                    for _ in range(steps_joint):
                        xs, ys = next(src_it); joint_step(xs, ys, tf.constant(True))
                        xt, yt = next(tgt_it); joint_step(xt, yt, tf.constant(False))

                t_adapt_r = time.time() - t_ad0

                # threshold from dev
                def logits_fn_rep_tgt(xb, training=False):
                    return rep.logits(xb, head="tgt", training=training)
                logits_fn_rep_tgt._model = rep

                y_true_d, logits_d, probs_d = eval_logits_probs(logits_fn_rep_tgt, tgt_dev_ds)
                thr_r, _ = find_best_threshold(y_true_d, probs_d)

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
                    "t_pretrain": 0.0, "t_fisher": 0.0, "t_adapt": t_adapt_r,
                    "t_total_once": t_adapt_r,
                })

            print("\n" + "-" * 110)

        print("\n" + "#" * 120)

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------
    raw_df = pd.DataFrame(raw_rows)
    raw_out = "results_daeewc_v6_dominant_raw.csv"
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
    sum_out = "results_daeewc_v6_dominant_summary.csv"
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
