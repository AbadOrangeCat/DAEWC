import os
import random
import time
import gc
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers as L
from tensorflow.keras import Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


# =========================================================
# 0. GPU + Reproducibility
# =========================================================
SEED = 42

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("GPU memory growth set failed:", e)

def reset_seeds(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def clear_tf():
    tf.keras.backend.clear_session()
    gc.collect()

setup_gpu()
reset_seeds(SEED)


# =========================================================
# 1. Config
# =========================================================
MAX_NUM_WORDS = 5000
MAXLEN = 1000
EMBED_DIM = 100
EPOCHS = 5
BATCH_SIZE = 64

# Adapter / Gate
ADAPTER_R = 16
GATE_HIDDEN = 64

# Continual learning setup (2 tasks): political -> medical
fractions = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]

# -------- EWC key knobs (重要) --------
USE_EWC = True

# 让 Fisher 更“有信息”，避免过拟合后梯度≈0
FISHER_SAMPLES = 2000
FISHER_BATCH_SIZE = 64
FISHER_LABEL_SMOOTH = 0.10     # ✅ 关键：让 fisher 不至于近0
FISHER_TRAINING_TRUE = True    # ✅ 关键：Dropout 打开

# EWC base strength（会自动按 fisher scale 做校正）
EWC_LAMBDA_BASE = 50.0
AUTO_SCALE_EWC_LAMBDA = True   # ✅ 关键：Adapter/Plain fisher量级差异会被补偿
EWC_EPS = 1e-12

# Optim & LR（OURS 通常用更小 lr 更稳）
LR_PLAIN_FT = 1e-3
LR_ADAPTER_PEFT = 1e-3
LR_OURS_TRUNK = 5e-4

CLIPNORM = 1.0                 # 防止大 lambda 导致不稳定

# Threshold
CALIBRATE_THRESHOLD = False     # True会“用val选thr”，偏乐观；默认False

# Baselines toggles
INCLUDE_MED_SCRATCH = True
INCLUDE_JOINT_REPLAY_UPPER = True   # joint = upper bound（不属于纯持续学习）

# Paths
PATH_FAKE = './news/Fake.csv'
PATH_REAL = './news/True.csv'
PATH_COVID_FAKE = './covid/fakeNews.csv'
PATH_COVID_REAL = './covid/trueNews.csv'


# =========================================================
# 2. Utilities
# =========================================================
def build_tokenizer(train_texts: List[str], num_words: int = MAX_NUM_WORDS) -> Tuple[Tokenizer, int]:
    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tok.fit_on_texts(train_texts)
    vocab_size = min(num_words, len(tok.word_index) + 1)
    return tok, vocab_size

def vectorize(tok: Tokenizer, texts: List[str], maxlen: int = MAXLEN) -> np.ndarray:
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen)

def domain_ids_like(n: int, domain: int) -> np.ndarray:
    return np.full((n,), domain, dtype=np.int32)

def stratified_subset(texts: List[str], labels: np.ndarray, frac: float, seed: int = SEED) -> Tuple[List[str], np.ndarray]:
    if frac >= 1.0:
        return list(texts), np.array(labels)
    if frac <= 0.0:
        raise ValueError("frac must be > 0.0")
    try:
        X_sub, _, y_sub, _ = train_test_split(
            texts, labels, train_size=frac, random_state=seed, stratify=labels
        )
        return list(X_sub), np.array(y_sub)
    except ValueError:
        texts_np = np.array(texts)
        idxs = np.arange(len(labels))
        rng = np.random.RandomState(seed)
        chosen = []
        for cls in np.unique(labels):
            cls_idxs = idxs[labels == cls]
            n = max(1, int(np.floor(frac * len(cls_idxs))))
            n = min(n, len(cls_idxs))
            chosen.extend(rng.choice(cls_idxs, size=n, replace=False))
        chosen = np.array(chosen)
        return texts_np[chosen].tolist(), labels[chosen]

def count_trainable_params(model: tf.keras.Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))

def bce_value(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.float32).reshape(-1, 1)
    y_prob = np.asarray(y_prob).astype(np.float32).reshape(-1, 1)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return float(loss_fn(y_true, y_prob).numpy())

def evaluate_probs(y_prob: np.ndarray, y_true: np.ndarray, name: str = "", calibrate: bool = CALIBRATE_THRESHOLD) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).ravel()

    if calibrate:
        P, R, T = precision_recall_curve(y_true, y_prob)
        F1 = 2 * P * R / (P + R + 1e-9)
        best_idx = int(np.argmax(F1[:-1])) if len(T) > 0 else 0
        thr = float(T[best_idx]) if len(T) > 0 else 0.5
    else:
        thr = 0.5

    y_pred = (y_prob > thr).astype("int32")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    loss = bce_value(y_true, y_prob)

    print(f"{name} | thr={thr:.3f}  Acc={acc:.4f}  F1={f1:.4f}  BCE={loss:.4f}")
    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1), "thr": float(thr), "bce": float(loss)}

def timed_fit(model: tf.keras.Model,
              train_x, train_y,
              val_x, val_y,
              epochs: int = EPOCHS,
              batch_size: int = BATCH_SIZE,
              verbose: int = 0) -> float:
    t0 = time.time()
    model.fit(train_x, train_y,
              epochs=epochs, batch_size=batch_size,
              validation_data=(val_x, val_y),
              verbose=verbose)
    return time.time() - t0

def make_optimizer(lr: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=CLIPNORM)


# =========================================================
# 3. Models
# =========================================================
def build_plain_cnn(vocab_size: int,
                    maxlen: int = MAXLEN,
                    embed_dim: int = EMBED_DIM,
                    conv_filters: int = 128,
                    kernel_size: int = 5,
                    dense_units: int = 10,
                    dropout: float = 0.5) -> Model:
    tok_in = L.Input(shape=(maxlen,), dtype="int32", name="tokens")
    x = L.Embedding(vocab_size, embed_dim, input_length=maxlen, name="emb")(tok_in)
    h = L.Conv1D(conv_filters, kernel_size, activation="relu", name="conv")(x)
    pooled = L.GlobalMaxPooling1D(name="gmp")(h)
    z = L.Dense(dense_units, activation="relu", name="fc")(pooled)
    z = L.Dropout(dropout, name="drop")(z)
    out = L.Dense(1, activation="sigmoid", name="out")(z)
    model = Model(tok_in, out, name="PlainCNN")
    model.compile(optimizer=make_optimizer(LR_PLAIN_FT), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_domain_adapter_cnn(vocab_size: int,
                             maxlen: int = MAXLEN,
                             embed_dim: int = EMBED_DIM,
                             conv_filters: int = 128,
                             kernel_size: int = 5,
                             dense_units: int = 10,
                             dropout: float = 0.5,
                             adapter_r: int = ADAPTER_R,
                             gate_hidden: int = GATE_HIDDEN) -> Model:
    tok_in = L.Input(shape=(maxlen,), dtype="int32", name="tokens")
    dom_in = L.Input(shape=(), dtype="int32", name="domain_id")  # 0=political, 1=medical

    x = L.Embedding(vocab_size, embed_dim, input_length=maxlen, name="emb")(tok_in)
    h = L.Conv1D(conv_filters, kernel_size, activation="relu", name="conv")(x)

    # Houlsby adapter + residual
    a = L.TimeDistributed(L.Dense(adapter_r, activation="relu"), name="adapter_down")(h)
    a = L.TimeDistributed(L.Dense(conv_filters), name="adapter_up")(a)
    h = L.Add(name="adapter_residual")([h, a])

    # Domain-conditioned gate (SE-like)
    d = L.Embedding(input_dim=2, output_dim=gate_hidden, name="dom_emb")(dom_in)
    d = L.Dense(gate_hidden, activation="relu", name="dom_proj1")(d)
    d = L.Dense(conv_filters, activation=None, name="dom_proj2")(d)
    d = L.Reshape((1, conv_filters), name="dom_reshape")(d)

    g_stat = L.GlobalAveragePooling1D(name="gap")(h)
    g_stat = L.Dense(conv_filters, activation=None, name="stat_proj")(g_stat)
    g_stat = L.Reshape((1, conv_filters), name="stat_reshape")(g_stat)

    gate_preact = L.Add(name="gate_preact")([d, g_stat])
    gate = L.Activation("sigmoid", name="gate_sigmoid")(gate_preact)
    h = L.Multiply(name="gated")([h, gate])

    pooled = L.GlobalMaxPooling1D(name="gmp")(h)
    z = L.Dense(dense_units, activation="relu", name="fc")(pooled)
    z = L.Dropout(dropout, name="drop")(z)
    out = L.Dense(1, activation="sigmoid", name="out")(z)

    model = Model(inputs={"tokens": tok_in, "domain_id": dom_in}, outputs=out, name="DomainAdapterCNN")
    model.compile(optimizer=make_optimizer(LR_OURS_TRUNK), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# =========================================================
# 4. Trainable strategies (关键：突出持续学习 + 时间效率)
# =========================================================
def set_trainable_plain_full(model: Model):
    for layer in model.layers:
        layer.trainable = True

def set_trainable_adapter_peft(model: Model):
    """
    Adapter-PEFT baseline:
    freeze Emb + Conv (shared trunk) -> train adapter/gate/head
    目的：参数少、快、遗忘小
    """
    train_prefix = ("adapter_", "dom_", "gate_", "stat_", "fc", "out", "drop")
    for layer in model.layers:
        if layer.name in ("tokens", "domain_id"):
            layer.trainable = False
        elif layer.name.startswith(train_prefix):
            layer.trainable = True
        elif layer.name in ("emb", "conv"):
            layer.trainable = False
        else:
            layer.trainable = False

def set_trainable_adapter_ours(model: Model):
    """
    OURS（更强适应）：
    freeze Emb（稳定词表） + unfreeze Conv + adapter/gate/head
    然后用 EWC 约束 shared trunk（conv/fc/out 等）
    """
    for layer in model.layers:
        if layer.name in ("tokens", "domain_id"):
            layer.trainable = False
        elif layer.name == "emb":
            layer.trainable = False
        else:
            layer.trainable = True


# =========================================================
# 5. EWC
# =========================================================
class EWCModel(tf.keras.Model):
    def __init__(self,
                 base_model: tf.keras.Model,
                 fisher: Dict[str, tf.Tensor],
                 theta_old: Dict[str, tf.Tensor],
                 ewc_lambda: float):
        super().__init__()
        self.base = base_model
        self.fisher = fisher
        self.theta_old = theta_old
        self.ewc_lambda = float(ewc_lambda)

    def call(self, inputs, training=False):
        return self.base(inputs, training=training)

    def train_step(self, data):
        x, y = data
        y = tf.cast(y, tf.float32)
        y = tf.reshape(y, (-1, 1))

        with tf.GradientTape() as tape:
            y_pred = self.base(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.base.losses)

            ewc_pen = tf.constant(0.0, dtype=loss.dtype)
            for v in self.base.trainable_variables:
                name = v.name
                if name in self.fisher:
                    delta = v - self.theta_old[name]
                    ewc_pen += tf.reduce_sum(self.fisher[name] * tf.square(delta))

            loss = loss + 0.5 * self.ewc_lambda * ewc_pen

        grads = tape.gradient(loss, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs

def snapshot_weights(model: tf.keras.Model, exclude_patterns: Tuple[str, ...]) -> Dict[str, tf.Tensor]:
    snap = {}
    for v in model.trainable_variables:
        if any(p in v.name for p in exclude_patterns):
            continue
        snap[v.name] = tf.identity(v)
    return snap

def smooth_labels_binary(y: tf.Tensor, smooth: float) -> tf.Tensor:
    # y' = y*(1-s) + 0.5*s  -> 0 -> 0.5s, 1 -> 1-0.5s
    y = tf.cast(y, tf.float32)
    return y * (1.0 - smooth) + 0.5 * smooth

def compute_fisher_diagonal(model: tf.keras.Model,
                            X_tokens: np.ndarray,
                            y: np.ndarray,
                            domain_ids: Optional[np.ndarray],
                            max_samples: int,
                            batch_size: int,
                            exclude_patterns: Tuple[str, ...],
                            label_smooth: float = FISHER_LABEL_SMOOTH,
                            training_true: bool = FISHER_TRAINING_TRUE) -> Tuple[Dict[str, tf.Tensor], float]:
    """
    Fisher ≈ (1/N) Σ (∂ log p(y|x,θ) / ∂θ)^2
    实践中为了避免过于自信导致梯度≈0：
      - 用 label smoothing
      - 用 training=True 打开 dropout
      - 用 sum loss（不是 mean）
    """
    rng = np.random.RandomState(SEED)
    n_total = len(y)
    n = min(n_total, int(max_samples))
    idx = rng.choice(n_total, size=n, replace=False)

    Xt = X_tokens[idx]
    yt = y[idx].astype(np.float32).reshape(-1, 1)

    if domain_ids is not None:
        dt = domain_ids[idx].astype(np.int32)
        ds = tf.data.Dataset.from_tensor_slices(({"tokens": Xt, "domain_id": dt}, yt)).batch(batch_size)
    else:
        ds = tf.data.Dataset.from_tensor_slices((Xt, yt)).batch(batch_size)

    shared_vars = [v for v in model.trainable_variables if not any(p in v.name for p in exclude_patterns)]
    fisher = {v.name: tf.zeros_like(v) for v in shared_vars}

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
    total = tf.constant(0, dtype=tf.int32)

    for x_b, y_b in ds:
        y_b = tf.cast(y_b, tf.float32)
        y_b = tf.reshape(y_b, (-1, 1))
        y_s = smooth_labels_binary(y_b, label_smooth)

        with tf.GradientTape() as tape:
            y_pred = model(x_b, training=training_true)
            nll = tf.reduce_sum(bce(y_s, y_pred))  # sum, not mean

        grads = tape.gradient(nll, shared_vars)
        bs = tf.shape(y_b)[0]
        total += bs

        for v, g in zip(shared_vars, grads):
            if g is not None:
                fisher[v.name] = fisher[v.name] + tf.square(g)

    total_f = tf.cast(total, next(iter(fisher.values())).dtype)
    for name in fisher:
        fisher[name] = fisher[name] / (total_f + 1e-12)

    fisher_mean = float(np.mean([float(tf.reduce_mean(v).numpy()) for v in fisher.values()])) if len(fisher) else 0.0
    return fisher, fisher_mean


# =========================================================
# 6. Load & filter data
# =========================================================
print("Loading datasets…")
fake_df = pd.read_csv(PATH_FAKE)
real_df = pd.read_csv(PATH_REAL)
covid_fake_df = pd.read_csv(PATH_COVID_FAKE)
covid_real_df = pd.read_csv(PATH_COVID_REAL)

# Political
pol_fake = fake_df[(fake_df["subject"] == "politics") & (fake_df["text"].astype(str).str.len() >= 40)]["text"].astype(str)
pol_real = real_df[(real_df["subject"] == "politicsNews") & (real_df["text"].astype(str).str.len() >= 40)]["text"].astype(str)

policy_texts = pd.concat([pol_fake, pol_real]).tolist()
policy_labels = np.concatenate([np.zeros(len(pol_fake), dtype=int),
                                np.ones(len(pol_real), dtype=int)])
print(f"Political | fake={len(pol_fake)}  real={len(pol_real)}  total={len(policy_texts)}")

# Medical
def pick_text_col(df: pd.DataFrame) -> str:
    for c in ["Text", "text", "content", "Content"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find text column in covid dataset")

col_f = pick_text_col(covid_fake_df)
col_r = pick_text_col(covid_real_df)

med_fake = covid_fake_df[covid_fake_df[col_f].astype(str).str.len() >= 40][col_f].astype(str)
med_real = covid_real_df[covid_real_df[col_r].astype(str).str.len() >= 40][col_r].astype(str)

medical_texts = pd.concat([med_fake, med_real]).tolist()
medical_labels = np.concatenate([np.zeros(len(med_fake), dtype=int),
                                 np.ones(len(med_real), dtype=int)])
print(f"Medical   | fake={len(med_fake)}   real={len(med_real)}   total={len(medical_texts)}")


# =========================================================
# 7. Split (stratified)
# =========================================================
X_train_p_texts, X_val_p_texts, y_train_p, y_val_p = train_test_split(
    policy_texts, policy_labels, test_size=0.2, random_state=SEED, stratify=policy_labels
)
X_train_m_texts, X_val_m_texts, y_train_m_full, y_val_m = train_test_split(
    medical_texts, medical_labels, test_size=0.2, random_state=SEED, stratify=medical_labels
)

print("\nSplit sizes:")
print(f"Political train={len(X_train_p_texts)}  val={len(X_val_p_texts)}")
print(f"Medical   train={len(X_train_m_texts)}  val={len(X_val_m_texts)}")

# =========================================================
# 8. Unified Tokenizer TOK_ALL
# =========================================================
print("\nBuilding unified tokenizer TOK_ALL on (POL_train + MED_train)…")
TOK_ALL, VOCAB_ALL = build_tokenizer(X_train_p_texts + X_train_m_texts, MAX_NUM_WORDS)
print(f"VOCAB_ALL={VOCAB_ALL}")

X_train_p = vectorize(TOK_ALL, X_train_p_texts, MAXLEN)
X_val_p = vectorize(TOK_ALL, X_val_p_texts, MAXLEN)

X_train_m_full_vec = vectorize(TOK_ALL, X_train_m_texts, MAXLEN)
X_val_m_vec = vectorize(TOK_ALL, X_val_m_texts, MAXLEN)

D_train_p = domain_ids_like(len(X_train_p), 0)
D_val_p = domain_ids_like(len(X_val_p), 0)
D_val_m = domain_ids_like(len(X_val_m_vec), 1)


# =========================================================
# 9. Pretrain (Plain + Adapter) + Fisher (with scaling)
# =========================================================
# ---- Plain pretrain
print("\n[Pretrain A] Plain CNN on POLITICAL…")
clear_tf(); reset_seeds(SEED)

plain_pre = build_plain_cnn(VOCAB_ALL)
t0 = time.time()
plain_pre.fit(X_train_p, y_train_p,
              epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val_p, y_val_p),
              verbose=1)
t_pre_plain = time.time() - t0
print(f"Plain pretrain time: {t_pre_plain:.2f}s")

prob_pol_pre_plain = plain_pre.predict(X_val_p, verbose=0).ravel()
met_pol_pre_plain = evaluate_probs(prob_pol_pre_plain, y_val_p, name="Political Val (Plain pretrain)")

plain_pre_weights = plain_pre.get_weights()

# Fisher Plain
t_fisher_plain = 0.0
plain_fisher, plain_theta_old, plain_fisher_mean = {}, {}, 0.0
if USE_EWC:
    print("\nComputing Fisher (Plain)…")
    t0 = time.time()
    # Plain：尽量不要 exclude emb（否则 EWC 很难限制“最容易漂移的词向量”）
    PLAIN_EXCLUDE = tuple()   # 不排除任何（你也可以尝试只排除 out，看效果）
    plain_fisher, plain_fisher_mean = compute_fisher_diagonal(
        plain_pre, X_train_p, y_train_p, domain_ids=None,
        max_samples=FISHER_SAMPLES, batch_size=FISHER_BATCH_SIZE,
        exclude_patterns=PLAIN_EXCLUDE,
        label_smooth=FISHER_LABEL_SMOOTH, training_true=FISHER_TRAINING_TRUE
    )
    plain_theta_old = snapshot_weights(plain_pre, exclude_patterns=PLAIN_EXCLUDE)
    t_fisher_plain = time.time() - t0
    print(f"Plain Fisher mean={plain_fisher_mean:.6e}  fisher_time={t_fisher_plain:.2f}s")

# ---- Adapter pretrain
print("\n[Pretrain B] Adapter CNN on POLITICAL (domain=0)…")
clear_tf(); reset_seeds(SEED)

adapter_pre = build_domain_adapter_cnn(VOCAB_ALL)
t0 = time.time()
adapter_pre.fit({"tokens": X_train_p, "domain_id": D_train_p}, y_train_p,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=({"tokens": X_val_p, "domain_id": D_val_p}, y_val_p),
                verbose=1)
t_pre_adp = time.time() - t0
print(f"Adapter pretrain time: {t_pre_adp:.2f}s")

prob_pol_pre_adp = adapter_pre.predict({"tokens": X_val_p, "domain_id": D_val_p}, verbose=0).ravel()
met_pol_pre_adp = evaluate_probs(prob_pol_pre_adp, y_val_p, name="Political Val (Adapter pretrain)")

adapter_pre_weights = adapter_pre.get_weights()

# Fisher Adapter
t_fisher_adp = 0.0
adapter_fisher, adapter_theta_old, adapter_fisher_mean = {}, {}, 0.0
if USE_EWC:
    print("\nComputing Fisher (Adapter shared trunk)…")
    t0 = time.time()
    # Adapter：只约束 shared trunk（emb/conv/fc/out），不约束 adapter/gate/dom
    ADP_EXCLUDE = ("adapter_", "dom_", "gate_", "stat_proj")
    adapter_fisher, adapter_fisher_mean = compute_fisher_diagonal(
        adapter_pre, X_train_p, y_train_p, domain_ids=D_train_p,
        max_samples=FISHER_SAMPLES, batch_size=FISHER_BATCH_SIZE,
        exclude_patterns=ADP_EXCLUDE,
        label_smooth=FISHER_LABEL_SMOOTH, training_true=FISHER_TRAINING_TRUE
    )
    adapter_theta_old = snapshot_weights(adapter_pre, exclude_patterns=ADP_EXCLUDE)
    t_fisher_adp = time.time() - t0
    print(f"Adapter Fisher mean={adapter_fisher_mean:.6e}  fisher_time={t_fisher_adp:.2f}s")

# ---- Auto scale lambda (关键：让 Adapter 的 EWC 强度不再“几乎为0”)
ewc_lambda_plain = EWC_LAMBDA_BASE
ewc_lambda_adp = EWC_LAMBDA_BASE

if USE_EWC and AUTO_SCALE_EWC_LAMBDA:
    # 让 “lambda * fisher_mean” 在两个模型上处于同一量级
    # 等价于：Adapter fisher 小 -> lambda 自动变大
    if plain_fisher_mean > 0 and adapter_fisher_mean > 0:
        scale = (plain_fisher_mean + EWC_EPS) / (adapter_fisher_mean + EWC_EPS)
        ewc_lambda_adp = EWC_LAMBDA_BASE * scale
    print(f"\nAuto-scaled EWC lambdas:")
    print(f"  lambda_plain = {ewc_lambda_plain:.3f}")
    print(f"  lambda_adapter = {ewc_lambda_adp:.3f}  (scaled)")

# =========================================================
# 10. Methods list (持续学习 + 时间效率 + 消融)
# =========================================================
METHODS = [
    "Transfer-Plain (Full FT)",
    "Transfer-Plain+EWC (Full FT)",
    "Adapter-PEFT (freeze emb+conv)",
    "OURS-noEWC (unfreeze conv)",
    "OURS=Adapter+EWC (unfreeze conv + EWC)",
]
if INCLUDE_MED_SCRATCH:
    METHODS.append("Med-Scratch (TOK_ALL)")
if INCLUDE_JOINT_REPLAY_UPPER:
    METHODS.append("Replay Upper (Joint Plain: POL_train + MED_sub)")


def compile_plain(model: Model, lr: float):
    model.compile(optimizer=make_optimizer(lr), loss="binary_crossentropy", metrics=["accuracy"])

def compile_adapter(model: Model, lr: float):
    model.compile(optimizer=make_optimizer(lr), loss="binary_crossentropy", metrics=["accuracy"])


records = []

def add_record(frac: float, method: str,
               met_med: Dict[str, float],
               met_pol_after: Optional[Dict[str, float]],
               acc_pol_pre: Optional[float], bce_pol_pre: Optional[float],
               t_pretrain: float, t_fisher: float, t_finetune: float,
               trainable_params: int):
    if met_pol_after is None or acc_pol_pre is None or bce_pol_pre is None:
        forget_acc = np.nan
        forget_bce = np.nan
        avg_acc = np.nan
    else:
        forget_acc = float(acc_pol_pre - met_pol_after["acc"])      # ↓ 越小越好
        forget_bce = float(met_pol_after["bce"] - bce_pol_pre)      # ↑ 越小越好
        avg_acc = 0.5 * float(met_pol_after["acc"] + met_med["acc"]) # ↑ 越大越好（两任务平均）

    records.append({
        "frac": frac,
        "method": method,
        "acc_med": met_med["acc"],
        "f1_med": met_med["f1"],
        "bce_med": met_med["bce"],
        "acc_pol_pre": acc_pol_pre if acc_pol_pre is not None else np.nan,
        "bce_pol_pre": bce_pol_pre if bce_pol_pre is not None else np.nan,
        "acc_pol_after": met_pol_after["acc"] if met_pol_after else np.nan,
        "f1_pol_after": met_pol_after["f1"] if met_pol_after else np.nan,
        "bce_pol_after": met_pol_after["bce"] if met_pol_after else np.nan,
        "forget_acc": forget_acc,
        "forget_bce": forget_bce,
        "avg_acc_after": avg_acc,
        "trainable_params": trainable_params,
        "t_pretrain": t_pretrain,
        "t_fisher": t_fisher,
        "t_finetune": t_finetune,
        "t_total_once": float(t_pretrain + t_fisher + t_finetune)
    })


# =========================================================
# 11. Fractions loop
# =========================================================
for frac in fractions:
    print("\n" + "=" * 110)
    print(f"Using {int(frac * 100)}% of MEDICAL train data…")

    X_train_m_sub_texts, y_train_m_sub = stratified_subset(X_train_m_texts, y_train_m_full, frac, seed=SEED)
    X_train_m_sub = vectorize(TOK_ALL, X_train_m_sub_texts, MAXLEN)
    D_train_m = domain_ids_like(len(X_train_m_sub), 1)

    # -----------------------------------------------------
    # (1) Transfer-Plain (Full FT)
    # -----------------------------------------------------
    print("\n[1] Transfer-Plain (Full FT)")
    clear_tf(); reset_seeds(SEED)

    m = build_plain_cnn(VOCAB_ALL)
    m.set_weights(plain_pre_weights)
    set_trainable_plain_full(m)
    compile_plain(m, LR_PLAIN_FT)
    tp = count_trainable_params(m)

    t_ft = timed_fit(m, X_train_m_sub, y_train_m_sub, X_val_m_vec, y_val_m, verbose=0)

    prob_med = m.predict(X_val_m_vec, verbose=0).ravel()
    met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Plain FT")

    prob_pol_after = m.predict(X_val_p, verbose=0).ravel()
    met_pol_after = evaluate_probs(prob_pol_after, y_val_p, name="Political Val(after FT) | Plain FT")

    add_record(frac, "Transfer-Plain (Full FT)", met_med, met_pol_after,
               acc_pol_pre=met_pol_pre_plain["acc"], bce_pol_pre=met_pol_pre_plain["bce"],
               t_pretrain=t_pre_plain, t_fisher=0.0, t_finetune=t_ft, trainable_params=tp)

    # -----------------------------------------------------
    # (2) Transfer-Plain+EWC (Full FT)
    # -----------------------------------------------------
    if USE_EWC:
        print("\n[2] Transfer-Plain+EWC (Full FT)")
        clear_tf(); reset_seeds(SEED)

        base = build_plain_cnn(VOCAB_ALL)
        base.set_weights(plain_pre_weights)
        set_trainable_plain_full(base)
        compile_plain(base, LR_PLAIN_FT)

        tp = count_trainable_params(base)
        ewc = EWCModel(base, fisher=plain_fisher, theta_old=plain_theta_old, ewc_lambda=ewc_lambda_plain)
        ewc.compile(optimizer=make_optimizer(LR_PLAIN_FT), loss="binary_crossentropy", metrics=["accuracy"])

        t_ft = timed_fit(ewc, X_train_m_sub, y_train_m_sub, X_val_m_vec, y_val_m, verbose=0)

        prob_med = ewc.predict(X_val_m_vec, verbose=0).ravel()
        met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Plain+EWC")

        prob_pol_after = ewc.predict(X_val_p, verbose=0).ravel()
        met_pol_after = evaluate_probs(prob_pol_after, y_val_p, name="Political Val(after FT) | Plain+EWC")

        add_record(frac, "Transfer-Plain+EWC (Full FT)", met_med, met_pol_after,
                   acc_pol_pre=met_pol_pre_plain["acc"], bce_pol_pre=met_pol_pre_plain["bce"],
                   t_pretrain=t_pre_plain, t_fisher=t_fisher_plain, t_finetune=t_ft, trainable_params=tp)

    # -----------------------------------------------------
    # (3) Adapter-PEFT baseline (freeze emb+conv)
    # -----------------------------------------------------
    print("\n[3] Adapter-PEFT (freeze emb+conv)")
    clear_tf(); reset_seeds(SEED)

    adp = build_domain_adapter_cnn(VOCAB_ALL)
    adp.set_weights(adapter_pre_weights)
    set_trainable_adapter_peft(adp)
    compile_adapter(adp, LR_ADAPTER_PEFT)
    tp = count_trainable_params(adp)

    t_ft = timed_fit(adp,
                     {"tokens": X_train_m_sub, "domain_id": D_train_m}, y_train_m_sub,
                     {"tokens": X_val_m_vec, "domain_id": D_val_m}, y_val_m,
                     verbose=0)

    prob_med = adp.predict({"tokens": X_val_m_vec, "domain_id": D_val_m}, verbose=0).ravel()
    met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Adapter-PEFT")

    prob_pol_after = adp.predict({"tokens": X_val_p, "domain_id": D_val_p}, verbose=0).ravel()
    met_pol_after = evaluate_probs(prob_pol_after, y_val_p, name="Political Val(after FT) | Adapter-PEFT")

    add_record(frac, "Adapter-PEFT (freeze emb+conv)", met_med, met_pol_after,
               acc_pol_pre=met_pol_pre_adp["acc"], bce_pol_pre=met_pol_pre_adp["bce"],
               t_pretrain=t_pre_adp, t_fisher=0.0, t_finetune=t_ft, trainable_params=tp)

    # -----------------------------------------------------
    # (4) OURS-noEWC (unfreeze conv)  —— 用来证明“EWC 的必要性”
    # -----------------------------------------------------
    print("\n[4] OURS-noEWC (unfreeze conv)")
    clear_tf(); reset_seeds(SEED)

    ours0 = build_domain_adapter_cnn(VOCAB_ALL)
    ours0.set_weights(adapter_pre_weights)
    set_trainable_adapter_ours(ours0)
    compile_adapter(ours0, LR_OURS_TRUNK)
    tp = count_trainable_params(ours0)

    t_ft = timed_fit(ours0,
                     {"tokens": X_train_m_sub, "domain_id": D_train_m}, y_train_m_sub,
                     {"tokens": X_val_m_vec, "domain_id": D_val_m}, y_val_m,
                     verbose=0)

    prob_med = ours0.predict({"tokens": X_val_m_vec, "domain_id": D_val_m}, verbose=0).ravel()
    met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | OURS-noEWC")

    prob_pol_after = ours0.predict({"tokens": X_val_p, "domain_id": D_val_p}, verbose=0).ravel()
    met_pol_after = evaluate_probs(prob_pol_after, y_val_p, name="Political Val(after FT) | OURS-noEWC")

    add_record(frac, "OURS-noEWC (unfreeze conv)", met_med, met_pol_after,
               acc_pol_pre=met_pol_pre_adp["acc"], bce_pol_pre=met_pol_pre_adp["bce"],
               t_pretrain=t_pre_adp, t_fisher=0.0, t_finetune=t_ft, trainable_params=tp)

    # -----------------------------------------------------
    # (5) OURS=Adapter+EWC
    # -----------------------------------------------------
    if USE_EWC:
        print("\n[5] OURS=Adapter+EWC (unfreeze conv + EWC)")
        clear_tf(); reset_seeds(SEED)

        base = build_domain_adapter_cnn(VOCAB_ALL)
        base.set_weights(adapter_pre_weights)
        set_trainable_adapter_ours(base)
        compile_adapter(base, LR_OURS_TRUNK)
        tp = count_trainable_params(base)

        ewc = EWCModel(base, fisher=adapter_fisher, theta_old=adapter_theta_old, ewc_lambda=ewc_lambda_adp)
        ewc.compile(optimizer=make_optimizer(LR_OURS_TRUNK), loss="binary_crossentropy", metrics=["accuracy"])

        t_ft = timed_fit(ewc,
                         {"tokens": X_train_m_sub, "domain_id": D_train_m}, y_train_m_sub,
                         {"tokens": X_val_m_vec, "domain_id": D_val_m}, y_val_m,
                         verbose=0)

        prob_med = ewc.predict({"tokens": X_val_m_vec, "domain_id": D_val_m}, verbose=0).ravel()
        met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | OURS=Adapter+EWC")

        prob_pol_after = ewc.predict({"tokens": X_val_p, "domain_id": D_val_p}, verbose=0).ravel()
        met_pol_after = evaluate_probs(prob_pol_after, y_val_p, name="Political Val(after FT) | OURS=Adapter+EWC")

        add_record(frac, "OURS=Adapter+EWC (unfreeze conv + EWC)", met_med, met_pol_after,
                   acc_pol_pre=met_pol_pre_adp["acc"], bce_pol_pre=met_pol_pre_adp["bce"],
                   t_pretrain=t_pre_adp, t_fisher=t_fisher_adp, t_finetune=t_ft, trainable_params=tp)

    # -----------------------------------------------------
    # (6) Med-Scratch (TOK_ALL)
    # -----------------------------------------------------
    if INCLUDE_MED_SCRATCH:
        print("\n[6] Med-Scratch (TOK_ALL)")
        clear_tf(); reset_seeds(SEED)

        ms = build_plain_cnn(VOCAB_ALL)
        set_trainable_plain_full(ms)
        compile_plain(ms, LR_PLAIN_FT)
        tp = count_trainable_params(ms)

        t_tr = timed_fit(ms, X_train_m_sub, y_train_m_sub, X_val_m_vec, y_val_m, verbose=0)
        prob_med = ms.predict(X_val_m_vec, verbose=0).ravel()
        met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Med-Scratch")

        add_record(frac, "Med-Scratch (TOK_ALL)", met_med, None,
                   acc_pol_pre=None, bce_pol_pre=None,
                   t_pretrain=0.0, t_fisher=0.0, t_finetune=t_tr, trainable_params=tp)

    # -----------------------------------------------------
    # (7) Replay upper bound (Joint training)
    # -----------------------------------------------------
    if INCLUDE_JOINT_REPLAY_UPPER:
        print("\n[7] Replay Upper (Joint Plain: POL_train + MED_sub)")
        clear_tf(); reset_seeds(SEED)

        X_joint = np.concatenate([X_train_p, X_train_m_sub], axis=0)
        y_joint = np.concatenate([y_train_p, y_train_m_sub], axis=0)

        jm = build_plain_cnn(VOCAB_ALL)
        set_trainable_plain_full(jm)
        compile_plain(jm, LR_PLAIN_FT)
        tp = count_trainable_params(jm)

        t_tr = timed_fit(jm, X_joint, y_joint, X_val_m_vec, y_val_m, verbose=0)

        prob_med = jm.predict(X_val_m_vec, verbose=0).ravel()
        met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Replay Upper")

        prob_pol = jm.predict(X_val_p, verbose=0).ravel()
        met_pol = evaluate_probs(prob_pol, y_val_p, name="Political Val | Replay Upper")

        add_record(frac, "Replay Upper (Joint Plain: POL_train + MED_sub)", met_med, met_pol,
                   acc_pol_pre=None, bce_pol_pre=None,
                   t_pretrain=0.0, t_fisher=0.0, t_finetune=t_tr, trainable_params=tp)


# =========================================================
# 12. Save + Pivot + Plots
# =========================================================
df = pd.DataFrame(records)
df.to_csv("comparison_cl_time.csv", index=False)
print("\nSaved: comparison_cl_time.csv")

print("\n==================== Pivot: Medical Acc ====================")
print(df.pivot(index="frac", columns="method", values="acc_med"))

print("\n==================== Pivot: Forgetting (Acc drop) [Lower is better] ====================")
cl_methods = [m for m in METHODS if ("Transfer" in m or "OURS" in m or "Adapter" in m)]
tmp = df[df["method"].isin(cl_methods)]
print(tmp.pivot(index="frac", columns="method", values="forget_acc"))

print("\n==================== Pivot: Forgetting (BCE increase) [Lower is better] ====================")
print(tmp.pivot(index="frac", columns="method", values="forget_bce"))

print("\n==================== Pivot: AvgAcc after step2 (2 tasks mean) [Higher is better] ====================")
print(tmp.pivot(index="frac", columns="method", values="avg_acc_after"))

print("\n==================== Pivot: Finetune time (seconds) ====================")
print(df.pivot(index="frac", columns="method", values="t_finetune"))

print("\n==================== Pivot: Trainable params ====================")
print(df.pivot(index="frac", columns="method", values="trainable_params"))


x_vals = [f * 100 for f in fractions]

plt.figure(figsize=(14, 10))

# (1) Medical Acc
plt.subplot(2, 2, 1)
for m in METHODS:
    tmp2 = df[df["method"] == m].sort_values("frac")
    plt.plot(x_vals, tmp2["acc_med"].values, marker="o", label=m)
plt.title("Medical Val Accuracy vs Medical Train Fraction")
plt.xlabel("Medical Train Fraction (%)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

# (2) Forgetting Acc drop (CL methods)
plt.subplot(2, 2, 2)
for m in cl_methods:
    tmp2 = df[df["method"] == m].sort_values("frac")
    plt.plot(x_vals, tmp2["forget_acc"].values, marker="o", label=m)
plt.title("Forgetting on Political (Acc_pre - Acc_after)  [Lower is better]")
plt.xlabel("Medical Train Fraction (%)")
plt.ylabel("Acc drop")
plt.grid(True)
plt.legend()

# (3) AvgAcc after step2
plt.subplot(2, 2, 3)
for m in cl_methods:
    tmp2 = df[df["method"] == m].sort_values("frac")
    plt.plot(x_vals, tmp2["avg_acc_after"].values, marker="o", label=m)
plt.title("Continual Learning Score: AvgAcc(after step2)  [Higher is better]")
plt.xlabel("Medical Train Fraction (%)")
plt.ylabel("AvgAcc")
plt.grid(True)
plt.legend()

# (4) Finetune time
plt.subplot(2, 2, 4)
for m in METHODS:
    tmp2 = df[df["method"] == m].sort_values("frac")
    plt.plot(x_vals, tmp2["t_finetune"].values, marker="o", label=m)
plt.title("Finetune/Train Time vs Medical Train Fraction")
plt.xlabel("Medical Train Fraction (%)")
plt.ylabel("Time (s)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Trade-off scatter: forgetting vs medical acc (for each fraction)
plt.figure(figsize=(10, 6))
for m in cl_methods:
    tmp2 = df[df["method"] == m].sort_values("frac")
    plt.plot(tmp2["forget_acc"].values, tmp2["acc_med"].values, marker="o", label=m)
plt.title("Stability-Plasticity Trade-off (Forgetting vs Medical Acc)")
plt.xlabel("Forgetting (Acc drop)  [Lower better]")
plt.ylabel("Medical Val Acc  [Higher better]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nDone.")
