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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
)

# =========================================================
# 0. Reproducibility
# =========================================================
SEED = 42

def reset_seeds(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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
USE_DOMAIN_ADAPTER = True
ADAPTER_R = 16
GATE_HIDDEN = 64

# EWC
USE_EWC = True
EWC_LAMBDA = 50.0          # 修复 Fisher 后，lambda 往往要比你之前略大/或相近；你可以试 20/50/100
FISHER_SAMPLES = 2000
FISHER_BATCH_SIZE = 64

# Threshold
CALIBRATE_THRESHOLD = False  # 如果 True：在验证集上选thr最大化F1（偏乐观，论文里不建议）

# Compare settings
fractions = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]

INCLUDE_MED_SCRATCH = True
INCLUDE_JOINT_TRAIN = True
INCLUDE_RETOKENIZE_BASELINE = False  # 这个 baseline 会改变词表，不公平；仅供参考

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
        raise ValueError("frac must be > 0")

    try:
        X_sub, _, y_sub, _ = train_test_split(
            texts, labels, train_size=frac, random_state=seed, stratify=labels
        )
        return list(X_sub), np.array(y_sub)
    except ValueError:
        # fallback: per-class sample at least 1
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

    y_pred = (y_prob > thr).astype('int32')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"{name} | thr={thr:.3f}  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1), "thr": float(thr)}

def fit_with_time(model: tf.keras.Model,
                  train_x, train_y,
                  val_x, val_y,
                  epochs: int = EPOCHS,
                  batch_size: int = BATCH_SIZE,
                  verbose: int = 0) -> float:
    start = time.time()
    model.fit(
        train_x, train_y,
        epochs=epochs, batch_size=batch_size,
        validation_data=(val_x, val_y),
        verbose=verbose
    )
    return time.time() - start

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
    tok_in = L.Input(shape=(maxlen,), dtype='int32', name='tokens')
    x = L.Embedding(vocab_size, embed_dim, input_length=maxlen, name='emb')(tok_in)
    h = L.Conv1D(conv_filters, kernel_size, activation='relu', name='conv')(x)
    pooled = L.GlobalMaxPooling1D(name='gmp')(h)
    z = L.Dense(dense_units, activation='relu', name='fc')(pooled)
    z = L.Dropout(dropout, name='drop')(z)
    out = L.Dense(1, activation='sigmoid', name='out')(z)
    model = Model(tok_in, out, name='PlainCNN')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_domain_adapter_cnn(vocab_size: int,
                             maxlen: int = MAXLEN,
                             embed_dim: int = EMBED_DIM,
                             conv_filters: int = 128,
                             kernel_size: int = 5,
                             dense_units: int = 10,
                             dropout: float = 0.5,
                             use_adapter: bool = USE_DOMAIN_ADAPTER,
                             adapter_r: int = ADAPTER_R,
                             gate_hidden: int = GATE_HIDDEN) -> Model:
    tok_in = L.Input(shape=(maxlen,), dtype='int32', name='tokens')
    dom_in = L.Input(shape=(), dtype='int32', name='domain_id')  # 0=political, 1=medical

    x = L.Embedding(vocab_size, embed_dim, input_length=maxlen, name='emb')(tok_in)
    h = L.Conv1D(conv_filters, kernel_size, activation='relu', name='conv')(x)

    if use_adapter:
        # Houlsby adapter (time-distributed bottleneck) + residual
        a = L.TimeDistributed(L.Dense(adapter_r, activation='relu'), name='adapter_down')(h)
        a = L.TimeDistributed(L.Dense(conv_filters), name='adapter_up')(a)
        h = L.Add(name='adapter_residual')([h, a])

        # Domain-conditioned gate (SE-like)
        d = L.Embedding(input_dim=2, output_dim=gate_hidden, name='dom_emb')(dom_in)
        d = L.Dense(gate_hidden, activation='relu', name='dom_proj1')(d)
        d = L.Dense(conv_filters, activation=None, name='dom_proj2')(d)
        d = L.Reshape((1, conv_filters), name='dom_reshape')(d)

        g_stat = L.GlobalAveragePooling1D(name='gap')(h)
        g_stat = L.Dense(conv_filters, activation=None, name='stat_proj')(g_stat)
        g_stat = L.Reshape((1, conv_filters), name='stat_reshape')(g_stat)

        gate_preact = L.Add(name='gate_preact')([d, g_stat])
        gate = L.Activation('sigmoid', name='gate_sigmoid')(gate_preact)
        h = L.Multiply(name='gated')([h, gate])

    pooled = L.GlobalMaxPooling1D(name='gmp')(h)
    z = L.Dense(dense_units, activation='relu', name='fc')(pooled)
    z = L.Dropout(dropout, name='drop')(z)
    out = L.Dense(1, activation='sigmoid', name='out')(z)

    model = Model(inputs={'tokens': tok_in, 'domain_id': dom_in}, outputs=out, name='DomainAdapterCNN')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =========================================================
# 4. EWC (Fix Fisher scaling!)
# =========================================================
class EWCModel(tf.keras.Model):
    def __init__(self, base_model: tf.keras.Model,
                 fisher: Dict[str, tf.Tensor],
                 theta_old: Dict[str, tf.Tensor],
                 ewc_lambda: float = EWC_LAMBDA):
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

def compute_fisher_diagonal(
    model: tf.keras.Model,
    X_tokens: np.ndarray,
    y: np.ndarray,
    domain_ids: Optional[np.ndarray] = None,
    max_samples: int = FISHER_SAMPLES,
    batch_size: int = FISHER_BATCH_SIZE,
    exclude_patterns: Tuple[str, ...] = ("emb",)
) -> Dict[str, tf.Tensor]:
    """
    ✅ 修复版 Fisher：用 sum NLL（不是 mean），避免 Fisher 被 1/B^2 缩小到几乎 0。
    Fisher ≈ (1/N) Σ (∂ log p(y|x,θ) / ∂θ)^2
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

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
    total = tf.constant(0, dtype=tf.int32)

    for x_b, y_b in ds:
        with tf.GradientTape() as tape:
            y_pred = model(x_b, training=False)
            # 关键：sum，不要 mean
            nll = tf.reduce_sum(bce(tf.cast(y_b, y_pred.dtype), y_pred))
        grads = tape.gradient(nll, shared_vars)

        bs = tf.shape(y_b)[0]
        total += bs
        for v, g in zip(shared_vars, grads):
            if g is not None:
                fisher[v.name] = fisher[v.name] + tf.square(g)

    total_f = tf.cast(total, next(iter(fisher.values())).dtype)
    for name in fisher:
        fisher[name] = fisher[name] / total_f
    return fisher

# =========================================================
# 5. Load & filter data
# =========================================================
print("Loading datasets…")
fake_df = pd.read_csv(PATH_FAKE)
real_df = pd.read_csv(PATH_REAL)
covid_fake_df = pd.read_csv(PATH_COVID_FAKE)
covid_real_df = pd.read_csv(PATH_COVID_REAL)

# Political
pol_fake = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].astype(str).str.len() >= 40)]['text'].astype(str)
pol_real = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].astype(str).str.len() >= 40)]['text'].astype(str)

policy_texts = pd.concat([pol_fake, pol_real]).tolist()
policy_labels = np.concatenate([np.zeros(len(pol_fake), dtype=int),
                                np.ones(len(pol_real), dtype=int)])
print(f"Political | fake={len(pol_fake)}  real={len(pol_real)}  total={len(policy_texts)}")

# Medical (COVID) — auto pick text col
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
# 6. Split (stratified)
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
# 7. ✅ Unified tokenizer: fit on (political train + medical train) texts
# =========================================================
print("\nBuilding unified tokenizer TOK_ALL on (POL_train + MED_train)…")
TOK_ALL, VOCAB_ALL = build_tokenizer(X_train_p_texts + X_train_m_texts, MAX_NUM_WORDS)
print(f"VOCAB_ALL={VOCAB_ALL}")

X_train_p = vectorize(TOK_ALL, X_train_p_texts, MAXLEN)
X_val_p = vectorize(TOK_ALL, X_val_p_texts, MAXLEN)

X_train_m_full = vectorize(TOK_ALL, X_train_m_texts, MAXLEN)
X_val_m = vectorize(TOK_ALL, X_val_m_texts, MAXLEN)

D_train_p = domain_ids_like(len(X_train_p), 0)
D_val_p = domain_ids_like(len(X_val_p), 0)
D_val_m = domain_ids_like(len(X_val_m), 1)

# =========================================================
# 8. Pretrain Plain on political + Fisher
# =========================================================
print("\n[Pretrain A] Plain CNN on POLITICAL (TOK_ALL)…")
tf.keras.backend.clear_session()
gc.collect()
reset_seeds(SEED)

plain_pre = build_plain_cnn(VOCAB_ALL)
t0 = time.time()
plain_pre.fit(X_train_p, y_train_p, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val_p, y_val_p), verbose=1)
t_pre_plain = time.time() - t0
print(f"Plain pretrain time: {t_pre_plain:.2f}s")

# pretrain metrics on political val
prob_pol_pre_plain = plain_pre.predict(X_val_p, verbose=0).ravel()
met_pol_pre_plain = evaluate_probs(prob_pol_pre_plain, y_val_p, name="Political Val (Plain pretrain)")

# optional: zero-shot on medical
prob_med_zs_plain = plain_pre.predict(X_val_m, verbose=0).ravel()
_ = evaluate_probs(prob_med_zs_plain, y_val_m, name="Zero-shot Plain (political->medical)")

plain_pre_weights = plain_pre.get_weights()

plain_fisher, plain_theta_old = {}, {}
if USE_EWC:
    print("Computing Fisher (Plain, fixed scaling)…")
    PLAIN_EWC_EXCLUDE = ("emb",)  # 不对 embedding 做 EWC，避免主导（可按需改）
    plain_fisher = compute_fisher_diagonal(
        plain_pre, X_train_p, y_train_p,
        domain_ids=None,
        max_samples=FISHER_SAMPLES,
        batch_size=FISHER_BATCH_SIZE,
        exclude_patterns=PLAIN_EWC_EXCLUDE
    )
    plain_theta_old = snapshot_weights(plain_pre, exclude_patterns=PLAIN_EWC_EXCLUDE)

    # quick sanity: show fisher magnitude
    fisher_means = [float(tf.reduce_mean(v).numpy()) for v in plain_fisher.values()]
    print(f"Plain Fisher mean(avg over vars)={np.mean(fisher_means):.6e}  (should NOT be ~0)")

# =========================================================
# 9. Pretrain Adapter on political + Fisher
# =========================================================
print("\n[Pretrain B] Adapter CNN on POLITICAL (TOK_ALL, domain=0)…")
tf.keras.backend.clear_session()
gc.collect()
reset_seeds(SEED)

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

prob_med_zs_adp = adapter_pre.predict({"tokens": X_val_m, "domain_id": D_val_m}, verbose=0).ravel()
_ = evaluate_probs(prob_med_zs_adp, y_val_m, name="Zero-shot Adapter (political->medical)")

adapter_pre_weights = adapter_pre.get_weights()

adapter_fisher, adapter_theta_old = {}, {}
if USE_EWC:
    print("Computing Fisher (Adapter shared trunk, fixed scaling)…")
    # 只约束共享主干：conv/fc/out 等；不约束 adapter/gate/domain embedding
    ADP_EWC_EXCLUDE = ("emb", "adapter_", "dom_", "gate_", "stat_proj")
    adapter_fisher = compute_fisher_diagonal(
        adapter_pre, X_train_p, y_train_p,
        domain_ids=D_train_p,
        max_samples=FISHER_SAMPLES,
        batch_size=FISHER_BATCH_SIZE,
        exclude_patterns=ADP_EWC_EXCLUDE
    )
    adapter_theta_old = snapshot_weights(adapter_pre, exclude_patterns=ADP_EWC_EXCLUDE)

    fisher_means = [float(tf.reduce_mean(v).numpy()) for v in adapter_fisher.values()]
    print(f"Adapter Fisher mean(avg over vars)={np.mean(fisher_means):.6e}  (should NOT be ~0)")

# =========================================================
# 10. Fractions loop: compare all methods
# =========================================================
METHODS = [
    "Transfer-Plain",
    "Transfer-Plain+EWC",
    "Transfer-Adapter",
    "Transfer-Adapter+EWC (OURS)",
]
if INCLUDE_MED_SCRATCH:
    METHODS.append("Med-Scratch (TOK_ALL)")
if INCLUDE_JOINT_TRAIN:
    METHODS.append("Joint-Plain (POL_train + MED_sub, TOK_ALL)")
if INCLUDE_RETOKENIZE_BASELINE:
    METHODS.append("Joint-Plain (retokenize each frac)")

records = []

def record_result(frac, method, met_med, met_pol_after,
                  t_pretrain, t_finetune,
                  acc_pol_pre=None, f1_pol_pre=None):
    acc_pol_after = met_pol_after["acc"] if met_pol_after is not None else np.nan
    f1_pol_after = met_pol_after["f1"] if met_pol_after is not None else np.nan

    if acc_pol_pre is None:
        forget = np.nan
        retain = np.nan
    else:
        forget = float(acc_pol_pre - acc_pol_after)
        retain = float(acc_pol_after / (acc_pol_pre + 1e-12))

    records.append({
        "frac": frac,
        "method": method,
        "acc_med": met_med["acc"],
        "f1_med": met_med["f1"],
        "acc_pol_pre": acc_pol_pre if acc_pol_pre is not None else np.nan,
        "f1_pol_pre": f1_pol_pre if f1_pol_pre is not None else np.nan,
        "acc_pol_after": acc_pol_after,
        "f1_pol_after": f1_pol_after,
        "forget_acc": forget,      # 越小越好
        "retain_ratio": retain,    # 越大越好
        "t_pretrain": t_pretrain,
        "t_finetune": t_finetune,
        "t_total": (t_pretrain + t_finetune)
    })

for frac in fractions:
    print("\n" + "=" * 100)
    print(f"Using {int(frac * 100)}% of MEDICAL train data…")

    # subset medical train TEXTS
    X_train_m_sub_texts, y_train_m_sub = stratified_subset(X_train_m_texts, y_train_m_full, frac, seed=SEED)
    X_train_m_sub = vectorize(TOK_ALL, X_train_m_sub_texts, MAXLEN)
    D_train_m = domain_ids_like(len(X_train_m_sub), 1)

    # -----------------------------------------------------
    # (1) Transfer-Plain
    # -----------------------------------------------------
    print("\n[1] Transfer-Plain")
    tf.keras.backend.clear_session(); gc.collect()
    reset_seeds(SEED)

    m = build_plain_cnn(VOCAB_ALL)
    m.set_weights(plain_pre_weights)
    t_ft = fit_with_time(m, X_train_m_sub, y_train_m_sub, X_val_m, y_val_m, verbose=0)

    prob_med = m.predict(X_val_m, verbose=0).ravel()
    met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Transfer-Plain")

    prob_pol_after = m.predict(X_val_p, verbose=0).ravel()
    met_pol_after = evaluate_probs(prob_pol_after, y_val_p, name="Political Val(after FT) | Transfer-Plain")

    record_result(frac, "Transfer-Plain", met_med, met_pol_after,
                  t_pre_plain, t_ft,
                  acc_pol_pre=met_pol_pre_plain["acc"], f1_pol_pre=met_pol_pre_plain["f1"])

    # -----------------------------------------------------
    # (2) Transfer-Plain+EWC
    # -----------------------------------------------------
    if USE_EWC:
        print("\n[2] Transfer-Plain+EWC")
        tf.keras.backend.clear_session(); gc.collect()
        reset_seeds(SEED)

        base = build_plain_cnn(VOCAB_ALL)
        base.set_weights(plain_pre_weights)

        ewc = EWCModel(base, fisher=plain_fisher, theta_old=plain_theta_old, ewc_lambda=EWC_LAMBDA)
        ewc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        t_ft = fit_with_time(ewc, X_train_m_sub, y_train_m_sub, X_val_m, y_val_m, verbose=0)

        prob_med = ewc.predict(X_val_m, verbose=0).ravel()
        met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Transfer-Plain+EWC")

        prob_pol_after = ewc.predict(X_val_p, verbose=0).ravel()
        met_pol_after = evaluate_probs(prob_pol_after, y_val_p, name="Political Val(after FT) | Transfer-Plain+EWC")

        record_result(frac, "Transfer-Plain+EWC", met_med, met_pol_after,
                      t_pre_plain, t_ft,
                      acc_pol_pre=met_pol_pre_plain["acc"], f1_pol_pre=met_pol_pre_plain["f1"])

    # -----------------------------------------------------
    # (3) Transfer-Adapter
    # -----------------------------------------------------
    print("\n[3] Transfer-Adapter")
    tf.keras.backend.clear_session(); gc.collect()
    reset_seeds(SEED)

    adp = build_domain_adapter_cnn(VOCAB_ALL)
    adp.set_weights(adapter_pre_weights)
    t_ft = fit_with_time(
        adp,
        {"tokens": X_train_m_sub, "domain_id": D_train_m}, y_train_m_sub,
        {"tokens": X_val_m, "domain_id": D_val_m}, y_val_m,
        verbose=0
    )

    prob_med = adp.predict({"tokens": X_val_m, "domain_id": D_val_m}, verbose=0).ravel()
    met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Transfer-Adapter")

    prob_pol_after = adp.predict({"tokens": X_val_p, "domain_id": D_val_p}, verbose=0).ravel()
    met_pol_after = evaluate_probs(prob_pol_after, y_val_p, name="Political Val(after FT) | Transfer-Adapter")

    record_result(frac, "Transfer-Adapter", met_med, met_pol_after,
                  t_pre_adp, t_ft,
                  acc_pol_pre=met_pol_pre_adp["acc"], f1_pol_pre=met_pol_pre_adp["f1"])

    # -----------------------------------------------------
    # (4) Transfer-Adapter+EWC (OURS)
    # -----------------------------------------------------
    if USE_EWC:
        print("\n[4] Transfer-Adapter+EWC (OURS)")
        tf.keras.backend.clear_session(); gc.collect()
        reset_seeds(SEED)

        base = build_domain_adapter_cnn(VOCAB_ALL)
        base.set_weights(adapter_pre_weights)

        ewc = EWCModel(base, fisher=adapter_fisher, theta_old=adapter_theta_old, ewc_lambda=EWC_LAMBDA)
        ewc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        t_ft = fit_with_time(
            ewc,
            {"tokens": X_train_m_sub, "domain_id": D_train_m}, y_train_m_sub,
            {"tokens": X_val_m, "domain_id": D_val_m}, y_val_m,
            verbose=0
        )

        prob_med = ewc.predict({"tokens": X_val_m, "domain_id": D_val_m}, verbose=0).ravel()
        met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Transfer-Adapter+EWC (OURS)")

        prob_pol_after = ewc.predict({"tokens": X_val_p, "domain_id": D_val_p}, verbose=0).ravel()
        met_pol_after = evaluate_probs(prob_pol_after, y_val_p, name="Political Val(after FT) | Transfer-Adapter+EWC (OURS)")

        record_result(frac, "Transfer-Adapter+EWC (OURS)", met_med, met_pol_after,
                      t_pre_adp, t_ft,
                      acc_pol_pre=met_pol_pre_adp["acc"], f1_pol_pre=met_pol_pre_adp["f1"])

    # -----------------------------------------------------
    # (5) Med-Scratch (same TOK_ALL, fair baseline)
    # -----------------------------------------------------
    if INCLUDE_MED_SCRATCH:
        print("\n[5] Med-Scratch (TOK_ALL)")
        tf.keras.backend.clear_session(); gc.collect()
        reset_seeds(SEED)

        ms = build_plain_cnn(VOCAB_ALL)
        t_tr = fit_with_time(ms, X_train_m_sub, y_train_m_sub, X_val_m, y_val_m, verbose=0)
        prob_med = ms.predict(X_val_m, verbose=0).ravel()
        met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Med-Scratch (TOK_ALL)")

        record_result(frac, "Med-Scratch (TOK_ALL)", met_med, None,
                      0.0, t_tr,
                      acc_pol_pre=None, f1_pol_pre=None)

    # -----------------------------------------------------
    # (6) Joint training baseline (POL_train + MED_sub) with same TOK_ALL
    # -----------------------------------------------------
    if INCLUDE_JOINT_TRAIN:
        print("\n[6] Joint-Plain (POL_train + MED_sub, TOK_ALL)")
        tf.keras.backend.clear_session(); gc.collect()
        reset_seeds(SEED)

        X_joint = np.concatenate([X_train_p, X_train_m_sub], axis=0)
        y_joint = np.concatenate([y_train_p, y_train_m_sub], axis=0)

        jm = build_plain_cnn(VOCAB_ALL)
        t_tr = fit_with_time(jm, X_joint, y_joint, X_val_m, y_val_m, verbose=0)

        prob_med = jm.predict(X_val_m, verbose=0).ravel()
        met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Joint-Plain (TOK_ALL)")

        prob_pol = jm.predict(X_val_p, verbose=0).ravel()
        met_pol = evaluate_probs(prob_pol, y_val_p, name="Political Val | Joint-Plain (TOK_ALL)")

        record_result(frac, "Joint-Plain (POL_train + MED_sub, TOK_ALL)", met_med, met_pol,
                      0.0, t_tr,
                      acc_pol_pre=None, f1_pol_pre=None)

    # -----------------------------------------------------
    # (7) Optional: retokenize baseline (not fair, for reference)
    # -----------------------------------------------------
    if INCLUDE_RETOKENIZE_BASELINE:
        print("\n[7] Joint-Plain (retokenize each frac)  [NOT FAIR, for reference]")
        tf.keras.backend.clear_session(); gc.collect()
        reset_seeds(SEED)

        comb_texts = list(X_train_p_texts) + list(X_train_m_sub_texts)
        comb_labels = np.concatenate([y_train_p, y_train_m_sub], axis=0)

        TOK_PM, VOCAB_PM = build_tokenizer(comb_texts, MAX_NUM_WORDS)
        X_joint2 = vectorize(TOK_PM, comb_texts, MAXLEN)
        X_val_m2 = vectorize(TOK_PM, X_val_m_texts, MAXLEN)
        X_val_p2 = vectorize(TOK_PM, X_val_p_texts, MAXLEN)

        jm2 = build_plain_cnn(VOCAB_PM)
        t_tr = fit_with_time(jm2, X_joint2, comb_labels, X_val_m2, y_val_m, verbose=0)

        prob_med = jm2.predict(X_val_m2, verbose=0).ravel()
        met_med = evaluate_probs(prob_med, y_val_m, name="Medical Val | Joint-Plain (retokenize)")

        prob_pol = jm2.predict(X_val_p2, verbose=0).ravel()
        met_pol = evaluate_probs(prob_pol, y_val_p, name="Political Val | Joint-Plain (retokenize)")

        record_result(frac, "Joint-Plain (retokenize each frac)", met_med, met_pol,
                      0.0, t_tr,
                      acc_pol_pre=None, f1_pol_pre=None)

# =========================================================
# 11. Save + summarize + plot
# =========================================================
df = pd.DataFrame(records)
df.to_csv("comparison_fixed.csv", index=False)
print("\nSaved: comparison_fixed.csv")

print("\n==================== Pivot: Medical Acc ====================")
print(df.pivot(index="frac", columns="method", values="acc_med"))

print("\n==================== Pivot: Medical F1 ====================")
print(df.pivot(index="frac", columns="method", values="f1_med"))

print("\n==================== Pivot: Forgetting (Acc_pol_pre - Acc_pol_after) [Transfer only] ====================")
df_forget = df[df["method"].isin(["Transfer-Plain", "Transfer-Plain+EWC", "Transfer-Adapter", "Transfer-Adapter+EWC (OURS)"])]
print(df_forget.pivot(index="frac", columns="method", values="forget_acc"))

# --- plots
x_vals = [f * 100 for f in fractions]

plt.figure(figsize=(14, 10))

# (1) Medical Accuracy
plt.subplot(2, 2, 1)
for m in METHODS:
    tmp = df[df["method"] == m].sort_values("frac")
    plt.plot(x_vals, tmp["acc_med"].values, marker='o', label=m)
plt.title("Medical Val Accuracy vs Medical Train Fraction (TOK_ALL)")
plt.xlabel("Medical Train Fraction (%)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

# (2) Medical F1
plt.subplot(2, 2, 2)
for m in METHODS:
    tmp = df[df["method"] == m].sort_values("frac")
    plt.plot(x_vals, tmp["f1_med"].values, marker='o', label=m)
plt.title("Medical Val F1 vs Medical Train Fraction (TOK_ALL)")
plt.xlabel("Medical Train Fraction (%)")
plt.ylabel("F1")
plt.grid(True)
plt.legend()

# (3) Forgetting (only transfer methods)
plt.subplot(2, 2, 3)
for m in ["Transfer-Plain", "Transfer-Plain+EWC", "Transfer-Adapter", "Transfer-Adapter+EWC (OURS)"]:
    tmp = df[df["method"] == m].sort_values("frac")
    plt.plot(x_vals, tmp["forget_acc"].values, marker='o', label=m)
plt.title("Forgetting on Political Val (Acc_pre - Acc_after)  [Lower is better]")
plt.xlabel("Medical Train Fraction (%)")
plt.ylabel("Forgetting (Accuracy drop)")
plt.grid(True)
plt.legend()

# (4) Finetune time
plt.subplot(2, 2, 4)
for m in METHODS:
    tmp = df[df["method"] == m].sort_values("frac")
    plt.plot(x_vals, tmp["t_finetune"].values, marker='o', label=m)
plt.title("Finetune/Train Time vs Medical Train Fraction")
plt.xlabel("Medical Train Fraction (%)")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Optional: total time including pretrain for transfer methods
plt.figure(figsize=(10, 5))
for m in ["Transfer-Plain", "Transfer-Plain+EWC", "Transfer-Adapter", "Transfer-Adapter+EWC (OURS)"]:
    tmp = df[df["method"] == m].sort_values("frac")
    plt.plot(x_vals, tmp["t_total"].values, marker='o', label=m)
plt.title("Total Time (pretrain + finetune) vs Medical Train Fraction")
plt.xlabel("Medical Train Fraction (%)")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nDone.")
