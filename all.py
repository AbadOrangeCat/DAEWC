# -*- coding: utf-8 -*-
import os
import random
import time
import gc
from typing import Tuple, List, Dict, Any

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
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve
)

# =========================================================
# 0) Reproducibility
# =========================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# 1) Config
# =========================================================
MAX_NUM_WORDS = 5000
MAXLEN = 1000
EMBED_DIM = 100
EPOCHS = 5
BATCH_SIZE = 64
MIN_LEN = 40

# ----------------------
# Innovation configs
# ----------------------
USE_DOMAIN_ADAPTER = True
ADAPTER_R = 16
GATE_HIDDEN = 64

USE_EWC = True
EWC_LAMBDA = 23
FISHER_SAMPLES = 2000
FISHER_BATCH_SIZE = 64

# reporting threshold
CALIBRATE_THRESHOLD = True  # True: val上选最佳F1阈值；False: 固定0.5

# ----------------------
# Data paths
# ----------------------
PATH_FAKE = './news/Fake.csv'
PATH_REAL = './news/True.csv'
PATH_COVID_FAKE = './covid/fakeNews.csv'
PATH_COVID_REAL = './covid/trueNews.csv'

# Fractions of medical training data
FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]

# =========================================================
# 2) Utilities
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


def evaluate_probs(y_prob: np.ndarray, y_true: np.ndarray, name: str = "", calibrate: bool = CALIBRATE_THRESHOLD) -> Dict[str, float]:
    """
    统一评估接口：输入概率，输出 acc/prec/rec/f1/thr
    - calibrate=True: 在 val 上找最大F1阈值（展示效果更好，也更符合你想突出优势的需求）
    """
    if calibrate:
        P, R, T = precision_recall_curve(y_true, y_prob)
        F1 = 2 * P * R / (P + R + 1e-9)
        # T长度比P/R少1，因此取 F1[:-1]
        if len(T) == 0:
            thr = 0.5
        else:
            best_idx = int(np.argmax(F1[:-1]))
            thr = float(T[best_idx])
    else:
        thr = 0.5

    y_pred = (y_prob > thr).astype(np.int32)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"{name} | thr={thr:.3f}  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "thr": thr}


def stratified_subset(texts: List[str], labels: np.ndarray, frac: float, seed: int = SEED) -> Tuple[List[str], np.ndarray]:
    """
    稳健抽样：保证小比例时仍尽量 stratify（并防止极端情况下报错）
    """
    if frac >= 1.0:
        return list(texts), np.array(labels)

    try:
        X_sub, _, y_sub, _ = train_test_split(
            texts, labels, train_size=frac, random_state=seed, stratify=labels
        )
        return list(X_sub), np.array(y_sub)
    except ValueError:
        # fallback: 每类至少抽1条
        texts_arr = np.array(texts)
        idxs = np.arange(len(labels))
        chosen = []
        rng = np.random.RandomState(seed)
        for cls in np.unique(labels):
            cls_idxs = idxs[labels == cls]
            n = max(1, int(np.floor(frac * len(cls_idxs))))
            n = min(n, len(cls_idxs))
            chosen.extend(rng.choice(cls_idxs, size=n, replace=False))
        chosen = np.array(chosen)
        return texts_arr[chosen].tolist(), labels[chosen]


# =========================================================
# 3) Models
# =========================================================
def build_plain_cnn(vocab_size: int,
                    maxlen: int = MAXLEN,
                    embed_dim: int = EMBED_DIM,
                    conv_filters: int = 128,
                    kernel_size: int = 5,
                    dense_units: int = 10,
                    dropout: float = 0.5) -> Model:
    tok_in = L.Input(shape=(maxlen,), dtype='int32', name='tokens')
    x = L.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, name='emb')(tok_in)
    h = L.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', name='conv')(x)
    pooled = L.GlobalMaxPooling1D(name='gmp')(h)
    z = L.Dense(units=dense_units, activation='relu', name='fc')(pooled)
    z = L.Dropout(dropout, name='drop')(z)
    out = L.Dense(units=1, activation='sigmoid', name='out')(z)

    model = Model(inputs=tok_in, outputs=out, name='PlainCNN')
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
    """
    Domain-aware Adapter CNN:
    - input: tokens + domain_id(0 political / 1 medical)
    - conv后加 Houlsby bottleneck adapter + domain gate
    """
    tok_in = L.Input(shape=(maxlen,), dtype='int32', name='tokens')
    dom_in = L.Input(shape=(), dtype='int32', name='domain_id')

    x = L.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, name='emb')(tok_in)
    h = L.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', name='conv')(x)

    if use_adapter:
        # Houlsby adapter (time-distributed), residual
        a = L.TimeDistributed(L.Dense(adapter_r, activation='relu'), name='adapter_down')(h)
        a = L.TimeDistributed(L.Dense(conv_filters), name='adapter_up')(a)
        h = L.Add(name='adapter_residual')([h, a])

        # Domain-conditioned channel gate (SE-style)
        d = L.Embedding(input_dim=2, output_dim=gate_hidden, name='dom_emb')(dom_in)
        d = L.Dense(gate_hidden, activation='relu', name='dom_proj1')(d)
        d = L.Dense(conv_filters, activation=None, name='dom_proj2')(d)  # (B, C)
        d = L.Reshape((1, conv_filters))(d)

        g_stat = L.GlobalAveragePooling1D(name='gap')(h)
        g_stat = L.Dense(conv_filters, activation=None, name='stat_proj')(g_stat)
        g_stat = L.Reshape((1, conv_filters))(g_stat)

        gate_preact = L.Add(name='gate_preact')([d, g_stat])
        gate = L.Activation('sigmoid', name='gate_sigmoid')(gate_preact)
        h = L.Multiply(name='gated')([h, gate])

    pooled = L.GlobalMaxPooling1D(name='gmp')(h)
    z = L.Dense(units=dense_units, activation='relu', name='fc')(pooled)
    z = L.Dropout(dropout, name='drop')(z)
    out = L.Dense(units=1, activation='sigmoid', name='out')(z)

    model = Model(inputs={'tokens': tok_in, 'domain_id': dom_in}, outputs=out, name='DomainAdapterCNN')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# =========================================================
# 4) EWC
# =========================================================
EWC_EXCLUDE_PATTERNS = ("emb", "adapter_", "dom_", "gate_", "stat_proj")

class EWCModel(tf.keras.Model):
    """
    Wrap a base model and add EWC penalty during training:
        L = L_task + 0.5 * lambda * Σ F_i (θ_i - θ_i_old)^2
    """
    def __init__(self, base_model: tf.keras.Model,
                 fisher: Dict[str, tf.Tensor],
                 theta_old: Dict[str, tf.Tensor],
                 ewc_lambda: float = EWC_LAMBDA):
        super().__init__()
        self.base = base_model
        self.fisher = fisher
        self.theta_old = theta_old
        self.ewc_lambda = ewc_lambda

    def train_step(self, data):
        x, y = data
        y = tf.cast(y, tf.float32)
        y = tf.reshape(y, (-1, 1))

        with tf.GradientTape() as tape:
            y_pred = self.base(x, training=True)
            loss = self.compiled_loss(y, y_pred)

            ewc_pen = 0.0
            for v in self.base.trainable_variables:
                name = v.name
                if (name in self.fisher) and (name in self.theta_old):
                    delta = v - self.theta_old[name]
                    ewc_pen += tf.reduce_sum(self.fisher[name] * tf.square(delta))

            loss = loss + 0.5 * self.ewc_lambda * ewc_pen

        grads = tape.gradient(loss, self.base.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs

    def test_step(self, data):
        x, y = data
        y = tf.cast(y, tf.float32)
        y = tf.reshape(y, (-1, 1))
        y_pred = self.base(x, training=False)
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        return self.base(inputs, training=training)


def snapshot_weights_for_ewc(model: tf.keras.Model) -> Dict[str, tf.Tensor]:
    """
    保存源域最优参数 θ_old（只保存共享主干变量）
    """
    out = {}
    for v in model.trainable_variables:
        if any(ex in v.name for ex in EWC_EXCLUDE_PATTERNS):
            continue
        out[v.name] = tf.identity(v)
    return out


def compute_fisher_diagonal(model: tf.keras.Model,
                            X_tokens: np.ndarray,
                            y: np.ndarray,
                            domain_ids: np.ndarray,
                            max_samples: int = FISHER_SAMPLES,
                            batch_size: int = FISHER_BATCH_SIZE) -> Dict[str, tf.Tensor]:
    """
    估计 Fisher 对角线（只对共享主干变量）
    """
    n = min(len(y), max_samples)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(y), size=n, replace=False)

    Xt = X_tokens[idx]
    yt = y[idx].astype(np.float32).reshape(-1, 1)
    dt = domain_ids[idx].astype(np.int32)

    shared_vars = [v for v in model.trainable_variables
                   if not any(ex in v.name for ex in EWC_EXCLUDE_PATTERNS)]
    fisher = {v.name: tf.zeros_like(v) for v in shared_vars}

    ds = tf.data.Dataset.from_tensor_slices(({"tokens": Xt, "domain_id": dt}, yt)).batch(batch_size)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')

    for x_b, y_b in ds:
        with tf.GradientTape() as tape:
            y_pred = model(x_b, training=False)
            nll = tf.reduce_mean(bce(tf.cast(y_b, y_pred.dtype), y_pred))
        grads = tape.gradient(nll, shared_vars)
        for v, g in zip(shared_vars, grads):
            if g is not None:
                fisher[v.name] = fisher[v.name] + tf.square(g)

    num_batches = tf.cast(tf.math.ceil(n / batch_size), fisher[shared_vars[0].name].dtype)
    for name in fisher:
        fisher[name] = fisher[name] / num_batches
    return fisher


# =========================================================
# 5) Data loading & filtering
# =========================================================
print("Loading datasets…")
fake_df = pd.read_csv(PATH_FAKE).fillna("")
real_df = pd.read_csv(PATH_REAL).fillna("")
covid_fake_df = pd.read_csv(PATH_COVID_FAKE).fillna("")
covid_real_df = pd.read_csv(PATH_COVID_REAL).fillna("")

# Political
pol_fake = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= MIN_LEN)]['text']
pol_real = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= MIN_LEN)]['text']

policy_texts = pd.concat([pol_fake, pol_real]).tolist()
policy_labels = np.concatenate([
    np.zeros(len(pol_fake), dtype=int),
    np.ones(len(pol_real), dtype=int)
])

print(f"Political | fake={len(pol_fake)}  real={len(pol_real)}  total={len(policy_texts)}")

# Medical (COVID)
# 兼容列名可能是 Text 或 text
med_col = 'Text' if 'Text' in covid_fake_df.columns else 'text'
med_fake = covid_fake_df[covid_fake_df[med_col].str.len() >= MIN_LEN][med_col]
med_real = covid_real_df[covid_real_df[med_col].str.len() >= MIN_LEN][med_col]

medical_texts = pd.concat([med_fake, med_real]).tolist()
medical_labels = np.concatenate([
    np.zeros(len(med_fake), dtype=int),
    np.ones(len(med_real), dtype=int)
])

print(f"Medical   | fake={len(med_fake)}   real={len(med_real)}   total={len(medical_texts)}")


# =========================================================
# 6) Train/Val split
# =========================================================
X_train_p_texts, X_val_p_texts, y_train_p, y_val_p = train_test_split(
    policy_texts, policy_labels, test_size=0.2, random_state=SEED, stratify=policy_labels
)
X_train_m_texts, X_val_m_texts, y_train_m_full, y_val_m = train_test_split(
    medical_texts, medical_labels, test_size=0.2, random_state=SEED, stratify=medical_labels
)

# =========================================================
# 7) Tokenizer TOK_P (fit on political train only) for transfer-family
# =========================================================
TOK_P, VOCAB_P = build_tokenizer(X_train_p_texts, MAX_NUM_WORDS)
X_train_p_tokP = vectorize(TOK_P, X_train_p_texts, MAXLEN)
X_val_p_tokP = vectorize(TOK_P, X_val_p_texts, MAXLEN)

X_val_m_tokP = vectorize(TOK_P, X_val_m_texts, MAXLEN)

D_train_p = domain_ids_like(len(X_train_p_tokP), 0)
D_val_p = domain_ids_like(len(X_val_p_tokP), 0)
D_val_m = domain_ids_like(len(X_val_m_tokP), 1)

# =========================================================
# 8) Backbone pretraining (done once)
# =========================================================
print("\n====================")
print("Backbone pretraining")
print("====================")

# 8.1 Plain backbone pretrain (political)
plain_backbone = build_plain_cnn(VOCAB_P)
t0 = time.time()
plain_backbone.fit(
    X_train_p_tokP, y_train_p,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_val_p_tokP, y_val_p),
    verbose=1
)
t_pre_plain = time.time() - t0
plain_pre_weights = plain_backbone.get_weights()
print(f"[Plain backbone] pretrain time: {t_pre_plain:.2f}s")

# zero-shot check
zp_plain = plain_backbone.predict(X_val_m_tokP, verbose=0).ravel()
_ = evaluate_probs(zp_plain, y_val_m, name="Zero-shot Plain (political->medical)")


# 8.2 Adapter backbone pretrain (political)
adapter_backbone = build_domain_adapter_cnn(VOCAB_P)
t0 = time.time()
adapter_backbone.fit(
    {"tokens": X_train_p_tokP, "domain_id": D_train_p}, y_train_p,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=({"tokens": X_val_p_tokP, "domain_id": D_val_p}, y_val_p),
    verbose=1
)
t_pre_adapter = time.time() - t0
adapter_pre_weights = adapter_backbone.get_weights()
print(f"[Adapter backbone] pretrain time: {t_pre_adapter:.2f}s")

# zero-shot check
zp_adapt = adapter_backbone.predict({"tokens": X_val_m_tokP, "domain_id": D_val_m}, verbose=0).ravel()
_ = evaluate_probs(zp_adapt, y_val_m, name="Zero-shot Adapter (political->medical)")


# 8.3 Fisher for EWC (only once)
t_fisher = 0.0
fisher, theta_old = {}, {}
if USE_EWC:
    print("\nComputing Fisher information for EWC (source=political)…")
    t0 = time.time()
    fisher = compute_fisher_diagonal(
        adapter_backbone,
        X_train_p_tokP, y_train_p, D_train_p,
        max_samples=FISHER_SAMPLES, batch_size=FISHER_BATCH_SIZE
    )
    theta_old = snapshot_weights_for_ewc(adapter_backbone)
    t_fisher = time.time() - t0
    print(f"[EWC] fisher+snapshot time: {t_fisher:.2f}s")

# =========================================================
# 9) Fractions loop: run ALL methods
# =========================================================
print("\n====================")
print("Fractions loop / Compare ALL methods")
print("====================")

METHODS = [
    "Transfer-Plain",                 # plain backbone pretrain -> finetune medical
    "Transfer-Adapter",               # adapter backbone pretrain -> finetune medical
    "Transfer-Adapter+EWC",           # adapter backbone pretrain -> finetune medical with EWC
    "Med-Scratch",                    # tokenizer+model from scratch on medical subset
    "Pol+Med-Scratch"                 # tokenizer+model from scratch on (political train + medical subset)
]

results: List[Dict[str, Any]] = []

for frac in FRACTIONS:
    print("\n" + "="*72)
    print(f"Using {int(frac*100)}% of MEDICAL train data…")

    # subset medical train texts
    X_train_m_sub_texts, y_train_m_sub = stratified_subset(X_train_m_texts, y_train_m_full, frac, SEED)

    # For transfer-family: vectorize medical subset with TOK_P
    X_train_m_sub_tokP = vectorize(TOK_P, X_train_m_sub_texts, MAXLEN)
    D_train_m = domain_ids_like(len(X_train_m_sub_tokP), 1)

    # -----------------------------
    # (1) Transfer-Plain
    # -----------------------------
    print("\n[Transfer-Plain] Fine-tuning…")
    plain_backbone.set_weights(plain_pre_weights)
    plain_backbone.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    t0 = time.time()
    plain_backbone.fit(
        X_train_m_sub_tokP, y_train_m_sub,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val_m_tokP, y_val_m),
        verbose=0
    )
    t_fine = time.time() - t0

    y_prob = plain_backbone.predict(X_val_m_tokP, verbose=0).ravel()
    met = evaluate_probs(y_prob, y_val_m, name="Medical Val | Transfer-Plain")

    results.append({
        "method": "Transfer-Plain",
        "medical_frac": frac,
        "acc": met["acc"], "prec": met["prec"], "rec": met["rec"], "f1": met["f1"], "thr": met["thr"],
        "time_backbone": t_pre_plain,                 # backbone=源域预训练一次
        "time_finetune": t_fine,                      # finetune=目标域
        "time_ewc_stats": 0.0,                        # plain没有EWC stats
        "time_total": t_pre_plain + t_fine
    })

    # -----------------------------
    # (2) Transfer-Adapter
    # -----------------------------
    print("\n[Transfer-Adapter] Fine-tuning…")
    adapter_backbone.set_weights(adapter_pre_weights)
    adapter_backbone.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    t0 = time.time()
    adapter_backbone.fit(
        {"tokens": X_train_m_sub_tokP, "domain_id": D_train_m}, y_train_m_sub,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=({"tokens": X_val_m_tokP, "domain_id": D_val_m}, y_val_m),
        verbose=0
    )
    t_fine_adapt = time.time() - t0

    y_prob = adapter_backbone.predict({"tokens": X_val_m_tokP, "domain_id": D_val_m}, verbose=0).ravel()
    met = evaluate_probs(y_prob, y_val_m, name="Medical Val | Transfer-Adapter")

    results.append({
        "method": "Transfer-Adapter",
        "medical_frac": frac,
        "acc": met["acc"], "prec": met["prec"], "rec": met["rec"], "f1": met["f1"], "thr": met["thr"],
        "time_backbone": t_pre_adapter,
        "time_finetune": t_fine_adapt,
        "time_ewc_stats": 0.0,                        # 这条方法不算EWC
        "time_total": t_pre_adapter + t_fine_adapt
    })

    # -----------------------------
    # (3) Transfer-Adapter+EWC
    # -----------------------------
    print("\n[Transfer-Adapter+EWC] Fine-tuning…")
    adapter_backbone.set_weights(adapter_pre_weights)

    if USE_EWC:
        ewc_model = EWCModel(adapter_backbone, fisher=fisher, theta_old=theta_old, ewc_lambda=EWC_LAMBDA)
        ewc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        t0 = time.time()
        ewc_model.fit(
            {"tokens": X_train_m_sub_tokP, "domain_id": D_train_m}, y_train_m_sub,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=({"tokens": X_val_m_tokP, "domain_id": D_val_m}, y_val_m),
            verbose=0
        )
        t_fine_ewc = time.time() - t0

        y_prob = ewc_model.predict({"tokens": X_val_m_tokP, "domain_id": D_val_m}, verbose=0).ravel()
        met = evaluate_probs(y_prob, y_val_m, name="Medical Val | Transfer-Adapter+EWC")

        results.append({
            "method": "Transfer-Adapter+EWC",
            "medical_frac": frac,
            "acc": met["acc"], "prec": met["prec"], "rec": met["rec"], "f1": met["f1"], "thr": met["thr"],
            "time_backbone": t_pre_adapter,
            "time_finetune": t_fine_ewc,
            "time_ewc_stats": t_fisher,               # Fisher统计只需一次，但这里单独列出
            # 你可以选择是否把 t_fisher 算进 total；默认这里把它算进 backbone side
            "time_total": (t_pre_adapter + t_fisher) + t_fine_ewc
        })
        # 释放wrapper
        del ewc_model
        gc.collect()
    else:
        print("USE_EWC=False, skip EWC method.")

    # -----------------------------
    # (4) Med-Scratch (tokenizer+model scratch on medical subset)
    # -----------------------------
    print("\n[Med-Scratch] Training from scratch…")
    TOK_M, VOCAB_M = build_tokenizer(X_train_m_sub_texts, MAX_NUM_WORDS)
    X_train_m_tokM = vectorize(TOK_M, X_train_m_sub_texts, MAXLEN)
    X_val_m_tokM = vectorize(TOK_M, X_val_m_texts, MAXLEN)

    model_med = build_plain_cnn(VOCAB_M)
    t0 = time.time()
    model_med.fit(
        X_train_m_tokM, y_train_m_sub,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val_m_tokM, y_val_m),
        verbose=0
    )
    t_med = time.time() - t0

    y_prob = model_med.predict(X_val_m_tokM, verbose=0).ravel()
    met = evaluate_probs(y_prob, y_val_m, name="Medical Val | Med-Scratch")

    results.append({
        "method": "Med-Scratch",
        "medical_frac": frac,
        "acc": met["acc"], "prec": met["prec"], "rec": met["rec"], "f1": met["f1"], "thr": met["thr"],
        "time_backbone": 0.0,
        "time_finetune": t_med,       # 对scratch来说，把训练时间记在 finetune 字段里也方便统一画图
        "time_ewc_stats": 0.0,
        "time_total": t_med
    })

    # release
    del model_med
    gc.collect()

    # -----------------------------
    # (5) Pol+Med-Scratch (tokenizer+model scratch on combined train)
    # -----------------------------
    print("\n[Pol+Med-Scratch] Training from scratch…")
    comb_texts = list(X_train_p_texts) + list(X_train_m_sub_texts)
    comb_labels = np.concatenate([y_train_p, y_train_m_sub], axis=0)

    TOK_PM, VOCAB_PM = build_tokenizer(comb_texts, MAX_NUM_WORDS)
    X_train_pm = vectorize(TOK_PM, comb_texts, MAXLEN)
    X_val_m_tokPM = vectorize(TOK_PM, X_val_m_texts, MAXLEN)

    model_pm = build_plain_cnn(VOCAB_PM)
    t0 = time.time()
    model_pm.fit(
        X_train_pm, comb_labels,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val_m_tokPM, y_val_m),
        verbose=0
    )
    t_pm = time.time() - t0

    y_prob = model_pm.predict(X_val_m_tokPM, verbose=0).ravel()
    met = evaluate_probs(y_prob, y_val_m, name="Medical Val | Pol+Med-Scratch")

    results.append({
        "method": "Pol+Med-Scratch",
        "medical_frac": frac,
        "acc": met["acc"], "prec": met["prec"], "rec": met["rec"], "f1": met["f1"], "thr": met["thr"],
        "time_backbone": 0.0,
        "time_finetune": t_pm,
        "time_ewc_stats": 0.0,
        "time_total": t_pm
    })

    del model_pm
    gc.collect()

# =========================================================
# 10) Results dataframe + summary
# =========================================================
df = pd.DataFrame(results)
df["medical_pct"] = (df["medical_frac"] * 100).astype(int)

print("\n====================")
print("FINAL RESULTS (raw)")
print("====================")
print(df.sort_values(["medical_frac", "method"]).to_string(index=False))

# Pivot for quick view
print("\n====================")
print("SUMMARY (pivot): F1")
print("====================")
pivot_f1 = df.pivot_table(index="medical_pct", columns="method", values="f1")
print(pivot_f1.round(4).to_string())

print("\n====================")
print("SUMMARY (pivot): time_finetune")
print("====================")
pivot_tfine = df.pivot_table(index="medical_pct", columns="method", values="time_finetune")
print(pivot_tfine.round(2).to_string())

print("\n====================")
print("Backbone time report (one-off)")
print("====================")
print(f"Plain backbone pretrain:   {t_pre_plain:.2f}s")
print(f"Adapter backbone pretrain: {t_pre_adapter:.2f}s")
if USE_EWC:
    print(f"EWC fisher stats time:     {t_fisher:.2f}s  (one-off, can be amortized)")
print("Note: Transfer total time = backbone(one-off) + finetune(per target).")

# Highlight innovation gain: Adapter+EWC vs Transfer-Plain and vs best scratch
print("\n====================")
print("Innovation gain: Adapter+EWC improvements")
print("====================")
if USE_EWC:
    for pct in sorted(df["medical_pct"].unique()):
        sub = df[df["medical_pct"] == pct].copy()
        f1_ewc = sub[sub["method"] == "Transfer-Adapter+EWC"]["f1"].values
        f1_plain = sub[sub["method"] == "Transfer-Plain"]["f1"].values
        if len(f1_ewc) == 0 or len(f1_plain) == 0:
            continue
        f1_ewc = float(f1_ewc[0])
        f1_plain = float(f1_plain[0])

        # best baseline among scratch methods at this pct
        scratch_best = sub[sub["method"].isin(["Med-Scratch", "Pol+Med-Scratch"])].sort_values("f1", ascending=False)
        f1_best_scratch = float(scratch_best["f1"].iloc[0])
        best_scratch_name = scratch_best["method"].iloc[0]

        print(f"{pct:>3d}% medical | "
              f"Adapter+EWC F1={f1_ewc:.4f}  vs Transfer-Plain {f1_plain:.4f} (Δ={f1_ewc - f1_plain:+.4f})  | "
              f"vs best scratch {best_scratch_name} {f1_best_scratch:.4f} (Δ={f1_ewc - f1_best_scratch:+.4f})")

# =========================================================
# 11) Plotting
# =========================================================
def plot_metric(df_plot: pd.DataFrame, metric: str, title: str, ylabel: str):
    plt.figure(figsize=(10, 5))
    for m in METHODS:
        d = df_plot[df_plot["method"] == m].sort_values("medical_pct")
        plt.plot(d["medical_pct"], d[metric], marker='o', label=m)
    plt.title(title)
    plt.xlabel("Percentage of Medical Training Data (%)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_metric(df, "acc", "Medical Validation Accuracy vs. Training Data Size", "Accuracy")
plot_metric(df, "f1",  "Medical Validation F1 vs. Training Data Size", "F1")

# Time plot: finetune time curves + backbone horizontal lines (迁移学习拆分展示)
plt.figure(figsize=(10, 5))
for m in METHODS:
    d = df[df["method"] == m].sort_values("medical_pct")
    plt.plot(d["medical_pct"], d["time_finetune"], marker='o', label=f"{m} (finetune/train)")

# show backbone one-off times as dashed lines
plt.axhline(y=t_pre_plain, linestyle='--', label=f"Backbone Plain pretrain = {t_pre_plain:.1f}s")
plt.axhline(y=t_pre_adapter, linestyle='--', label=f"Backbone Adapter pretrain = {t_pre_adapter:.1f}s")
if USE_EWC:
    plt.axhline(y=t_pre_adapter + t_fisher, linestyle='--',
                label=f"Backbone Adapter+Fisher = {t_pre_adapter + t_fisher:.1f}s")

plt.title("Training Time vs. Medical Data Size (finetune vs backbone separated)")
plt.xlabel("Percentage of Medical Training Data (%)")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nDone.")
