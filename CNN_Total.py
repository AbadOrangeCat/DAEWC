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


# ======================
# 0. Reproducibility
# ======================
SEED = 42
def reset_seeds(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

reset_seeds(SEED)

# ======================
# 1. Config
# ======================
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
USE_EWC_PLAIN = True
USE_EWC_ADAPTER = True
EWC_LAMBDA = 23
FISHER_SAMPLES = 2000
FISHER_BATCH_SIZE = 64

# Reporting
# True = 在验证集上挑阈值最大化F1（会偏乐观；论文里建议用独立校准集）
CALIBRATE_THRESHOLD = False

# Fractions
fractions = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]

# Paths
PATH_FAKE = './news/Fake.csv'
PATH_REAL = './news/True.csv'
PATH_COVID_FAKE = './covid/fakeNews.csv'
PATH_COVID_REAL = './covid/trueNews.csv'


# ======================
# 2. Utilities
# ======================
def build_tokenizer(train_texts: List[str], num_words: int = MAX_NUM_WORDS) -> Tuple[Tokenizer, int]:
    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tok.fit_on_texts(train_texts)
    vocab_size = min(num_words, len(tok.word_index) + 1)
    return tok, vocab_size

def vectorize(tok: Tokenizer, texts: List[str], maxlen: int = MAXLEN) -> np.ndarray:
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen)

def domain_ids_like(length: int, domain: int) -> np.ndarray:
    return np.full((length,), domain, dtype=np.int32)

def stratified_subset(texts: List[str], labels: np.ndarray, frac: float, seed: int = SEED) -> Tuple[List[str], np.ndarray]:
    """Stratified subset. Robust for tiny frac edge cases."""
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
                  X_train, y_train,
                  X_val, y_val,
                  epochs: int = EPOCHS,
                  batch_size: int = BATCH_SIZE,
                  verbose: int = 0) -> float:
    start = time.time()
    model.fit(X_train, y_train,
              epochs=epochs, batch_size=batch_size,
              validation_data=(X_val, y_val),
              verbose=verbose)
    return time.time() - start


# ======================
# 3. Models
# ======================
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
    tok_in = L.Input(shape=(maxlen,), dtype='int32', name='tokens')
    dom_in = L.Input(shape=(), dtype='int32', name='domain_id')  # 0=political, 1=medical

    x = L.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen, name='emb')(tok_in)
    h = L.Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', name='conv')(x)

    if use_adapter:
        # Houlsby adapter (time-distributed bottleneck) + residual
        a = L.TimeDistributed(L.Dense(adapter_r, activation='relu'), name='adapter_down')(h)
        a = L.TimeDistributed(L.Dense(conv_filters), name='adapter_up')(a)
        h = L.Add(name='adapter_residual')([h, a])

        # Domain-conditioned channel gate (SE-like)
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
    z = L.Dense(units=dense_units, activation='relu', name='fc')(pooled)
    z = L.Dropout(dropout, name='drop')(z)
    out = L.Dense(units=1, activation='sigmoid', name='out')(z)

    model = Model(inputs={'tokens': tok_in, 'domain_id': dom_in}, outputs=out, name='DomainAdapterCNN')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ======================
# 4. EWC
# ======================
class EWCModel(tf.keras.Model):
    """Wrap a base model and add EWC penalty in train_step."""
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

    def test_step(self, data):
        x, y = data
        y = tf.cast(y, tf.float32)
        y = tf.reshape(y, (-1, 1))
        y_pred = self.base(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.base.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

def snapshot_weights(model: tf.keras.Model, exclude_patterns: Tuple[str, ...]) -> Dict[str, tf.Tensor]:
    snap = {}
    for v in model.trainable_variables:
        if any(p in v.name for p in exclude_patterns):
            continue
        snap[v.name] = tf.identity(v)
    return snap

def compute_fisher_diagonal(model: tf.keras.Model,
                            X_tokens: np.ndarray,
                            y: np.ndarray,
                            domain_ids: Optional[np.ndarray] = None,
                            max_samples: int = FISHER_SAMPLES,
                            batch_size: int = FISHER_BATCH_SIZE,
                            exclude_patterns: Tuple[str, ...] = ("emb",)) -> Dict[str, tf.Tensor]:
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

    for x_b, y_b in ds:
        with tf.GradientTape() as tape:
            y_pred = model(x_b, training=False)
            nll = tf.reduce_mean(bce(tf.cast(y_b, y_pred.dtype), y_pred))
        grads = tape.gradient(nll, shared_vars)
        for v, g in zip(shared_vars, grads):
            if g is not None:
                fisher[v.name] = fisher[v.name] + tf.square(g)

    num_batches = int(np.ceil(n / batch_size))
    for name in fisher:
        fisher[name] = fisher[name] / tf.cast(num_batches, fisher[name].dtype)
    return fisher


# ======================
# 5. Load & filter data
# ======================
print("Loading datasets…")
fake_df = pd.read_csv(PATH_FAKE)
real_df = pd.read_csv(PATH_REAL)
covid_fake_df = pd.read_csv(PATH_COVID_FAKE)
covid_real_df = pd.read_csv(PATH_COVID_REAL)

# Political
pol_fake = fake_df[(fake_df['subject'] == 'politics') & (fake_df['text'].str.len() >= 40)]['text']
pol_real = real_df[(real_df['subject'] == 'politicsNews') & (real_df['text'].str.len() >= 40)]['text']
policy_texts = pd.concat([pol_fake, pol_real]).tolist()
policy_labels = np.concatenate([np.zeros(len(pol_fake), dtype=int),
                                np.ones(len(pol_real), dtype=int)])
print(f"Political | fake={len(pol_fake)}  real={len(pol_real)}  total={len(policy_texts)}")

# Medical (COVID) — robust column name
def pick_text_col(df: pd.DataFrame) -> str:
    for c in ["Text", "text", "content", "Content"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find text column in covid dataset; tried Text/text/content/Content")

covid_fake_col = pick_text_col(covid_fake_df)
covid_real_col = pick_text_col(covid_real_df)

med_fake = covid_fake_df[covid_fake_df[covid_fake_col].astype(str).str.len() >= 40][covid_fake_col].astype(str)
med_real = covid_real_df[covid_real_df[covid_real_col].astype(str).str.len() >= 40][covid_real_col].astype(str)
medical_texts = pd.concat([med_fake, med_real]).tolist()
medical_labels = np.concatenate([np.zeros(len(med_fake), dtype=int),
                                 np.ones(len(med_real), dtype=int)])
print(f"Medical   | fake={len(med_fake)}   real={len(med_real)}   total={len(medical_texts)}")


# ======================
# 6. Train/Val splits on TEXTS (stratified)
# ======================
X_train_p_texts, X_val_p_texts, y_train_p, y_val_p = train_test_split(
    policy_texts, policy_labels, test_size=0.2, random_state=SEED, stratify=policy_labels
)
X_train_m_texts, X_val_m_texts, y_train_m_full, y_val_m = train_test_split(
    medical_texts, medical_labels, test_size=0.2, random_state=SEED, stratify=medical_labels
)

print(f"\nSplit sizes:")
print(f"Political train={len(X_train_p_texts)}  val={len(X_val_p_texts)}")
print(f"Medical   train={len(X_train_m_texts)}  val={len(X_val_m_texts)}")


# ======================
# 7. Tokenizer: fit on POLITICAL TRAIN only (TOK_P)
# ======================
TOK_P, VOCAB_P = build_tokenizer(X_train_p_texts, MAX_NUM_WORDS)
X_train_p_tokP = vectorize(TOK_P, X_train_p_texts, MAXLEN)
X_val_p_tokP = vectorize(TOK_P, X_val_p_texts, MAXLEN)
X_val_m_tokP = vectorize(TOK_P, X_val_m_texts, MAXLEN)

D_train_p = domain_ids_like(len(X_train_p_tokP), 0)
D_val_p = domain_ids_like(len(X_val_p_tokP), 0)
D_val_m = domain_ids_like(len(X_val_m_tokP), 1)


# ======================
# 8. Pretrain (A) Plain CNN on political
# ======================
print("\n[Pretrain A] Plain CNN on POLITICAL…")
tf.keras.backend.clear_session()
reset_seeds(SEED)

plain_pre = build_plain_cnn(VOCAB_P)
t0 = time.time()
plain_pre.fit(X_train_p_tokP, y_train_p, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val_p_tokP, y_val_p), verbose=1)
plain_pretrain_time = time.time() - t0
print(f"Plain pretrain time: {plain_pretrain_time:.2f}s")

# Zero-shot on medical
zp_plain = plain_pre.predict(X_val_m_tokP, verbose=0).ravel()
_ = evaluate_probs(zp_plain, y_val_m, name="Zero-shot Plain (political->medical)")

plain_pre_weights = plain_pre.get_weights()

plain_fisher, plain_theta_old = {}, {}
if USE_EWC_PLAIN:
    print("Computing Fisher (Plain)…")
    # 如果你发现 Plain+EWC 太“抑制迁移”，可以试试 ("emb","out") 放开分类头
    PLAIN_EWC_EXCLUDE = ("emb",)
    plain_fisher = compute_fisher_diagonal(
        plain_pre, X_train_p_tokP, y_train_p,
        domain_ids=None,
        max_samples=FISHER_SAMPLES,
        batch_size=FISHER_BATCH_SIZE,
        exclude_patterns=PLAIN_EWC_EXCLUDE
    )
    plain_theta_old = snapshot_weights(plain_pre, exclude_patterns=PLAIN_EWC_EXCLUDE)


# ======================
# 9. Pretrain (B) Adapter CNN on political (domain=0)
# ======================
print("\n[Pretrain B] Adapter CNN on POLITICAL (domain=0)…")
tf.keras.backend.clear_session()
reset_seeds(SEED)

adapter_pre = build_domain_adapter_cnn(VOCAB_P)
t0 = time.time()
adapter_pre.fit({"tokens": X_train_p_tokP, "domain_id": D_train_p}, y_train_p,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=({"tokens": X_val_p_tokP, "domain_id": D_val_p}, y_val_p),
                verbose=1)
adapter_pretrain_time = time.time() - t0
print(f"Adapter pretrain time: {adapter_pretrain_time:.2f}s")

# Zero-shot on medical (domain=1)
zp_adp = adapter_pre.predict({"tokens": X_val_m_tokP, "domain_id": D_val_m}, verbose=0).ravel()
_ = evaluate_probs(zp_adp, y_val_m, name="Zero-shot Adapter (political->medical)")

adapter_pre_weights = adapter_pre.get_weights()

adapter_fisher, adapter_theta_old = {}, {}
if USE_EWC_ADAPTER:
    print("Computing Fisher (Adapter shared trunk)…")
    ADAPTER_EWC_EXCLUDE = ("emb", "adapter_", "dom_", "gate_", "stat_proj")
    adapter_fisher = compute_fisher_diagonal(
        adapter_pre, X_train_p_tokP, y_train_p,
        domain_ids=D_train_p,
        max_samples=FISHER_SAMPLES,
        batch_size=FISHER_BATCH_SIZE,
        exclude_patterns=ADAPTER_EWC_EXCLUDE
    )
    adapter_theta_old = snapshot_weights(adapter_pre, exclude_patterns=ADAPTER_EWC_EXCLUDE)


# ======================
# 10. Fractions loop (ALL methods together)
# ======================
METHODS = [
    "Transfer-Plain",
    "Transfer-Plain+EWC",
    "Transfer-Adapter",
    "Transfer-Adapter+EWC (OURS)",
    "Med-Scratch",
    "Pol+Med-Scratch (retokenize)",
    "Pol+Med-Scratch (fixed TOK_P)"
]

records = []

for frac in fractions:
    print("\n" + "=" * 90)
    print(f"Using {int(frac * 100)}% of MEDICAL train data…")

    # (a) subset medical train texts
    X_train_m_sub_texts, y_train_m_sub = stratified_subset(X_train_m_texts, y_train_m_full, frac, seed=SEED)

    # (b) vectorize medical subset with TOK_P (for all transfer + fixed vocab baseline)
    X_train_m_sub_tokP = vectorize(TOK_P, X_train_m_sub_texts, MAXLEN)
    D_train_m = domain_ids_like(len(X_train_m_sub_tokP), 1)

    # =====================================================
    # (1) Transfer Plain
    # =====================================================
    print("\n[1] Transfer-Plain")
    tf.keras.backend.clear_session(); gc.collect()
    reset_seeds(SEED)

    m = build_plain_cnn(VOCAB_P)
    m.set_weights(plain_pre_weights)
    t_fine = fit_with_time(m, X_train_m_sub_tokP, y_train_m_sub, X_val_m_tokP, y_val_m, verbose=0)
    y_prob_med = m.predict(X_val_m_tokP, verbose=0).ravel()
    met_med = evaluate_probs(y_prob_med, y_val_m, name="Medical Val | Transfer-Plain")

    y_prob_pol = m.predict(X_val_p_tokP, verbose=0).ravel()
    met_pol = evaluate_probs(y_prob_pol, y_val_p, name="Political Val(after FT) | Transfer-Plain")

    records.append({
        "frac": frac, "method": "Transfer-Plain",
        "acc_med": met_med["acc"], "f1_med": met_med["f1"],
        "acc_pol": met_pol["acc"], "f1_pol": met_pol["f1"],
        "t_pretrain": plain_pretrain_time, "t_train": t_fine, "t_total": plain_pretrain_time + t_fine
    })

    # =====================================================
    # (2) Transfer Plain + EWC
    # =====================================================
    if USE_EWC_PLAIN:
        print("\n[2] Transfer-Plain+EWC")
        tf.keras.backend.clear_session(); gc.collect()
        reset_seeds(SEED)

        base = build_plain_cnn(VOCAB_P)
        base.set_weights(plain_pre_weights)
        ewc = EWCModel(base, fisher=plain_fisher, theta_old=plain_theta_old, ewc_lambda=EWC_LAMBDA)
        ewc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        t_fine = fit_with_time(ewc, X_train_m_sub_tokP, y_train_m_sub, X_val_m_tokP, y_val_m, verbose=0)
        y_prob_med = ewc.predict(X_val_m_tokP, verbose=0).ravel()
        met_med = evaluate_probs(y_prob_med, y_val_m, name="Medical Val | Transfer-Plain+EWC")

        y_prob_pol = ewc.predict(X_val_p_tokP, verbose=0).ravel()
        met_pol = evaluate_probs(y_prob_pol, y_val_p, name="Political Val(after FT) | Transfer-Plain+EWC")

        records.append({
            "frac": frac, "method": "Transfer-Plain+EWC",
            "acc_med": met_med["acc"], "f1_med": met_med["f1"],
            "acc_pol": met_pol["acc"], "f1_pol": met_pol["f1"],
            "t_pretrain": plain_pretrain_time, "t_train": t_fine, "t_total": plain_pretrain_time + t_fine
        })

    # =====================================================
    # (3) Transfer Adapter
    # =====================================================
    print("\n[3] Transfer-Adapter")
    tf.keras.backend.clear_session(); gc.collect()
    reset_seeds(SEED)

    adp = build_domain_adapter_cnn(VOCAB_P)
    adp.set_weights(adapter_pre_weights)
    t_fine = fit_with_time(
        adp,
        {"tokens": X_train_m_sub_tokP, "domain_id": D_train_m}, y_train_m_sub,
        {"tokens": X_val_m_tokP, "domain_id": D_val_m}, y_val_m,
        verbose=0
    )
    y_prob_med = adp.predict({"tokens": X_val_m_tokP, "domain_id": D_val_m}, verbose=0).ravel()
    met_med = evaluate_probs(y_prob_med, y_val_m, name="Medical Val | Transfer-Adapter")

    y_prob_pol = adp.predict({"tokens": X_val_p_tokP, "domain_id": D_val_p}, verbose=0).ravel()
    met_pol = evaluate_probs(y_prob_pol, y_val_p, name="Political Val(after FT) | Transfer-Adapter")

    records.append({
        "frac": frac, "method": "Transfer-Adapter",
        "acc_med": met_med["acc"], "f1_med": met_med["f1"],
        "acc_pol": met_pol["acc"], "f1_pol": met_pol["f1"],
        "t_pretrain": adapter_pretrain_time, "t_train": t_fine, "t_total": adapter_pretrain_time + t_fine
    })

    # =====================================================
    # (4) Transfer Adapter + EWC (OURS)
    # =====================================================
    if USE_EWC_ADAPTER:
        print("\n[4] Transfer-Adapter+EWC (OURS)")
        tf.keras.backend.clear_session(); gc.collect()
        reset_seeds(SEED)

        base = build_domain_adapter_cnn(VOCAB_P)
        base.set_weights(adapter_pre_weights)
        ewc = EWCModel(base, fisher=adapter_fisher, theta_old=adapter_theta_old, ewc_lambda=EWC_LAMBDA)
        ewc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        t_fine = fit_with_time(
            ewc,
            {"tokens": X_train_m_sub_tokP, "domain_id": D_train_m}, y_train_m_sub,
            {"tokens": X_val_m_tokP, "domain_id": D_val_m}, y_val_m,
            verbose=0
        )
        y_prob_med = ewc.predict({"tokens": X_val_m_tokP, "domain_id": D_val_m}, verbose=0).ravel()
        met_med = evaluate_probs(y_prob_med, y_val_m, name="Medical Val | Transfer-Adapter+EWC (OURS)")

        y_prob_pol = ewc.predict({"tokens": X_val_p_tokP, "domain_id": D_val_p}, verbose=0).ravel()
        met_pol = evaluate_probs(y_prob_pol, y_val_p, name="Political Val(after FT) | Transfer-Adapter+EWC (OURS)")

        records.append({
            "frac": frac, "method": "Transfer-Adapter+EWC (OURS)",
            "acc_med": met_med["acc"], "f1_med": met_med["f1"],
            "acc_pol": met_pol["acc"], "f1_pol": met_pol["f1"],
            "t_pretrain": adapter_pretrain_time, "t_train": t_fine, "t_total": adapter_pretrain_time + t_fine
        })

    # =====================================================
    # (5) Med-Scratch
    # =====================================================
    print("\n[5] Med-Scratch")
    tf.keras.backend.clear_session(); gc.collect()
    reset_seeds(SEED)

    TOK_M, VOCAB_M = build_tokenizer(X_train_m_sub_texts, MAX_NUM_WORDS)
    X_train_m_sub_scratch = vectorize(TOK_M, X_train_m_sub_texts, MAXLEN)
    X_val_m_scratch = vectorize(TOK_M, X_val_m_texts, MAXLEN)

    med_scratch = build_plain_cnn(VOCAB_M)
    t_train = fit_with_time(med_scratch, X_train_m_sub_scratch, y_train_m_sub, X_val_m_scratch, y_val_m, verbose=0)
    y_prob_med = med_scratch.predict(X_val_m_scratch, verbose=0).ravel()
    met_med = evaluate_probs(y_prob_med, y_val_m, name="Medical Val | Med-Scratch")

    records.append({
        "frac": frac, "method": "Med-Scratch",
        "acc_med": met_med["acc"], "f1_med": met_med["f1"],
        "acc_pol": np.nan, "f1_pol": np.nan,
        "t_pretrain": 0.0, "t_train": t_train, "t_total": t_train
    })

    # =====================================================
    # (6) Pol+Med-Scratch (retokenize)  - from your 1st script
    # =====================================================
    print("\n[6] Pol+Med-Scratch (retokenize)")
    tf.keras.backend.clear_session(); gc.collect()
    reset_seeds(SEED)

    comb_texts = list(X_train_p_texts) + list(X_train_m_sub_texts)
    comb_labels = np.concatenate([y_train_p, y_train_m_sub], axis=0)

    TOK_PM, VOCAB_PM = build_tokenizer(comb_texts, MAX_NUM_WORDS)
    X_train_comb = vectorize(TOK_PM, comb_texts, MAXLEN)
    X_val_m_comb = vectorize(TOK_PM, X_val_m_texts, MAXLEN)
    X_val_p_comb = vectorize(TOK_PM, X_val_p_texts, MAXLEN)

    comb_model = build_plain_cnn(VOCAB_PM)
    t_train = fit_with_time(comb_model, X_train_comb, comb_labels, X_val_m_comb, y_val_m, verbose=0)
    y_prob_med = comb_model.predict(X_val_m_comb, verbose=0).ravel()
    met_med = evaluate_probs(y_prob_med, y_val_m, name="Medical Val | Pol+Med-Scratch (retokenize)")

    y_prob_pol = comb_model.predict(X_val_p_comb, verbose=0).ravel()
    met_pol = evaluate_probs(y_prob_pol, y_val_p, name="Political Val | Pol+Med-Scratch (retokenize)")

    records.append({
        "frac": frac, "method": "Pol+Med-Scratch (retokenize)",
        "acc_med": met_med["acc"], "f1_med": met_med["f1"],
        "acc_pol": met_pol["acc"], "f1_pol": met_pol["f1"],
        "t_pretrain": 0.0, "t_train": t_train, "t_total": t_train
    })

    # =====================================================
    # (7) Pol+Med-Scratch (fixed TOK_P)  - from your 2nd script baseline
    # =====================================================
    print("\n[7] Pol+Med-Scratch (fixed TOK_P)")
    tf.keras.backend.clear_session(); gc.collect()
    reset_seeds(SEED)

    X_train_comb_fix = np.concatenate([X_train_p_tokP, X_train_m_sub_tokP], axis=0)
    y_train_comb_fix = np.concatenate([y_train_p, y_train_m_sub], axis=0)

    comb_fix_model = build_plain_cnn(VOCAB_P)
    t_train = fit_with_time(comb_fix_model, X_train_comb_fix, y_train_comb_fix, X_val_m_tokP, y_val_m, verbose=0)
    y_prob_med = comb_fix_model.predict(X_val_m_tokP, verbose=0).ravel()
    met_med = evaluate_probs(y_prob_med, y_val_m, name="Medical Val | Pol+Med-Scratch (fixed TOK_P)")

    y_prob_pol = comb_fix_model.predict(X_val_p_tokP, verbose=0).ravel()
    met_pol = evaluate_probs(y_prob_pol, y_val_p, name="Political Val | Pol+Med-Scratch (fixed TOK_P)")

    records.append({
        "frac": frac, "method": "Pol+Med-Scratch (fixed TOK_P)",
        "acc_med": met_med["acc"], "f1_med": met_med["f1"],
        "acc_pol": met_pol["acc"], "f1_pol": met_pol["f1"],
        "t_pretrain": 0.0, "t_train": t_train, "t_total": t_train
    })


# ======================
# 11. Summary tables + plots
# ======================
df = pd.DataFrame.from_records(records)
df.to_csv("all_methods_comparison.csv", index=False)
print("\nSaved: all_methods_comparison.csv")

print("\n==================== Overall Summary (Medical Val) ====================")
pivot_acc = df.pivot(index="frac", columns="method", values="acc_med")
pivot_f1 = df.pivot(index="frac", columns="method", values="f1_med")
pivot_time = df.pivot(index="frac", columns="method", values="t_train")

print("\n[Accuracy on Medical Val]")
print(pivot_acc)

print("\n[F1 on Medical Val]")
print(pivot_f1)

print("\n[Train time per fraction (fine-tune / scratch) seconds]")
print(pivot_time)

# Best method per fraction (highlight superiority)
best_by_acc = df.loc[df.groupby("frac")["acc_med"].idxmax(), ["frac", "method", "acc_med", "f1_med"]].sort_values("frac")
best_by_f1  = df.loc[df.groupby("frac")["f1_med"].idxmax(),  ["frac", "method", "acc_med", "f1_med"]].sort_values("frac")

print("\n=== Best method per fraction (by Medical Acc) ===")
print(best_by_acc.to_string(index=False))

print("\n=== Best method per fraction (by Medical F1) ===")
print(best_by_f1.to_string(index=False))

# Ours vs Transfer-Plain delta
if ("Transfer-Adapter+EWC (OURS)" in df["method"].unique()) and ("Transfer-Plain" in df["method"].unique()):
    base = df[df["method"] == "Transfer-Plain"][["frac", "acc_med", "f1_med", "acc_pol"]].rename(
        columns={"acc_med": "acc_med_base", "f1_med": "f1_med_base", "acc_pol": "acc_pol_base"}
    )
    ours = df[df["method"] == "Transfer-Adapter+EWC (OURS)"][["frac", "acc_med", "f1_med", "acc_pol"]].rename(
        columns={"acc_med": "acc_med_ours", "f1_med": "f1_med_ours", "acc_pol": "acc_pol_ours"}
    )
    delta = base.merge(ours, on="frac", how="inner")
    delta["dAcc_med"] = delta["acc_med_ours"] - delta["acc_med_base"]
    delta["dF1_med"] = delta["f1_med_ours"] - delta["f1_med_base"]
    delta["dAcc_pol_retention"] = delta["acc_pol_ours"] - delta["acc_pol_base"]
    print("\n=== OURS vs Transfer-Plain (delta) ===")
    print(delta[["frac", "dAcc_med", "dF1_med", "dAcc_pol_retention"]].to_string(index=False))

# Plots
x_vals = [f * 100 for f in fractions]
plt.figure(figsize=(14, 10))

# (a) Medical Accuracy
plt.subplot(2, 2, 1)
for mth in METHODS:
    tmp = df[df["method"] == mth].sort_values("frac")
    plt.plot(x_vals, tmp["acc_med"].values, marker='o', label=mth)
plt.title("Medical Validation Accuracy vs. Medical Train Fraction")
plt.xlabel("Percentage of Medical Train Data (%)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

# (b) Medical F1
plt.subplot(2, 2, 2)
for mth in METHODS:
    tmp = df[df["method"] == mth].sort_values("frac")
    plt.plot(x_vals, tmp["f1_med"].values, marker='o', label=mth)
plt.title("Medical Validation F1 vs. Medical Train Fraction")
plt.xlabel("Percentage of Medical Train Data (%)")
plt.ylabel("F1")
plt.grid(True)
plt.legend()

# (c) Train time (fine-tune / scratch)
plt.subplot(2, 2, 3)
for mth in METHODS:
    tmp = df[df["method"] == mth].sort_values("frac")
    plt.plot(x_vals, tmp["t_train"].values, marker='o', label=mth)
plt.title("Training Time (per fraction: finetune/scratch) vs. Medical Train Fraction")
plt.xlabel("Percentage of Medical Train Data (%)")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.legend()

# (d) Political retention (after FT / joint training)
plt.subplot(2, 2, 4)
for mth in METHODS:
    tmp = df[(df["method"] == mth) & (~df["acc_pol"].isna())].sort_values("frac")
    if len(tmp) == 0:
        continue
    plt.plot(x_vals, tmp["acc_pol"].values, marker='o', label=mth)
plt.title("Political Val Accuracy (Retention) vs. Medical Train Fraction")
plt.xlabel("Percentage of Medical Train Data (%)")
plt.ylabel("Accuracy on Political Val")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Optional: show total time including pretrain (transfer will have constant offset)
plt.figure(figsize=(10, 5))
for mth in METHODS:
    tmp = df[df["method"] == mth].sort_values("frac")
    plt.plot(x_vals, tmp["t_total"].values, marker='o', label=mth)
plt.title("Total Time (including pretrain for transfer) vs. Medical Train Fraction")
plt.xlabel("Percentage of Medical Train Data (%)")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\nDone.")
