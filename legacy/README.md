# DAEWC: Domain-Aware Adapters with Elastic Weight Consolidation

This repository contains the reference code for our research on **cross-domain fake news detection under extreme label scarcity**.

In plain terms: when a sudden event happens (e.g., a pandemic), fake news appears immediately, but **labeled training data is scarce**. Models trained on an older domain often fail on the new domain. Even worse, if we fine-tune the model on the new domain, it may **forget** what it learned before.  
**DAEWC** is designed to handle this situation in a practical and deployable way.

---

## What problem does DAEWC solve?

Fake news detection is usually trained as a **binary text classification** task: given a post/article, predict **real** vs **fake**.

However, real deployments face two connected problems:

- **Domain shift**: the topic, vocabulary, and writing style change across domains (politics → health → entertainment). A model that performs well in one domain can drop sharply in another.
- **Few-shot learning**: in a new domain, we may only have **tens of labeled examples**, especially early in an event.
- **Catastrophic forgetting**: after adapting to a new domain, a model may lose accuracy on the original domain.

DAEWC focuses on the **stability–plasticity trade-off**:
- **Stability** means “do not break what already works.”
- **Plasticity** means “learn the new domain quickly.”

DAEWC aims to get both.

---

## Core idea in one page

DAEWC splits the model into two parts with different roles:

- **Backbone (stable)**: the main encoder that learns general language patterns.
- **Adapter (plastic)**: a small “plug-in” module that learns domain-specific corrections with few parameters.

To protect the backbone, DAEWC uses:

- **EWC (Elastic Weight Consolidation)**: a regularization method that estimates which backbone weights were important for the source domain, then discourages those weights from moving during adaptation.
- **Anchor / proximity regularization**: a gentle pull that keeps the backbone close to the source solution.

A helpful analogy is a chef:
- The **backbone** is the chef’s core cooking skill.
- The **adapter** is a small spice kit for a new cuisine.
- **EWC** is the rule “do not change the techniques that are critical.”

---

## Repository layout

The code is intentionally kept simple. Each backbone is implemented as a single runnable script.

```text
DAEWC/
├─ daeewc_transformer.py   # Transformer backbone experiments
├─ daeewc_cnn.py           # CNN backbone experiments
├─ daeewc_lstm.py          # LSTM backbone experiments
├─ news/                   # Source-domain CSVs (ISOT-style): Fake.csv, True.csv
├─ covid/                  # Target-domain CSVs: fakeNews.csv, trueNews.csv
├─ More data/              # Optional extra data (not required for default runs)
└─ README.md
```

---

## Quick start

This section walks you from “I just cloned the repo” to “I got a result file”.

### 1) Create a Python environment

Any standard Python environment works. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

### 2) Install dependencies

The scripts are written in **TensorFlow 2 (Keras)** and use only a few common libraries.

```bash
pip install --upgrade pip
pip install tensorflow pandas numpy
```

Notes:
- GPU is optional. The code runs on CPU, but training will be slower.
- On Apple Silicon, the scripts automatically try `tf.keras.optimizers.legacy.Adam` for compatibility.

### 3) Check the datasets

By default, the scripts expect the following files:

- `./news/Fake.csv` and `./news/True.csv` (source domain)
- `./covid/fakeNews.csv` and `./covid/trueNews.csv` (target domain)

These paths are defined at the top of each script:

```python
PATH_FAKE = "./news/Fake.csv"
PATH_REAL = "./news/True.csv"
PATH_COVID_FAKE = "./covid/fakeNews.csv"
PATH_COVID_REAL = "./covid/trueNews.csv"
```

### 4) Run an experiment

Pick one backbone script and run it directly:

```bash
python daeewc_transformer.py
# or
python daeewc_cnn.py
# or
python daeewc_lstm.py
```

Each script runs:
- multiple random seeds (default: 5 seeds)
- multiple few-shot settings (default: 10/20/80/160 shots **per class**)

If you want a faster sanity check, edit the config block at the top of the script, for example:

```python
SEEDS = [42]
SHOTS = [10]
```

### 5) Read the outputs

After the run, the script will save CSV files with aggregated results.

- **CNN / LSTM scripts**
  - `results_daeewc_v6_dominant_raw.csv`
  - `results_daeewc_v6_dominant_summary.csv`

- **LSTM scripts**
  - `results_daeewc_LSTM_v6_dominant_raw.csv`
  - `results_daeewc_LSTM_v6_dominant_summary.csv`

- **Transformer script**
  - `results_daeewc_transformer_v6_dominant_raw.csv`
  - `results_daeewc_transformer_v6_dominant_summary.csv`

---

## What is implemented in the scripts?

The scripts follow the paper’s workflow in a compact form. Below is a map from the paper-level idea to the code-level steps.

### Stage A: Source pre-training (build a strong backbone)

The model is first trained on the **source domain** (in this repo: `news/`).  
This produces:
- a backbone that captures general language cues
- a source-domain decision threshold (selected on a dev set)

After training, the script estimates a **diagonal Fisher information** on the source domain.  
This Fisher score is the “importance map” used by EWC.

### Stage B: Few-shot target adaptation (learn the new domain safely)

In the **target domain** (in this repo: `covid/`), the code samples a few labeled examples under a **few-shot** protocol.

Then the script evaluates several adaptation strategies (see next section), including DAEWC.

The goal is:
- **high target macro-F1** (plasticity)
- **minimal drop on the source domain** (stability)

### Stage C: Inference (evaluate target + check forgetting)

After adaptation, the script evaluates:
- target-domain performance (macro-F1, accuracy)
- source-domain performance after adaptation (to measure forgetting)

---

## Methods reported in the result files

The output CSVs include a `method` field. Each value corresponds to a training strategy.

### Baselines

- **Scratch-Plain**  
  Train only on the target few-shot data, starting from random initialization.  
  This is the “no transfer learning” baseline.

- **Transfer-Plain**  
  Pre-train on the source domain, then fine-tune the full model on the target few-shot data.  
  This often adapts well, but can forget the source domain.

- **Transfer-Plain+EWC**  
  Same as Transfer-Plain, but adds an EWC penalty on backbone weights.  
  This improves stability by limiting harmful backbone updates.

- **Adapter-Only**  
  Freeze the backbone and train only a small target adapter + target head.  
  This preserves the source domain, but may underfit the target domain.

- **ReplayUpper** (optional, slow)  
  Jointly trains on source and target during adaptation by mixing batches.  
  This is used as a practical “upper bound” for stability, but it costs more compute.

### DAEWC (ours)

- **DAEWC**  
  This repo includes a strong training recipe that reflects the paper’s core idea (adapter + backbone protection) and also supports a practical semi-supervised extension:
  - **few labeled target samples**
  - **unlabeled target pool** (the remaining target training examples)
  - **FixMatch-style pseudo-labeling** (high-confidence teacher predictions become training targets)
  - **EMA teacher** (an exponential moving average copy of the student weights, used for stable pseudo-labels)
  - optional **EWC + anchor** when a small part of the backbone is unfrozen
  - optional **source replay** as an extra stabilizer

If you want the simplest “paper core” comparison, focus on:
- Adapter-Only (pure adapter training)
- Transfer-Plain+EWC (backbone fine-tuning with stability regularization)

---

## Configuration guide

All important settings are in a **single CONFIG block** at the top of each script.  
This section explains the most useful knobs to edit.

### Few-shot protocol

You can choose how shots are counted:

- `SHOT_MODE = "per_class"`: `K` labeled examples **per class** (real and fake). Total is `2K`.
- `SHOT_MODE = "total"`: `K` labeled examples total, sampled in a balanced way.

The actual values are in:

```python
SHOTS = [10, 20, 80, 160]
SEEDS = [42, 43, 44, 45, 46]
```

### Tokenization and sequence length

The repo uses a simple **source-only tokenizer**:
- it builds a vocabulary from the source training split only
- it keeps the vocabulary fixed for the target domain

Key parameters:

```python
MAX_VOCAB = 5000
MAX_LEN = 256
```

### DAEWC (FixMatch) knobs

If you want to explore the semi-supervised behavior, these are the most relevant parameters:

- `DAEWC_TAU_BY_SHOT`: confidence threshold for accepting pseudo-labels
- `DAEWC_LAMBDA_U_BY_SHOT`: weight of unlabeled loss
- `DAEWC_EMA_DECAY`: EMA teacher smoothing factor
- `DAEWC_WEAK_DROP` / `DAEWC_STRONG_DROP`: token-drop augmentation strength
- `DAEWC_ULB_MAX`: optional cap on unlabeled pool size (for speed)

---

## Using your own data

This repo is designed for research experiments. Still, you can adapt it to new datasets with minimal changes.

### Expected input format

Each domain is loaded from **two CSV files**:
- one file for **fake** samples
- one file for **real** samples

The loader tries to find a suitable text column automatically. It prefers:
- `title` + one of `text/content/body/...`

If those are missing, it falls back to the longest string column.

Label convention in the code:
- **fake = 1**
- **real = 0**

### Practical tips

To reduce surprises:
- Keep each CSV in UTF-8 if possible.
- Make sure the text is not empty after cleaning.
- If your dataset is short-form (tweets), consider lowering `MAX_LEN`.

---

## Reproducibility notes

The scripts include several choices meant to make results comparable:

- **Stratified split** into train/dev/test (default: 70/10/20).
- **Exact de-duplication** inside each domain to remove repeated texts.
- **Threshold calibration** on the dev set to maximize macro-F1.
- Reporting macro-F1, which is robust when classes are imbalanced.

---

---

## Data sources and acknowledgements

This repo follows common practice in fake news detection research and builds on public benchmark datasets.

- **ISOT Fake News Dataset** (True.csv / Fake.csv)  
  Widely used fake-news benchmark with Reuters-based real news and curated fake news.  
  See the ISOT dataset readme (University of Victoria):  
  https://onlineacademiccommunity.uvic.ca/isot/

- **COVID-19 Fake News Dataset (Constraint / Patwa et al.)**  
  COVID-19 misinformation dataset released for research and shared tasks.  
  Paper: https://arxiv.org/abs/2011.03327  
  Official repo: https://github.com/parthpatwa/covid19-fake-news-detection

Please respect the original licenses and terms of use of each dataset.

---

## Ethical note

Misinformation detection is a dual-use technology. In real deployments:
- false positives can suppress legitimate speech
- domain shift can create uneven error rates across communities

DAEWC is designed to be more **auditable** (changes are localized in adapters and regularized in the backbone), but it is still not a substitute for careful evaluation and human oversight.


## License
This project is licensed under the MIT License - see the LICENSE file for details.
