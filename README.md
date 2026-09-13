# DAEWC revision 2.0.0

This revision evaluates adaptation of a text classifier under a strict target-label budget. It implements one model with a residual adapter and learned domain feature gate after every BERT block. Elastic weight consolidation (EWC) and a proximity penalty apply only to the shared block biases and LayerNorm scales. The target optimizer receives no source examples, extra target development labels, or unlabelled pool.

The model predicts the supplied dataset labels. It does not retrieve evidence or verify a statement against external facts. Known domain identity is required at inference.

## What changed

The submitted scripts are preserved in `legacy/` and retired as entry points. Their results cannot support the revised paper: they used political test information in threshold or candidate selection and did not implement the described model. The revised manuscript does not reuse their numerical tables or CNN/LSTM claims.

The primary political source is non-health LIAR, with an explicit binary label mapping and official split roles. The health target is LIAR's `health-care` subject subset. The two other targets use only the `title` fields of FakeNewsNet PolitiFact and GossipCop CSV files. The original ISOT and medical files are used for diagnostics only. The medical files are not identified as CONSTRAINT/Patwa data. The older PolitiFact full-text fake/real files are byte-identical and excluded.

## Environment

Use Python 3.13 and install `requirements.txt`. The saved environment metadata records exact runtime versions. CPU, Apple MPS, and CUDA are supported. For the additional runners, set `--device cpu` or `--device cuda` when not using their local MPS default. Training and primary evaluation use MPS; the post-fit second-test and objective diagnostics use CPU. Cross-device floating-point results may differ slightly.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
```

## Reproduce the experiments

Run these commands from the repository root. The preparation command downloads the original LIAR archive and requires the existing `Moredata/politifact_{real,fake}.csv` and `Moredata/gossipcop_{real,fake}.csv` inputs. Input hashes are recorded and should match the supplied release manifest. Missing fields or malformed rows are errors; the loader never guesses an input column or silently skips CSV errors.

```bash
python -m daewc.data --data-root . --out revision_artifacts/data
python -m daewc.audit --data-root . --prepared revision_artifacts/data --out revision_artifacts/audit
python -m daewc.run --config configs/local.json --data revision_artifacts/data --out revision_artifacts/local
```

The primary run uses the pinned Google two-layer BERT checkpoint. Configurations are saved before test results are produced. Resume is refused if the configuration, processed data, or training code differs. Each result includes the selected procedure, exact labels, thresholds, trainable parameter names, elapsed training time, full-precision predictions, and a checkpoint checksum.

To run the additional protocols, download the checkpoint named in the configuration, then pass its local directory as `MODEL_DIR` below. `MODEL_DIR` must contain `config.json`, `vocab.txt`, and `model.safetensors`; it is not a shell variable created automatically by this repository.

```bash
python -m daewc.sequential --config configs/local.json --data revision_artifacts/data --single revision_artifacts/local --out revision_artifacts/sequential --model-path MODEL_DIR
python -m daewc.budget_cv --config configs/local.json --data revision_artifacts/data --single revision_artifacts/local --out revision_artifacts/budget_cv --model-path MODEL_DIR
python -m daewc.mechanism_sweep --config configs/local.json --data revision_artifacts/data --single revision_artifacts/local --out revision_artifacts/mechanism --model-path MODEL_DIR
python -m daewc.run --config configs/random_initialization.json --data revision_artifacts/data --out revision_artifacts/random
python -m daewc.run --config configs/bert_base.json --data revision_artifacts/data --out revision_artifacts/bert_base
```

The fixed configuration uses one candidate and 80 updates. The budgeted cross-validation protocol uses two folds and three learning-rate multipliers for **every** method, then refits on all 2K labels. These protocols are reported separately. No target development labels enter training or selection. The mechanism sweep reports its complete 3-by-3 grid; it does not choose a replacement DAEWC configuration using test scores.

## Smaller modules and post-fit evaluation

The smaller configuration uses adapter width 2 and domain-vector width 4. It trains 6,154 parameters with the compact encoder, compared with 16,434 for the reference DAEWC and 8,450 for rank-8 LoRA. It is a separately identified exploratory extension. It reuses **exact** primary source checkpoints and Fisher buffers, with checksum provenance; it does not retrain a new source for the paired comparison.

```bash
python -m daewc.reuse_source --config configs/low_footprint.json --source revision_artifacts/local --data revision_artifacts/data --out revision_artifacts/low_footprint --model-path MODEL_DIR
python -m daewc.budget_cv --config configs/low_footprint.json --single revision_artifacts/low_footprint --data revision_artifacts/data --out revision_artifacts/low_footprint_cv --model-path MODEL_DIR
python -m daewc.confirmatory --data revision_artifacts/data --artifacts revision_artifacts --configs configs --out revision_artifacts/confirmation --model-path MODEL_DIR --freeze-only
python -m daewc.confirmatory --data revision_artifacts/data --artifacts revision_artifacts --configs configs --out revision_artifacts/confirmation --model-path MODEL_DIR
```

The second test uses the original target development partitions only after all compared K=80 models have been fitted. Its frozen plan includes all seven reference methods and both smaller methods under both selection protocols, for all domains and seeds. Its labels never enter adaptation, model selection, threshold fitting, or any subsequent model change. This analysis contains 162 post-fit evaluations; it does not add training runs or extra supervision.

The directory `exploratory_repeat_source/` preserves an earlier numerical repeat that independently retrained the source. Source decisions were unchanged, but small floating-point weight differences occurred. Those records are excluded from the paired architecture comparison for this stated procedural reason, with their status retained. They are not silently discarded or selected by performance.

## Tables and independent verification

```bash
python -m daewc.summarize --runs revision_artifacts/local --sequential revision_artifacts/sequential --out revision_artifacts/tables/local
python -m daewc.summarize --runs revision_artifacts/budget_cv --out revision_artifacts/tables/budget_cv
python -m daewc.verify --data revision_artifacts/data --runs revision_artifacts/local --source revision_artifacts/local --config configs/local.json --model-path MODEL_DIR --out revision_artifacts/verification.json
```

Use the same summary command for the standard BERT, random-initialization, and smaller-module directories. For the extra protocols and the conditional objective diagnostic:

```bash
python -m daewc.verify_additional --artifacts revision_artifacts --data revision_artifacts/data --configs configs --confirmation --out revision_artifacts/verification_additional.json
python -m daewc.stability_diagnostics --config configs/local.json --data revision_artifacts/data --runs revision_artifacts/local --model-path MODEL_DIR --out revision_artifacts/stability
python scripts/build_tables.py --artifacts revision_artifacts --manuscript manuscript
python scripts/draw_results.py --artifacts revision_artifacts --manuscript manuscript
python scripts/draw_architecture.py --manuscript manuscript
```

The second-test summary also computes the complete prespecified paired bootstrap comparisons:

```bash
python -m daewc.second_test_summary --input revision_artifacts/confirmation --out revision_artifacts/tables/confirmation
python scripts/build_extension_tables.py --artifacts revision_artifacts --manuscript manuscript
```

The objective diagnostic checks the actual deterministic target loss and weighted parameter drift. It does not certify the integrated source-gradient bound or source F1. All 36 reference and all 36 smaller DAEWC runs satisfy the measured objective condition.

The verifier recomputes metrics from saved predictions, checks train/test membership and label budgets, verifies checkpoint hashes, and reconstructs one result per method from the source checkpoint plus saved trainable parameters. Do not load untrusted PyTorch checkpoints.

The signed retention measure is `source_after_f1 - source_before_f1`, in percentage points. Improvements remain positive for every method. Sensitivity tables include both loss-only and absolute-band rules at 0.5, 1, 2, and 5 percentage points. Seed intervals are conditional on one fixed test set; they are not population-level confidence guarantees.

## Sequential validation of smaller modules

The smaller configuration also uses all six orders and three seeds, with all four sequential controls. It starts from the exact imported source checkpoints.

```bash
python -m daewc.sequential --config configs/low_footprint.json --data revision_artifacts/data --single revision_artifacts/low_footprint --out revision_artifacts/sequential_low --model-path MODEL_DIR
```

## Exact gate precomputation for inference

A trained gate is constant for a known domain. The exporter stores that feature-scale vector instead of its domain embedding and projection. It preserves the trained prediction function. The smaller architecture's source-relative export contains 5,126 floating-point values (changed shared calibration plus domain modules), versus 8,450 for rank-8 LoRA. The stored domain-specific part contains 1,798 values. These counts exclude the common source model and serialization headers.

```bash
python -m daewc.export_inference --config configs/low_footprint.json --data revision_artifacts/data --runs revision_artifacts/low_footprint --out revision_artifacts/inference_exports --model-path MODEL_DIR
```

All 36 serialized exports were reloaded and checked against the original models on their complete target tests, using CPU and identical batches. Their probabilities match exactly. `daewc.export_inference.load_export` reconstructs an exported model from its source checkpoint and configuration. These exports are for inference; retain the original training checkpoint for further optimization.

## Artifacts and storage

`revision_artifacts/` contains processed records and manifests, source checkpoints and Fisher buffers, run JSON files, source-relative checkpoints, predictions, and tables. Sequential checkpoints are incremental: reconstruction uses the source plus each earlier stage in order. A `training` record lists the exact parameter names in each checkpoint.

Peak CUDA allocation is measured only on CUDA. On MPS, the process-lifetime resident-memory maximum is labelled as such and is not presented as a per-run GPU memory peak. Deployment storage counts parameter elements at four bytes each; optimizer states and Fisher buffers must be counted separately if they are retained for later training.

## Licenses and public release

The repository's existing software license applies to its code. Dataset and pretrained-model licenses remain separate. Input transformations and checksums identify the releases used; the software license does not grant rights to redistribute third-party text. The revised code is prepared locally. A public release and permanent identifier must be created by the authors before claiming that an updated public archive is available.

## Rebuilding the complete revision

The full local snapshot contains `manuscript/`, the original model files in `models/`, and saved checkpoints. The smaller ZIP contains code, configurations, processed data, predictions, and result records, but omits `.pt` and `model.safetensors` files. Its omission manifest identifies those files. Use the complete local snapshot to reconstruct saved predictions. To retrain from the smaller ZIP, first obtain the pinned initial models and **use a fresh output directory**, such as `rerun/local`; existing result JSON files in `revision_artifacts/` must not be treated as retraining outputs without their matching weights.

```bash
python scripts/download_models.py --out models
python scripts/verify_release.py --artifacts revision_artifacts
python -m daewc.verify_additional --artifacts revision_artifacts --data revision_artifacts/data --configs configs --sequence-directory sequential_low --skip-mechanism --out revision_artifacts/verification_sequential_low.json
python scripts/draw_parameter_comparison.py --artifacts revision_artifacts --manuscript manuscript
python scripts/build_references.py --metadata-cache reference_sources --manuscript manuscript
tectonic manuscript/manuscript.tex
tectonic manuscript/manuscript_marked.tex
```

`verify_release.py` requires the full snapshot because it checks source reuse and exported-weight hashes. It checks all final model counts, shared target identifiers across protocols, nested budgets, frozen training code, and the numerical claims about the smaller model. The detailed verification reports also recompute metrics and reconstruct representative checkpoints. Test the implementation with `python -m pytest -q`.
