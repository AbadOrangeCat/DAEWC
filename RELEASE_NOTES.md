# Code and evidence snapshot, 24 September 2026

This snapshot contains the current revised implementation and all saved experiment records. It does not represent a new training run. The frozen training code, labels, predictions, and numerical results are unchanged.

## Included evidence

| Final-model group | Records |
| --- | ---: |
| Primary fixed configuration | 288 |
| Internal cross-validation | 126 |
| Random initialization | 45 |
| Standard BERT | 63 |
| Smaller modules | 108 |
| Smaller modules with cross-validation | 36 |
| Total | 666 |

The additional directories retain 81 mechanism runs, 144 sequential runs, 162 post-fit second-test evaluations, 36 inference-export verification records, and the excluded independent-source repeat. The repeated source experiment remains explicitly excluded from the paired comparison. Saved verification reports distinguish checks performed with the full local archive from checks possible in this repository.

## Packaging changes

- Moved paper-only source builders, figures, bibliography metadata, and retired training scripts to the separate local archive. The repository contains the current experiment implementation and verification tools.
- Added an explicit saved-prediction verification command. Missing weights are reported as unverified; available weights still have their checksums checked.
- Included experiment records in Git tracking rules and disabled line-ending conversion to preserve frozen file hashes.
- Removed local editor settings, operating-system metadata, caches, and the separately supplied manuscript from this folder. Git history is preserved.
- Replaced local machine paths in diagnostic logs and the data manifest with portable locations. Numerical records and frozen experiment plans are unchanged. `release/PORTABILITY.json` records the affected file hashes; originals remain in the authors' local archive.

The two paper-prose tests were archived with their paper utility; all 25 remaining model and protocol tests pass. The archived training commit is recorded in `revision_artifacts/code_revision.json`. The complete current file list is identified by `MANIFEST_SHA256.json`, which excludes itself, Git internals, and newly generated local outputs.

## Verification and reproduction

`scripts/verify_manifest.py` checks every supplied file's byte count and SHA-256 hash using only the Python standard library. `scripts/verify_saved_results.py` recomputes all 666 final-model records after installing `requirements.txt`. Its scope includes complete run designs, frozen code/data hashes, full and matched test metrics, source baselines, fixed thresholds, paired nested label budgets, and cross-validation selection records. It does not rerun checkpoint inference or recompute the additional protocols.

Trained weights are omitted from this Git repository. `OMITTED_WEIGHTS.json` records their names and hashes, and `INITIAL_MODELS.json` identifies the pinned pretrained model versions. A checksum is not a substitute for access to a checkpoint: exact reconstruction requires the trained weights from the authors' full archive. Fresh training is supported with downloaded initial models and new output directories. All original weight files are supplied in the [v2.0.0 release](https://github.com/AbadOrangeCat/DAEWC/releases/tag/v2.0.0), with archive and per-file SHA-256 checksums. `scripts/download_weights.py` restores and verifies them in the repository layout.

## Upload to GitHub

Use Git or GitHub Desktop to publish this directory and preserve its nested file layout. Commit the saved results as well as the code. No file in this snapshot exceeds GitHub's 100 MiB regular-file limit, but the two diagnostic `news/*.csv` files exceed the 25 MiB browser-upload limit. Do not upload only a ZIP as a replacement for the browseable repository.

GitHub size guidance: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

This release publishes the current implementation on `main` and supplies the matching weight archives. The manuscript and reviewer correspondence are delivered separately. Dataset and model rights remain separate from the software license.

## Weight release

All 1,355 archived weight files are supplied in 15 independent ZIP archives. Each archive is below 2 GiB. `SHA256SUMS` checks the downloads; `WEIGHTS_MANIFEST.json` lists every file and its original checksum. Restore them with `python scripts/download_weights.py`, then run `python scripts/download_weights.py --verify-only`. See README for extraction locations and full-checkpoint verification.
