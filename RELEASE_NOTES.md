# Code and evidence snapshot, 22 September 2026

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

- Restored the current Overleaf export script, including the original Elsevier CAS template dependencies, and the reference generator's HTML-entity handling.
- Added an explicit saved-prediction verification command. Missing weights are reported as unverified; available weights still have their checksums checked.
- Included experiment records in Git tracking rules and disabled line-ending conversion to preserve frozen file hashes.
- Removed local editor settings, operating-system metadata, caches, and the separately supplied manuscript from this folder. Git history is preserved.
- Replaced local machine paths in diagnostic logs and the data manifest with portable locations. Numerical records and frozen experiment plans are unchanged. `release/PORTABILITY.json` records the affected file hashes; originals remain in the authors' local archive.

The archived training commit is recorded in `revision_artifacts/code_revision.json`. The complete current file list is identified by `MANIFEST_SHA256.json`, which excludes itself, Git internals, and newly generated local outputs.

## Verification and reproduction

`scripts/verify_manifest.py` checks every supplied file's byte count and SHA-256 hash using only the Python standard library. `scripts/verify_saved_results.py` recomputes all 666 final-model records after installing `requirements.txt`. Its scope includes complete run designs, frozen code/data hashes, full and matched test metrics, source baselines, fixed thresholds, paired nested label budgets, and cross-validation selection records. It does not rerun checkpoint inference or recompute the additional protocols.

Trained weights are omitted from this Git repository. `OMITTED_WEIGHTS.json` records their names and hashes, and `INITIAL_MODELS.json` identifies the pinned pretrained model versions. A checksum is not a substitute for access to a checkpoint: exact reconstruction requires the trained weights from the authors' full archive. Fresh training is supported with downloaded initial models and new output directories. A public location for the archived trained weights remains to be supplied by the authors.

## Upload to GitHub

Use Git or GitHub Desktop to publish this directory and preserve its nested file layout. Commit the saved results as well as the code. No file in this snapshot exceeds GitHub's 100 MiB regular-file limit, but the two diagnostic `news/*.csv` files exceed the 25 MiB browser-upload limit. Do not upload only a ZIP as a replacement for the browseable repository.

GitHub size guidance: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

The repository has not been pushed as part of this preparation. The manuscript and reviewer correspondence are delivered separately. Dataset and model rights remain separate from the software license.
