# Code and saved evidence

This folder contains the revised code, configurations, datasets, saved predictions, result records, and verification reports. The manuscript is separate. Weights are separate release attachments. See `README.md` for execution instructions and `RELEASE_NOTES.md` for the scope of this snapshot.

Run `python scripts/verify_manifest.py` to check file integrity. After installing `requirements.txt`, run `python scripts/verify_saved_results.py` to independently recompute the 666 final single-target records without weights.

`OMITTED_WEIGHTS.json` identifies omitted initial models and trained checkpoints by hash. Initial models can be downloaded with `scripts/download_models.py`; all original trained checkpoints are provided by the [v2.0.0 release](https://github.com/AbadOrangeCat/DAEWC/releases/tag/v2.0.0). Run `python scripts/download_weights.py` to restore and verify them. Use fresh output directories when retraining.
