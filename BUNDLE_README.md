# Code and saved evidence

This folder contains the revised code, configurations, datasets, saved predictions, result records, and verification reports. It does not contain the manuscript or model weights. See `README.md` for execution instructions and `RELEASE_NOTES.md` for the scope of this snapshot.

Run `python scripts/verify_manifest.py` to check file integrity. After installing `requirements.txt`, run `python scripts/verify_saved_results.py` to independently recompute the 666 final single-target records without weights.

`OMITTED_WEIGHTS.json` identifies omitted initial models and trained checkpoints by hash. Initial models can be downloaded with `scripts/download_models.py`; trained checkpoints are a separate part of the authors' full archive. Use fresh output directories when retraining.
