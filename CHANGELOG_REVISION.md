# Revision 2.0.0-revision

This revision replaces the submitted efficacy evidence with executable, independently checkable experiments.

- Replaced post-pooling legacy code with per-block residual adapters, explicit domain gates, and calibrated shared biases and LayerNorm scales.
- Restricted DAEWC EWC and proximity penalties to the shared calibration subset. Estimated the observed-label empirical Fisher per example on source training data.
- Removed source-test threshold fitting, test-guided candidate selection, clipped source improvements, hidden replay, and unlabelled target pools from the revised procedures.
- Added strict total label accounting, nested paired samples, a fixed training protocol, and equally budgeted internal cross-validation.
- Rebuilt the main benchmark using LIAR statements and documented FakeNewsNet titles, with exact and verified near-duplicate exclusions. Kept misleading legacy data only as explicit diagnostics.
- Added pretrained head-only, full fine-tuning, full EWC, per-block adapters, LoRA and LwF controls; four individual component removals; complete rate/Fisher grid; random initialization and standard BERT controls.
- Recorded all six target orders with three seeds, intermediate matrices, source-relative stage checkpoints, and separate source/older-target forgetting measures.
- Added a smaller domain-module configuration, exact source-cache reuse, and a frozen post-fit second test of all comparison methods.
- Added run-level predictions, parameter-name scopes, data identifiers, numerical configurations, costs, immutable model revisions, checksums and reconstruction tools.
- Rewrote the English manuscript, conditional stability analysis, BibTeX references, architecture figure, tables, and six-section reviewer response.

Historical scripts are retained in `legacy/`; their old headline scores are not revised-study results. This is a local revision. No public repository update or journal submission is implied.
