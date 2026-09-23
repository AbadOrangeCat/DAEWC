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

Historical scripts are preserved in Git history and the separate local archive; their old headline scores are not revised-study results. The versioned code and weight release is v2.0.0.

On 24 September 2026, paper-only utilities and retired entry points were moved to the separate local archive. Training, evaluation, data-audit code, and numerical evidence are unchanged.

The v2.0.0 release adds verified weight archives, a resumable downloader, extraction checks, and restoration instructions. No model is retrained and no numerical result changes.
