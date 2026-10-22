# Frozen research protocol

Protocol version: `2026-10-22.1`. Machine-readable source: `research/protocol.json`.

The primary question is whether a modality-aware routed system can provide
reproducible sensitivity-constrained operating points across head CT, chest
X-ray, and knee MRI. Each modality retains its own labels, preprocessing,
checkpoint, calibration, threshold, and evaluation unit.

Training selects parameters; validation selects models; calibration fits score
calibration and thresholds; locked test partitions estimate final performance.
Locked test results cannot guide later choices. Post-test work must use a new
untouched cohort or be labelled exploratory.

Primary reporting includes ROC-AUC, average precision, sensitivity, specificity,
sample counts, and grouped 95% confidence intervals. Calibration reporting uses
negative log-likelihood, Brier score, and expected calibration error. Every run
must record its seed, Git commit, dataset identity, split manifest hash, config
hash, and artifact hashes in `research/experiment_registry.json`.

Required ablations must change one declared factor at a time and reuse the same
development partitions. Comparisons across modalities are prohibited because
their cohorts, targets, prevalence, and evaluation units differ. Missing or
corrupt inputs must fail closed. This remains a research demonstration and is
not authorized for clinical use.

## Phase 2 head CT study endpoint

Study aggregation is selected using validation studies, its sensitivity-constrained
threshold is selected using calibration studies, and the locked test is evaluated
once. Initial deterministic candidates are maximum, mean, top-k mean, and noisy-OR
pooling. A cohort must contain a median of at least ten slices per study; the
existing enriched slice pilot fails this gate and cannot support a study-level
claim. Learned attention or sequence pooling requires a separately registered
training run and cannot use locked test outcomes for architecture selection.
