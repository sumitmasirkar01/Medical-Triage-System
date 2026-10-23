# Phase 2 evidence-classification amendment

This amendment preserves the original frozen protocol and corrects the evidence
hierarchy using the repository's Git chronology. It was added before manuscript
preparation and does not change any prediction, threshold, metric, or confidence
interval.

The three modality baselines are retrospective locked evaluations. Their evidence
was archived in commit `5ae9adf`, before the formal protocol first appeared in
commit `393a2e4`; they therefore must not be described as prospectively
preregistered.

The CT Phase 2 study-level evaluation is the confirmatory component. Its cohort,
candidate pooling methods, selection partition, calibration partition, untouched
test partition, and minimum study coverage were specified before the final evidence
was evaluated.

The CT study-size and chest X-ray subgroup analyses are exploratory. The CT
selective-prediction fallback and MRNet threshold-perturbation analysis are
post-test exploratory because their final analysis choices were made after their
respective locked test results were known.

Exploratory results may support hypotheses and robustness discussion. They may not
be used for confirmatory claims, causal or fairness conclusions, or prospective
operating guarantees. The machine-readable amendment is
`research/protocol_amendment_phase2.json`.
