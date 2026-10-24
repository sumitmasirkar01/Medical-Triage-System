# Phase 3 head CT series ingestion

Status: implemented and validated on 2026-10-24.

`stat_triage.ct_series.validate_head_ct_series` accepts one complete, single-series
head CT request. Validation occurs before pixel decoding and requires:

- 10–512 single-frame CT DICOM instances;
- one patient, study, and series identifier;
- unique file paths, SOP instance identifiers, and slice-order values;
- consistent two-dimensional slice geometry;
- deterministic ordering by `ImagePositionPatient[2]`, with `InstanceNumber` used
  only when every slice lacks image-position metadata;
- consistent physical Z spacing, or consecutive fallback instance numbers, to
  reject likely missing interior slices;
- no file larger than 32 MiB and no series larger than 512 MiB.

Malformed reads are classified as processing failures. Incomplete, mixed,
multi-frame, non-CT, duplicate, or excessive inputs are classified as unsupported.
Each error contains a stable machine-readable code for the Phase 3 service layer.

`HeadCTStudyInferenceAdapter` verifies the archived locked-protocol ZIP against its
declared SHA-256, requires the frozen mean-pooling rule, streams the validated
slices through the locked EfficientNet-B0 checkpoint, and applies the frozen
study threshold of `0.20713453119969927`. It emits `ModelEvidence` for the
`study_level_any_hemorrhage` endpoint with one input hash per SOP instance.

This contract does not combine multiple acquisitions or series, infer missing
anatomy, or establish prospective clinical performance. Such requests must be
returned as `UNSUPPORTED` until a separately validated selection policy exists.
