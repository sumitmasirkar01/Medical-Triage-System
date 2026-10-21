# Validated multimodal execution plan

Validated on: 2026-10-12

## 1. Evidence boundary

The intended product is a research-grade multimodal triage prototype with three
independently evaluated imaging adapters and one auditable worklist:

1. RSNA head CT: epidural, intraparenchymal, intraventricular,
   subarachnoid, subdural, and `any` hemorrhage.
2. Chest X-ray: pneumothorax, pleural effusion, consolidation, and
   atelectasis, with NIH and CheXpert results reported separately.
3. MRNet knee MRI: abnormality, ACL tear, and meniscal tear at exam level.

The current evidence supports only the RSNA **slice-level pilot**. Its locked test
cohort contains 242 slices from 222 patients. The saved test ROC-AUC is 0.8417,
sensitivity 0.8831, and specificity 0.5091 at calibration threshold 0.253861.
The checkpoint SHA-256 is
`4a15c605bb1481de7f12a4881f9c9b5adb36d9386ecac0aa6eb3ae7c3ae805b1`.
These results do not establish study-level CT performance or performance on CXR
or MRI.

The supported package currently contains ingestion, evaluation, the RSNA adapter,
and the worklist. The test suite passes: 68 tests. The legacy tree contains useful ideas
and several checkpoints, but it is not a validated second implementation.

## 2. Legacy decision register

| Legacy component | Validation finding | Decision |
| --- | --- | --- |
| `core/preprocessing.py`, `core/dicom_parser.py` | Some transforms and DICOM handling are useful, but the supported RSNA implementation is now the reference | Compare behavior, migrate only missing functions with focused tests |
| `core/calibration.py::TemperatureScaler` | Scalar temperature scaling is mathematically reasonable, but legacy parameters were not fitted to documented held-out logits | Rewrite in the supported package; fit and save parameters per modality/target on calibration data only |
| `ClinicalThresholdOptimizer` | Arbitrary threshold clipping can violate its sensitivity target | Replace with an explicit constrained operating-point selector and bootstrap uncertainty |
| `core/axial_transformer.py::AsymmetricLoss` | Useful multi-label loss family; legacy version omits probability clipping from the published formulation | Keep as a registered ablation after weighted BCE; implement and test the complete formula |
| Axial Transformer | Shape logic is usable, but its five-class head omits IVH and the saved weights were trained on synthetic embeddings | Architecture experiment only; never load as clinical evidence |
| `core/conformal.py` | Implements categorical APS, while every target here is binary/multi-label; legacy adapters calibrate it on synthetic samples | Defer; later implement and evaluate a multi-label method on a real, untouched calibration set |
| `core/sequential_sprt.py` | Not a Wald SPRT: it resets the likelihood ratio, never returns its lower decision, and legacy tests expect a nonexistent state | Exclude from the supported pipeline; reconsider only with an incremental model and a formally specified sequential test |
| `core/counterfactual.py` | Produces optimized pixel perturbations, not causal or anatomically constrained explanations | Research-only experiment after the three baselines; never present as clinical explanation |
| `core/open_clip_engine.py` | Does not apply the returned validation transform and applies CLIP scale incorrectly; its temperature argument is unused | Rewrite before any BiomedCLIP experiment |
| Legacy head model | Five-way softmax, synthetic calibration, omitted IVH, and SPRT after full computation | Do not migrate; retain only study-aggregation architecture ideas |
| Legacy chest model | Uses softmax for coexisting findings and a heuristic combined risk; adapter does not match saved DenseNet checkpoint | Replace with independent sigmoid outputs and a documented policy |
| Legacy MRI model | Random projection/four-way softmax does not implement MRNet | Replace completely |
| `core/gateway.py` | Raw scores across modalities are incomparable; certainty weighting depresses ambiguous exams | Retain the policy concept, use explicit zones and a deterministic within-zone queue |
| `core/simulation.py` | Useful synthetic engineering fixture, but cannot validate diagnostic accuracy or clinical queue benefit | Keep for fault/load tests; add a real discrete-event queue experiment later |

### Checkpoint disposition

The three additional checkpoints are **quarantined legacy artifacts** until they
have data manifests, split identities, training code, preprocessing contracts,
and held-out predictions:

- Chest checkpoint: DenseNet121, six outputs, including `Critical_Any`; its labels
  omit atelectasis and do not match the required contract.
- Head GRU checkpoint: bidirectional GRU and six-output classifier, but the image
  encoder and matching implementation are missing.
- Knee checkpoint: EfficientNet-like, 13 outputs, and a target ontology that does
  not match MRNet's three tasks. Its SHA-256 is
  `db02267340a0a73d35f540fd08b828752eb621ec7379c58676e45ecdb9ede84a`.

No performance number stored inside one of these checkpoint dictionaries is an
evaluation result. A claim requires cohort-level predictions and reproducible
metric computation.

## 3. Common mathematical contract

Each finding is an independent binary target. For logits `z_ik`, labels `y_ik`,
optional annotation mask `m_ik`, and training-only positive weights `w_k`, the
primary loss is:

```text
p_ik = sigmoid(z_ik)
L = sum(m_ik * BCEWithLogits(z_ik, y_ik, pos_weight=w_k)) / sum(m_ik)
```

Weighted BCE is the primary objective because it is simple and auditable. The
complete asymmetric loss is an ablation for rare labels; it advances only if it
improves a predeclared validation metric without damaging calibration or key-label
sensitivity.

Calibration is fitted to frozen validation logits on a dedicated calibration
partition. Candidate methods are identity, per-target temperature scaling, and
isotonic regression when sample counts support it. Select using calibration NLL
and Brier score, record ECE for description, and retain identity when calibration
does not improve held-out calibration metrics.

For a required sensitivity `S*`, choose the largest threshold on the calibration
set that satisfies `sensitivity(t) >= S*`. Do not clip it to a preferred numerical
range. Lock the model, calibrator, threshold, and hashes before one test evaluation.

Report ROC-AUC, average precision, sensitivity, specificity, PPV, NPV, accuracy,
Brier score, confusion counts, prevalence, and sample counts. Bootstrap confidence
intervals at the patient/exam level, preserving all images from the resampled unit.
Undefined metrics and targets with too few positives must be labelled as such.

## 4. Modality implementations

### 4.1 Head CT

Keep the current slice baseline frozen as release 0.1. Build a separate study-level
experiment rather than changing its test result.

1. Construct complete studies from ordered DICOM slices with patient-disjoint
   train, validation, calibration, and test splits.
2. Establish simple aggregation baselines first: maximum and top-k mean of frozen
   slice probabilities.
3. Add a learned sequence model only after those baselines: frozen EfficientNet
   slice embeddings followed by a masked bidirectional GRU or small Transformer.
4. Produce six sigmoid logits. `any` is trained/evaluated explicitly and checked
   against the logical union of subtype labels.
5. Compare weighted BCE and full ASL. Select once on validation, calibrate once,
   and evaluate test once.

The legacy GRU may guide tensor shapes, but its weights are unusable until its
missing backbone and provenance are recovered. The sequential stopping idea waits
until inference truly processes slices causally; applying it after encoding the
whole series cannot support a compute-saving claim.

### 4.2 Chest X-ray

Use a DenseNet121 ImageNet baseline with four independent sigmoid logits. Maintain
dataset-specific manifests and evaluations for NIH and CheXpert; do not pool test
sets or hide domain shift.

1. Define frontal-view inclusion, patient-disjoint splits, image normalization,
   and CheXpert uncertainty policy before training.
2. For CheXpert, represent uncertain and unmentioned states explicitly in the
   manifest. Use a per-target mask or a predeclared uncertainty mapping; never
   silently coerce them during loading.
3. Train the supervised DenseNet baseline. Reconstruct the legacy checkpoint only
   as a forensic experiment and only if its exact six-label contract can be
   recovered.
4. Rewrite BiomedCLIP as a comparator: use the official preprocessing transform,
   normalized image/text embeddings, correct learned logit scale, and independent
   positive/negative prompts per finding. A softmax across diseases is invalid
   because findings can coexist.
5. Evaluate supervised, zero-shot, and optionally frozen-embedding linear-probe
   models under the same splits. Calibration and thresholds remain model-specific.

### 4.3 Knee MRI

Implement the MRNet exam contract from scratch: three independent tasks and three
anatomical planes.

1. Build one manifest row per exam with axial, coronal, and sagittal series plus
   abnormality, ACL, and meniscus labels.
2. Decode and normalize each series reproducibly; use variable-length masks.
3. Encode slices with a shared or plane-specific 2D backbone, pool within each
   plane using max pooling as the first baseline, and combine plane logits with a
   learned logistic fusion layer.
4. Compare single-plane and three-plane results. Use exam-level splits, bootstrap,
   calibration, and latency.

The 13-output legacy knee checkpoint cannot initialize the MRNet head. Its encoder
could be considered only after provenance recovery demonstrates compatible MRI
preprocessing; ImageNet initialization is the auditable default.

## 5. Unified evidence and worklist contract

Every adapter returns one immutable record containing:

```text
exam_id, patient_id, modality, model_id, model_sha256,
preprocess_version, target_scores, calibrated_scores,
thresholds, decisions, uncertainty_status, latency_ms,
input_hashes, warnings, created_at
```

The worklist receives decisions from modality-specific policies. It must never rank
raw CT, CXR, and MRI probabilities as if they shared one scale. Each policy assigns
an auditable zone (`red`, `yellow`, `green`, or `indeterminate`) and a normalized
within-zone priority. Sort deterministically by manual override, zone severity,
aged within-zone priority, arrival time, and exam ID. Log every override.

Knee MRI is normally a non-emergency prioritization task; it must not inherit the
same clinical urgency policy as intracranial hemorrhage or pneumothorax without a
documented workflow rule.

Queue-benefit claims require a discrete-event simulation with declared arrival and
service distributions, baseline comparator, multiple seeds, confidence intervals,
and sensitivity analyses. Synthetic PACS fixtures validate software behavior only.

## 6. Code layout and migration order

Add code to the supported package; do not import production paths from `core/`:

```text
stat_triage/
  contracts.py
  ingestion.py
  calibration.py
  thresholds.py
  metrics.py
  worklist.py
  adapters/{base,rsna_slice,rsna_study,chest_xray,mrnet}.py
  datasets/{rsna,chexpert,nih_cxr14,mrnet}.py
  models/{densenet_multilabel,sequence_aggregator,mrnet_planes,biomedclip}.py
  evaluation/{bootstrap,evaluator,latency}.py
  policy/{head_ct,chest_xray,knee_mri}.py
```

Every training run writes a manifest snapshot, split hashes, configuration,
environment, seed, checkpoint hash, epoch history, calibration parameters,
operating points, predictions, metrics, and failure log. Release creation fails if
any required artifact is absent or internally inconsistent.

## 7. Gated execution order

### Gate 0 — repository baseline

- Commit or otherwise snapshot the current passing state before migration.
- Mark `core/` as legacy and prevent new supported imports from it.
- Add an evidence registry mapping every reported value to its artifact.
- Exit: reproducible 68-test baseline and documented artifact inventory.

### Gate 1 — common infrastructure

- Implement contracts, masked multi-label loss, calibration, threshold locking,
  grouped bootstrap, and release validation.
- Add mathematical unit tests and tiny end-to-end synthetic fixtures.
- Exit: one evaluator can recompute the frozen RSNA metrics from saved predictions.

### Gate 2 — chest vertical slice

- Build a small real-data manifest and preprocessing preview.
- Train/evaluate DenseNet121; then run corrected BiomedCLIP comparison.
- Exit: patient-disjoint held-out predictions, calibration, locked test results,
  latency, and worklist integration.

### Gate 3 — MRNet vertical slice

- Build three-plane manifest and exam loader.
- Train plane-pooling baseline and optional fusion refinement.
- Exit: exam-level locked evaluation and worklist integration.

### Gate 4 — study-level head CT

- Preserve release 0.1 and add study manifests and aggregation baselines.
- Train a sequence aggregator only if it beats the fixed aggregation baseline.
- Exit: patient-disjoint study-level evaluation clearly separated from slice claims.

### Gate 5 — unified product validation

- Run adapters through one ingestion/evidence/worklist pipeline.
- Test corrupt files, missing series/planes, unsupported modalities, duplicate
  requests, deterministic ordering, manual overrides, and audit replay.
- Benchmark CPU/GPU latency and memory by modality.
- Exit: versioned research release and report generated entirely from artifacts.

### Gate 6 — research extensions

- Multi-label conformal prediction with empirical coverage reporting.
- True incremental inference and formal sequential testing.
- Anatomically constrained explanations and counterfactual research.
- Queue simulation and external/domain-shift evaluation.

These extensions cannot block delivery of the three honest modality baselines.

## 8. Practical schedule and next action

On one Kaggle T4, an honest pilot is approximately four focused weeks: common
infrastructure plus CXR, MRNet, study-level CT, then unified validation/reporting.
A one-week deadline can support one additional modality vertical slice, most
sensibly CXR, but not research-level validation of all three modalities.

The immediate implementation task is **Gate 0 followed by Gate 1**. Specifically:
freeze the current state, create the common evidence record, move calibration and
threshold selection into `stat_triage`, and prove that the evaluator reproduces
the locked RSNA result from `test_predictions.csv`. This creates the foundation
needed by CXR and MRI without rewriting evaluation logic three times.
