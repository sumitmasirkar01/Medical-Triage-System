# STAT-Triage AI execution plan

Last updated: 2026-10-12

The validated code, mathematics, checkpoint, and execution audit is maintained in
[MULTIMODAL_VALIDATED_EXECUTION_PLAN.md](docs/MULTIMODAL_VALIDATED_EXECUTION_PLAN.md).
It is the controlling plan for new implementation work; this file retains the
earlier execution history and milestone record.

## Objective and scope

Deliver a reproducible **multimodal** research prototype with three independently
validated adapters: RSNA Head CT, NIH/CheXpert chest X-ray, and MRNet knee MRI.
Each adapter executes a documented model and produces a common evidence contract
consumed by one auditable priority worklist. Demonstrate software correctness
separately from model performance. Every numerical claim in the final report must
link to a saved evaluation artifact.

Head CT is the completed reference vertical slice and establishes the engineering
and evidence standard for the other modalities. It is not the final product. This
plan does not promise clinical deployment, a particular accuracy, conformal
coverage, or a compute reduction.

## Multimodal end state

```text
RSNA Head CT adapter  ─┐
NIH/CheXpert CXR      ─┼─> common evidence contract ─> policy-normalized worklist
MRNet knee MRI        ─┘
```

The adapters share identity, provenance, failure, audit, and worklist interfaces,
but they do not share raw labels, preprocessing, checkpoints, thresholds, or
calibration. Raw scores from different models are not directly comparable. The
worklist consumes modality-specific operating decisions and explicit workflow
severity policies rather than treating every model score as the same probability.

Required targets:

- **Head CT:** five hemorrhage subtypes plus `any`, six independent binary outputs.
- **Chest X-ray:** pneumothorax, pleural effusion, consolidation, and atelectasis as
  independent findings; NIH and CheXpert results remain separately identifiable.
- **Knee MRI:** abnormality, ACL tear, and meniscal tear as three exam-level binary
  targets aggregated across axial, coronal, and sagittal planes.

Each modality must independently satisfy: documented dataset contract, leakage-safe
splits, checkpoint provenance, reproducible preprocessing, held-out predictions,
threshold/calibration provenance, confidence intervals, latency measurement, and
an adapter-to-worklist integration test.

## Decisions to make together

| Decision | Current working assumption | Needed by |
| --- | --- | --- |
| Submission/demo deadline | One week, confirmed by user; exact submission time pending | Scheduling |
| Compute budget | Kaggle Tesla T4 (14.6 GB VRAM) confirmed; local laptop GPU remains outside the measured run | M2 |
| Dataset and access | RSNA ICH 2019 stage 2 confirmed in Kaggle; NIH, CheXpert, and MRNet are documented extensions outside the one-week baseline | M2 |
| Checkpoint provenance | Fresh EfficientNet-B0 pilot trained from torchvision ImageNet weights; locked checkpoint SHA-256 `4a15c605bb1481de7f12a4881f9c9b5adb36d9386ecac0aa6eb3ae7c3ae805b1` | M2 |
| Intended target | CT first: retain RSNA's five subtype labels plus `any`; distinguish slice evaluation from derived study-level evaluation | M2 |
| Demonstration scope | Local research demonstration with explicit synthetic mode | M1 |

If data access is delayed, finish engineering with synthetic fixtures and controlled
model substitutes. Record model evaluation as pending; do not substitute synthetic
accuracy for held-out performance.

## Dataset contracts now identified

Dataset identities were supplied by the user. Access paths, exact release/subset,
checkpoint provenance, and past evaluation artifacts still need inspection. Three
modalities involve four imaging datasets because NIH and CheXpert are separate
sources. See [DATASET_CONTRACTS.md](docs/DATASET_CONTRACTS.md) for sources, labels,
input rules, and concrete migration requirements.

Immediate priorities from the dataset review:

- RSNA: restore the missing intraventricular label and represent five subtypes plus
  `any` as multi-label targets. Do not append IVH to a five-output checkpoint without
  an explicit architecture/weight migration and appropriate training.
- Separate slice-level RSNA benchmark results from study-level aggregation results.
- Chest: keep NIH and CheXpert cohorts separate, define CheXpert uncertainty handling,
  and use per-finding scores rather than a softmax across coexisting findings.
- MRI: abnormality, ACL tear, and meniscal tear are three binary targets; the current
  random four-way head is not a validated MRNet model.
- BiomedCLIP pretraining is backbone provenance, not evidence of project evaluation.
- Revisit the categorical conformal implementation when adopting multi-label outputs;
  its current prediction-set semantics cannot simply be carried across unchanged.

## Committed one-week scope and daily schedule

The user confirmed a one-week deadline after the initial plan was drafted. The
schedule below takes precedence over the broader milestone estimates. The full
milestones remain a backlog; completing all of them is not a seven-day commitment.

Submission target: a working single-modality research demonstration, a reproducible
evaluation on a clearly specified available cohort, and a report that matches the
evidence. Start with RSNA Head CT, subject to data access. Verify the exact release
and available labels before locking the cohort. Use a modest explicit
subset if full-cohort processing cannot finish; report its size and selection rule.

| Day | Assistant work | User contribution | Evidence required before moving on |
| --- | --- | --- | --- |
| 1 | Repair contracts, audit/migrate tests and supported entry point; label synthetic results | Supply dataset names/links, access method, exact deadline | Enqueue regression passes; list of remaining integration failures; dataset choice recorded |
| 2 | Build dataset manifest, patient-disjoint splits, preprocessing checks; verify checkpoint provenance and model input contract | Enable dataset/notebook access; confirm available accelerator | Real samples decode; labels and identifiers verified; no split overlap; usable baseline selected |
| 3 | Run the simplest feasible baseline; cache predictions/features; fit only a small head if justified and resources allow | Run prepared Kaggle notebook when account access is needed and return logs/artifacts | Actual image-derived predictions saved with model/configuration provenance |
| 4 | Evaluate fixed thresholds and metrics; measure end-to-end latency; add calibration only if a sufficient separate cohort is available | Review failure examples and intended presentation scope | Saved metrics, cohort counts and uncertainty; no test-set tuning |
| 5 | Complete queue aging/override behavior; connect supported demo path; run integration regressions | Try the demo workflow and identify presentation blockers | Supported suite passes; normal/error paths and ranking behavior work |
| 6 | Generate report tables from saved evidence; update README and limitations; inspect rendered report | Review narrative and submission requirements | All reported figures trace to artifacts; demonstration rehearsed |
| 7 | Fix blockers only; reproduce essential commands; package environment/configuration/results | Final rehearsal and submission | Reproducible handoff with documented limitations and no unverified claims |

Scope cuts: no new MRI pipeline, no streaming redesign, no architecture search,
no counterfactual-validation project, and no deployment infrastructure this week.
X-ray is a possible alternative first modality if that is the accessible labelled
dataset, not an additional mandatory implementation.

Decision deadlines:

- End of Day 1: identify competition, label availability, and compute access.
- End of Day 2: if a compatible validated checkpoint is unavailable, choose a feasible
  pretrained-feature baseline with a small trained head. Existing `.pt` files alone
  do not establish medical training or validity.
- End of Day 3: if real-data execution remains blocked, finish an explicitly synthetic
  engineering demonstration and report empirical model validation as incomplete.
- End of Day 4: freeze model/threshold selection. Report weak results honestly;
  do not tune repeatedly on the test cohort to improve the final table.
- End of Day 5: freeze features and reserve the last two days for evidence and delivery.

Hardware: a Windows hardware query was denied in this session, so GPU specifications
remain unknown. Choose batch sizes only after checking the actual execution runtime.

## Locked RSNA pilot result

Status: first test evaluation completed and the Kaggle bundle is present under
`artifacts/rsna_baseline/rsna_baseline_bundle`. Its manifest and checkpoint hash
were verified during report generation. These are slice-level pilot results, not 3D study-level or
clinical-deployment results.

- Source label index: 752,803 unique slices; 107,933 `any` positives (14.34%).
- Pilot selection: 2,400 slices, deliberately enriched to 800 positive and 1,600
  negative slices before splitting.
- Patient-separated split: 1,682 train / 232 validation / 244 calibration / 242 test;
  zero patient, study, or series overlap across splits.
- Model: ImageNet-pretrained EfficientNet-B0, six sigmoid outputs, three HU windows
  at 224 x 224; classifier trained for 2 epochs and upper backbone for 6 epochs.
- Selection: epoch 4, validation `any` ROC-AUC 0.8553. Later epochs overfit.
- Locked operating threshold: 0.253861, selected on calibration for at least 90%
  sensitivity. Calibration sensitivity 0.9114 and specificity 0.5515.
- First locked test: ROC-AUC 0.8417 (patient-bootstrap 95% CI 0.7855-0.8914),
  sensitivity 0.8831 (0.8068-0.9444), specificity 0.5091 (0.4364-0.5864),
  NPV 0.9032 (0.8404-0.9574), with TP/TN/FP/FN = 68/84/81/9.
- Test PPV 0.4564 and accuracy 0.6281 reflect the enriched pilot prevalence and
  must not be presented as population estimates.
- Batched cohort processing: 2.061 seconds for 242 slices (8.52 ms/image average),
  including DICOM decode, preprocessing, transfer, and inference. This is throughput,
  not single-study clinical latency.
- Subtype AUCs are exploratory: epidural 0.5990 (3 positives), intraparenchymal
  0.9146 (24), intraventricular 0.9064 (17), subarachnoid 0.7569 (30), and subdural
  0.8429 (35). Sparse subtype results are not validated claims.

The locked test must not be used for further model, threshold, preprocessing, or
cohort-selection decisions. Any next model iteration requires a new untouched test
cohort or must be reported as post-test exploratory work.

## Baseline findings

- Gateway enqueueing passes fields absent from the queue dataclass.
- Five collection errors were observed with `.venv` pytest before its max-failure limit.
- Several `run_*` benchmark functions under `tests/` are not pytest tests.
- CLI constructors and CT embedding dimensions disagree with current interfaces.
- Queue scores do not refresh with elapsed time; overrides do not affect sorting.
- Synthetic benchmark outputs are presented under clinical dataset names.
- Calibration uses fabricated predictions; MRI has no trained checkpoint loader.
- CT stopping statistics are computed after full-volume model execution.
- Existing user changes are present. Preserve them and make targeted edits.

## Codebase audit and immediate execution path

Audit completed: 2026-09-30. The repository currently contains two generations of
the project. The supported evidence path is the six-output RSNA EfficientNet pilot
and its saved artifacts. The older unified prototype contains useful experiments,
but its CT, chest, and MRI adapters do not match the documented dataset contracts
and must not be used to make performance claims.

Current executable state:

- `pytest --collect-only -q` discovers four tests and stops on five collection
  errors caused by removed interfaces (`ClinicalCTPreprocessor`, `TriageGateway`,
  `TriageRecord`, `HeadCTTriPathwayNet`, and related helpers).
- The four currently discoverable calibration/gateway tests pass. Most newer files
  expose `run_*` scripts and therefore are not pytest regression tests.
- `dashboard.py`, `run_triage_gateway.py`, `benchmark_pipeline.py`, and parts of
  `cli_workstation.py` use incompatible generations of the API. The dashboard also
  reads output attributes not defined by `ModalityTriageOutput`.
- The legacy CT adapter uses a five-way softmax and omits intraventricular hemorrhage;
  the evaluated RSNA model uses six independent sigmoid outputs. They are different
  models and their checkpoints are not interchangeable.
- Chest uses mutually exclusive softmax scores for findings that may coexist. MRI
  uses an untrained random four-class head. Both remain research placeholders.
- Conformal calibration inside adapters is generated from synthetic probabilities.
  The claimed 99% coverage is not established on the clinical cohorts.
- SPRT is applied after the complete CT feature sequence has already passed through
  the transformer, so the reported percentage is simulated downstream work and is
  not measured end-to-end compute savings.
- Queue aging is only evaluated at enqueue time. A clinician override changes the
  displayed zone but does not establish a persistent manual ordering policy.
- The dependency file is an environment snapshot rather than a minimal reproducible
  application specification, and optional model/runtime dependencies are not isolated.

The next supported vertical slice is:

```text
RSNA DICOM slice -> verified three-window preprocessing -> locked EfficientNet-B0
-> six named probabilities -> locked `any` threshold -> evidence record
-> deterministic queue policy -> Streamlit research worklist
```

Execution order for the remaining week:

1. **Stabilize contracts and tests.** Move obsolete tests and entry points into a
   clearly marked legacy/experimental area, convert meaningful checks into pytest,
   and make collection clean. Define one evidence record with patient/study/image
   identity, six probabilities, threshold decision, model hash, and timestamps.
2. **Implement the evaluated RSNA adapter.** Load only the verified checkpoint,
   reproduce the Kaggle HU windows and normalization exactly, validate label order,
   and fail clearly on incompatible weights. Test preprocessing and inference with
   small local fixtures; do not consume the locked test set during development.
3. **Make the queue behavior explicit.** Refresh aging from an injectable clock,
   use deterministic tie-breaking, preserve manual overrides and reasons, and test
   enqueue, refresh, override, and completion. Treat its score as workflow policy,
   not a calibrated probability.
4. **Replace the dashboard path.** Build a small CT-only research worklist on the
   tested service layer. Cache the checkpoint as a Streamlit resource, keep each
   user's queue in session state, and show provenance, threshold, uncertainty limits,
   and errors. Remove unsupported 99% safety and compute-saving statements.
5. **Reproduce and package.** Add a minimal setup/README, run the supported tests,
   execute a local DICOM-to-worklist smoke test, regenerate the report from the
   unchanged bundle, and rehearse the demo.

Deferred until after the CT vertical slice: study-level CT aggregation and a new
untouched evaluation cohort; real NIH/CheXpert adapters; the MRNet loader/model;
conformal validation; causal explanations; and a causal/streaming early-exit model.
The locked 242-slice test remains report-only and cannot guide these changes.

## Milestones and acceptance criteria

### M1 — Restore a trustworthy engineering baseline

Dependencies: none. Initial planning estimate: 3–5 focused working days, subject to
the entry-point migration scope. Current status: in progress.

Tasks:

- [ ] M1.1 Align `ModalityTriageOutput`, `DynamicQueueItem`, and gateway consumers;
  add a regression test for enqueue → sorted worklist → override state.
- [ ] M1.2 Inventory old entry points and tests; migrate supported behavior to current
  interfaces. Preserve obsolete experiments with clear documentation. Do not hide
  failing tests or add aliases that only move failures to runtime.
- [x] M1.3 Turn meaningful `run_*` checks into discoverable tests. Separate lightweight
  tests from explicit model-download, performance, and memory-stress experiments.
- [x] M1.4 Repair CLI keyword arguments, batch dimensions, and patient/study identity
  propagation. Add a controlled end-to-end test without external model downloads.
- [x] M1.5 Implement elapsed-time refresh with an injectable clock; deterministic
  ties; explicit promotion/demotion rules; retained override reasons. Test these
  behaviors before connecting the UI.
- [x] M1.6 Declare runtime and development dependencies, isolate optional model
  imports, document setup, and verify installation in a clean environment.
- [ ] M1.7 Add clear demo/synthetic labels to benchmarks and reports immediately;
  remove unsupported claims while real evaluation remains pending.

Gate: full pytest collection succeeds; the supported default suite passes without
network/model downloads; a synthetic study traverses ingestion → controlled
inference → queue; enqueue, aging, ties, and overrides have regression coverage.
Record exact commands and results. This gate proves software behavior, not accuracy.

### M2 — Establish real data and a reproducible CT baseline

Dependencies: M1 contracts, dataset/checkpoint decisions. Estimate: 4–7 working days
after access is available; downloading and training time are additional.

- [ ] M2.1 Document dataset source, permitted use, label definitions, exclusions,
  patient/study identifiers, and class counts. Keep patient data out of source control.
  Dataset identities and proposed contracts are recorded; manifests and access remain pending.
- [ ] M2.2 Create patient-disjoint train, tuning/validation, calibration, and locked
  test manifests. Verify no patient or duplicate-study overlap.
- [ ] M2.3 Consolidate DICOM preprocessing: group by study/series; use spatial metadata
  for ordering; verify rescale/windowing and expected model normalization; reject
  mixed or unsupported series with actionable errors. Test representative fixtures.
- [x] M2.4 Select a documented trained baseline. Record checkpoint hash, architecture,
  label map, input transform, training source, and license. Missing/incompatible
  weights must fail explicitly outside demo mode.
- [ ] M2.5 Define long-volume handling and batch/padding behavior. Ensure slice
  exclusions and aggregation policy are documented and tested.
- [ ] M2.6 Save study-level predictions with anonymized IDs, label references,
  configuration, model hash, split name, and run identifier.

Gate: rerunning a fixed cohort with the same model/configuration reproduces results
within documented numerical tolerance. No test examples influence model selection.

### M3 — Replace synthetic performance claims with measured evidence

Dependencies: M2. Estimate: 3–5 working days plus inference time.

- [ ] M3.1 Use actual image-to-model predictions in the evaluator. Preserve synthetic
  experiments as separately named engineering demonstrations.
- [ ] M3.2 Specify metrics before evaluating the locked test set: ROC-AUC, PR-AUC,
  sensitivity, specificity, predictive values, and confusion counts. Select operating
  thresholds on tuning data only; report attained test operating points.
- [ ] M3.3 Add patient-level confidence intervals, class counts, and handling of
  undefined metrics. Avoid selecting thresholds on the reported test cohort.
- [ ] M3.4 Fit probability/conformal calibration using real held-out outputs. Report
  target coverage separately from observed coverage, set size, and subgroup results.
  Record assumptions and failure cases; do not treat target alpha as measured safety.
- [ ] M3.5 Measure DICOM decode, preprocessing, feature extraction, aggregation,
  postprocessing, and total latency. Record hardware, warmup, batch size, and p50/p95.
- [ ] M3.6 Export machine-readable metrics and prediction artifacts consumed by reports.

Gate: a documented command regenerates the metrics from saved inputs/configuration;
metrics have provenance and uncertainty; no fabricated predictions or hardcoded
performance values appear in real evaluation results.

### M4 — Test whether advanced methods improve the baseline

Dependencies: M3 baseline. Estimate: 4–8 working days; optional if deadline is short.

- [ ] M4.1 Compare simple slice aggregation with the axial transformer on identical
  splits. Align training loss, label semantics, and inference probability conversion.
- [ ] M4.2 Decide whether streaming is technically appropriate. Full-volume attention
  uses future slices; a streaming version needs a compatible incremental/causal design.
- [ ] M4.3 Compare full inference with actual sequential execution. Measure work
  skipped, latency, false halts, missed positives, and stopping positions. Separate
  simulated savings from measured end-to-end savings.
- [ ] M4.4 Evaluate explanations using suitable evidence. Do not call feature norms
  attention maps or optimized perturbations validated lesion segmentation.
- [ ] M4.5 Simulate queue policies with identical arrivals and service times; report
  waiting-time distributions by class and override behavior, not just sorting speed.

Gate: retain a feature only with reproducible evidence of benefit and documented
tradeoffs. An unsuccessful experiment is a valid reportable result.

### M5 — Integrate the demonstration and generate the report

Dependencies: M1–M3; include M4 only if completed. Estimate: 3–5 working days.

- [x] M5.1 Connect the supported CLI/UI workflow to the same tested service path.
  Display model availability, research/demo status, processing failures, and study identity.
- [ ] M5.2 Add queue completion/removal, persistent worklist/override history if required
  by the demo, and tests for reruns/restarts. Confirm scope before adding infrastructure.
- [x] M5.3 Define one authoritative evidence-backed pilot report generator.
  Populate tables from evaluation artifacts and visibly identify missing results.
- [ ] M5.4 Update README architecture, setup, limitations, and commands. Remove outdated
  model descriptions and unsupported production/regulatory assertions.
- [x] M5.5 Render and inspect the six-page pilot report. Demo rehearsal remains pending.

Gate: another person can reproduce the demonstration and trace every result in the
report to a run artifact. Clearly distinguish engineering, synthetic, and held-out results.

### M6 — Complete the multimodal system

Dependencies: M5. Estimate only after dataset and checkpoint discovery.

- [ ] M6.1 Generalize `RSNAEvidence` into a versioned common evidence envelope while
  preserving modality-specific label maps, endpoint names, thresholds, and provenance.
- [ ] M6.2 Add chest X-ray using independent multi-label outputs, explicit CheXpert
  uncertain/missing-label handling, patient-separated splits, held-out evaluation,
  and an adapter-to-worklist test. Report NIH and CheXpert separately.
- [ ] M6.3 Add MRNet using its three-plane exam loader and three binary exam targets.
  Require a trained checkpoint, exam-safe splits, held-out evaluation, and an
  adapter-to-worklist test.
- [ ] M6.4 Define modality-specific workflow severity mappings and evaluate the unified
  queue under common arrival/service simulations. Do not compare raw scores directly.
- [ ] M6.5 Extend Streamlit and the release manifest/report to show modality-specific
  evidence, provenance, limitations, and errors through the same service path.

Gate: each modality independently meets the M2/M3 standards before being described
as evaluated in the unified demonstration.

## Detailed multimodal execution plan

Target: a production-structured research prototype with three real-data modality
adapters. The CT pilot remains frozen while chest X-ray and knee MRI pass the same
provenance and held-out-evaluation gates. If a modality misses a gate, the software
may still be demonstrated in an explicitly labelled engineering mode, but no model
performance claim will be made for it.

### Phase A — Freeze CT and generalize the contract

Objective: make the current CT implementation one plugin of a multimodal system
without changing its checkpoint, threshold, predictions, or locked metrics.

Tasks:

1. Define a versioned `EvidenceEnvelope` containing:
   - case, patient, study/exam, series, and source-instance identity;
   - modality, dataset, endpoint, label order, scores, thresholds, and decisions;
   - workflow severity and the policy rule that produced it;
   - model architecture, checkpoint hash, preprocessing version, and run ID;
   - inference timestamp, latency fields, warnings, and limitations.
2. Define a `ModalityAdapter` protocol with `validate_input`, `preprocess`,
   `predict`, and `provenance` responsibilities.
3. Add an RSNA-to-envelope adapter while retaining `RSNAEvidence` compatibility.
4. Update the worklist to consume the common envelope. Keep scores inside the
   modality payload; sort on explicit workflow severity plus aging, not raw scores.
5. Add schema serialization/deserialization and version rejection tests.

Artifacts:

- `stat_triage/evidence.py`
- `stat_triage/adapters/base.py`
- CT compatibility migration and tests
- JSON examples for each planned modality

Gate A:

- Existing CT outputs and threshold decisions are unchanged.
- The full CT suite passes through the common envelope.
- Unknown schema versions, missing identity, unordered labels, and non-finite scores
  fail before entering the worklist.

### Phase B — Chest X-ray real-data pilot

Objective: support four coexisting thoracic findings using NIH ChestX-ray14 and
CheXpert without treating diseases as mutually exclusive.

Data contract:

- Targets: `pneumothorax`, `pleural_effusion`, `consolidation`, `atelectasis`.
- Keep NIH and CheXpert source labels and cohort identities separate.
- Restrict the first pilot to frontal AP/PA images with documented exclusions.
- Split by patient. Use train, validation/model-selection, calibration, and locked
  test cohorts. Do not tune prompts, thresholds, or transforms on locked tests.
- Preserve CheXpert positive, negative, uncertain, and unmentioned states with an
  observation mask. Exclude uncertain/unmentioned labels from evaluation for that
  target; record the chosen masked-loss policy for training.
- Map NIH `Effusion` to project `pleural_effusion` while retaining the source label.
  Do not silently equate `Pneumonia`, `Infiltration`, and `Consolidation`.

Baseline experiments:

1. Frozen BiomedCLIP zero-shot prompts, fixed before test evaluation.
2. Frozen BiomedCLIP image embeddings with a four-output sigmoid linear head.
3. Use a conventional ImageNet baseline only if it is needed to interpret whether
   BiomedCLIP adds value; do not expand into architecture search.

Training and selection:

- Use masked binary cross-entropy and class weights computed from training only.
- Select one model using validation macro PR-AUC with pneumothorax sensitivity as a
  reported safety-oriented secondary measure.
- Select per-finding thresholds on the calibration split. Define the worklist
  endpoint separately, initially `urgent_thoracic_finding`, with an explicit mapping
  that gives pneumothorax greater workflow severity than the other pilot targets.

Evaluation:

- Per finding: ROC-AUC, PR-AUC, sensitivity, specificity, PPV, NPV, counts, and
  patient-bootstrap 95% confidence intervals.
- Multi-label: macro/micro ROC-AUC and PR-AUC, observed-label coverage, and missing
  label counts.
- Report NIH and CheXpert results separately. An NIH evaluation may serve as an
  external-cohort experiment only if no NIH information influenced model selection.
- Measure decode, preprocessing, embedding, head, and total batch/single-image time.

Required artifacts:

- dataset audit and patient-split manifests;
- run configuration, label policy, prompt file, checkpoint hash, and history;
- calibration/test predictions and metrics JSON;
- preprocessing preview and failure summary;
- `ChestXRayAdapter` plus inference and worklist integration tests.

Gate B:

- Real radiographs reach the checkpoint through the documented transform.
- No patient overlap exists across splits.
- Four independent scores and decisions reproduce from saved artifacts.
- The locked test is evaluated once after model and thresholds are frozen.

### Phase C — MRNet knee MRI real-data pilot

Objective: support exam-level abnormality, ACL tear, and meniscal tear using all
three anatomical planes.

Data contract:

- Targets: `abnormal`, `acl`, `meniscus`; three independent binary outputs.
- Join axial, coronal, and sagittal NumPy series by exam ID.
- Keep all planes and slices from an exam in one split. Use patient separation if
  recoverable metadata permits it; otherwise disclose exam-level split limitations.
- Document missing/corrupt plane handling and never replace a missing plane silently.

Baseline model:

1. ImageNet-pretrained 2D encoder applied to individual grayscale slices replicated
   to three channels.
2. Pool slice features within each plane using max or attention pooling chosen on
   validation only.
3. Fuse plane-level predictions with a documented fixed or learned validation-only
   rule and produce three sigmoid exam outputs.
4. Compare against a simple per-plane/max-pooling baseline before retaining a more
   complex attention model.

Training and selection:

- Use three masked/weighted binary losses computed from training statistics.
- Select using validation macro PR-AUC while reporting all three targets separately.
- Reserve separate calibration and locked test exams. Do not reuse MRNet validation
  as both model-selection and final-test evidence.
- Define workflow semantics honestly: MRNet findings indicate review priority in
  this project and are not assumed to represent emergency instability.

Evaluation:

- Per target ROC-AUC, PR-AUC, operating-point metrics, counts, and exam-bootstrap
  confidence intervals.
- Ablate single-plane versus three-plane fusion on validation data.
- Measure loading, slice encoding, plane aggregation, fusion, and total exam latency.

Required artifacts:

- exam/plane audit and split manifests;
- training configuration, checkpoint, history, calibration/test predictions;
- per-target metrics and confidence intervals;
- representative preprocessing contact sheet;
- `MRNetAdapter` plus missing-plane, inference, and worklist integration tests.

Gate C:

- All three planes are joined deterministically by exam.
- A trained checkpoint reproduces three named exam scores.
- Locked metrics derive from saved exam-level predictions.
- Missing planes and incompatible arrays fail with actionable errors.

### Phase D — Cross-modality worklist policy

Objective: combine modality evidence without pretending raw scores are comparable.

Tasks:

1. Introduce an explicit policy output: `RED`, `YELLOW`, or `GREEN`, a bounded
   within-zone ranking score, rule/version identifier, and human-readable reason.
2. Create modality policy tables using calibration-split decisions:
   - CT: threshold-positive `any` hemorrhage receives the CT urgent policy;
   - X-ray: pneumothorax and the composite urgent endpoint use the CXR policy;
   - MRI: abnormal/ACL/meniscus decisions use a non-emergency review policy unless
     a clinically justified urgency rule is documented.
3. Apply wait-time aging inside each effective zone. Clinician overrides remain the
   highest authority and must preserve actor, reason, previous zone, and timestamp.
4. Simulate identical arrival streams and service times for FIFO and candidate
   policies. Report waiting-time distributions, starvation, override effects, and
   modality composition by zone.

Gate D:

- Changing a raw score in one modality cannot silently alter another modality's
  policy scale.
- Ordering is deterministic and refreshes with an injected clock.
- Queue simulations use saved scenarios and produce reproducible JSON results.

### Phase E — Unified application

Objective: expose one coherent system with modality-specific input and evidence.

Tasks:

1. Add an explicit modality selector and separate upload contracts:
   - CT: one supported slice initially, followed by study upload after study-level work;
   - CXR: one frontal radiograph/DICOM or supported image file;
   - MRI: one MRNet exam containing three plane arrays.
2. Cache each model independently and load it only when its modality is used.
3. Render shared worklist fields consistently and modality findings separately.
4. Display checkpoint/dataset provenance, threshold source, evaluation level, and
   limitations for the selected modality.
5. Preserve session-local research behavior for this release; do not imply PACS,
   authentication, or persistent clinical audit integration.
6. Test valid uploads, wrong-modality inputs, malformed files, duplicates, overrides,
   completion, and mixed-modality ordering.

Gate E:

- A valid example from each modality traverses upload to audited completion.
- A failure in one adapter does not prevent the other adapters from loading.
- The UI never displays unavailable or engineering-only results as validated.

### Phase F — Evidence, report, and release

Tasks:

1. Extend the release manifest with separate CT, CXR, and MRI evidence sections.
2. Generate modality-specific metric tables directly from locked JSON artifacts.
3. Add a unified-system section only for software integration and queue simulation;
   do not pool accuracy metrics across unrelated tasks.
4. Run clean-environment installation, the supported suite, all three CLI rehearsals,
   Streamlit interaction tests, artifact hash verification, and rendered report QA.
5. Tag release maturity per modality: `validated_pilot`, `engineering_only`, or
   `unavailable`. The overall system cannot claim three evaluated modalities unless
   all three pass their respective gates.

Gate F:

- Every numerical claim traces to a saved prediction or configuration artifact.
- Another person can install, reproduce each demo, and identify the evidence maturity
  of every modality.

## Time-boxed schedule from 2026-10-01

This is an aggressive pilot schedule using the Kaggle T4. It targets research-level
pilots, not equal-depth clinical validation.

| Day | Primary work | Exit evidence |
| --- | --- | --- |
| Oct 1 | Phase A common evidence/adapter contract | CT behavior unchanged; schema and worklist tests pass |
| Oct 2 | CXR dataset audit, patient splits, preprocessing, BiomedCLIP preflight | Real images/labels verified; leakage check and batch inference pass |
| Oct 3 | CXR baseline training/selection, calibration, locked evaluation | Saved checkpoint, predictions, metrics, confidence intervals |
| Oct 4 | MRNet audit, exam splits, three-plane loader and preflight | Plane joins, labels, exclusions, and batch shapes verified |
| Oct 5 | MRNet baseline training/selection, calibration, locked evaluation | Saved checkpoint, exam predictions, metrics, confidence intervals |
| Oct 6 | Cross-modality policy, unified app, mixed-modality tests | Three examples traverse one audited worklist |
| Oct 7 | Report, clean install, rehearsal, release manifest and packaging | Reproducible multimodal handoff or explicit maturity labels |

Schedule controls:

- Freeze each modality's model, preprocessing, prompts, and thresholds before its
  locked test evaluation.
- Prefer a smaller declared pilot cohort over an unfinished full-dataset run.
- If a real-data gate cannot be passed by Oct 6, ship that adapter as
  `engineering_only`; never substitute synthetic metrics.
- Do not add conformal prediction, SPRT, counterfactual explanations, or architecture
  search until all three basic adapters and evidence pipelines are complete.

## How we execute together

- Assistant: implement bounded changes, run relevant checks, report failures honestly,
  and update this plan with evidence after each task.
- User: provide deadline, compute/data availability, and intended demonstration scope;
  participate in model/dataset and final presentation decisions.
- Work one acceptance criterion at a time. Avoid large architecture rewrites while
  stabilizing the project. Do not declare a milestone complete because a subset passes.
- Default verification uses local fixtures; downloads, full training, and access-bound
  dataset work are explicit steps once their resources and scope are established.
- If time is short: finish M1–M3 and a reduced M5; defer M4/M6. Do not lower evidence
  standards to preserve a feature count.

## Execution log

| Date | Task | Result |
| --- | --- | --- |
| 2026-09-30 | Planning | Recorded scope, dependencies, gates, decisions, and execution order. |
| 2026-09-30 | M1.1 (partial) | Repaired gateway construction against the queue schema; added shared score/zone aliases and declared override state. Focused regression command: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_gateway_contract.py` — 3 passed. Full caller migration and full-suite validation remain pending. |
| 2026-09-30 | Scope revision | User confirmed one week, ROG Strix G15, and Kaggle competition datasets. Added seven-day delivery schedule and scope cuts; exact dataset remains pending. |
| 2026-09-30 | Dataset clarification | User identified RSNA ICH 2019, NIH ChestX-ray14, CheXpert, MRNet, and BiomedCLIP backbone provenance. Checked primary documentation and recorded label/input contracts in `docs/DATASET_CONTRACTS.md`; actual access and run artifacts remain pending. |
| 2026-09-30 | M2/M3 RSNA pilot | Indexed 752,803 labels, constructed a 2,400-slice patient-separated pilot, trained a six-output EfficientNet-B0 baseline on a Kaggle T4, locked the epoch-4 checkpoint and calibration threshold, and performed the first test evaluation. Metrics are recorded above; Kaggle artifacts must be transferred before report generation. |
| 2026-09-30 | M5 evidence report | Verified all transferred artifacts against `bundle_manifest.json`, generated the authoritative six-page RSNA pilot report from saved evidence, exported it with Microsoft Word, and visually inspected every rendered page after two layout corrections. |
| 2026-09-30 | M1.2 cleanup stage 1 | Moved incompatible applications, obsolete tests, synthetic benchmark scripts, and superseded report generators under `legacy/`; added a codebase inventory and truthful root README; ignored reproducible QA output; reduced the dependency surface; and constrained pytest to the supported test directory. Verification: full collection succeeds and `.\\.venv\\Scripts\\python.exe -m pytest -q` reports 4 passed. |
| 2026-09-30 | M2.4 RSNA adapter | Added the supported six-output EfficientNet-B0 inference adapter and immutable evidence record. It verifies the locked SHA-256, architecture, label order, epoch, operating endpoint, and threshold; reproduces the saved HU-window/resize/normalization configuration; and loads weights strictly. Three tests exercise preprocessing, provenance, and real checkpoint inference on a synthetic CT fixture. |
| 2026-09-30 | M1.5 worklist service | Added a deterministic service that consumes RSNA evidence, refreshes elapsed-time aging through an injectable clock, sorts by effective zone/score/arrival/ID, retains explicit overrides and audit events, rejects duplicates, and completes cases without deleting their history. Five focused policy tests and an adapter-to-worklist integration assertion pass. |
| 2026-09-30 | M5.1 CT worklist app | Added the supported Streamlit entry point for single-slice DICOM ingestion, six-score evidence display, active worklist ranking, clinician overrides, completion, audit history, and model provenance. Model loading is resource-cached and worklist state is session-local. Headless server startup and Streamlit AppTest rendering complete without application exceptions. |
| 2026-09-30 | M5 demo hardening | Added strict CT identity/physics/pixel validation and actionable upload failures; generated a complete de-identified demo DICOM; added a repeatable CLI rehearsal covering inference, routing, override, restoration, completion, and retained audit events; and documented the interactive demonstration and prohibited claims. The stricter contract correctly rejects the older incomplete sample fixtures. |
| 2026-10-01 | M1.6 reproducible install | Added `pyproject.toml`, exact runtime/development requirement files, bundle-path configuration, and installation documentation. Built wheel `stat_triage_ai-0.1.0-py3-none-any.whl`, installed it into a clean Python 3.12 environment, loaded the verified checkpoint from outside the repository through `STAT_TRIAGE_BUNDLE_DIR`, and obtained a clean `pip check`. Moved the final synthetic/legacy `core` tests out of the supported suite after clean installation exposed their undeclared optional BiomedCLIP import. |
| 2026-10-01 | M5 release preparation | Expanded Streamlit AppTest coverage to execute generated-DICOM upload, checkpoint inference, RED routing, override to GREEN, override clearing, and completion across reruns. Added a deterministic release manifest containing locked evidence, report/source hashes, commands, and limitations; added traceability coverage and a submission/demo checklist. The supported suite now contains 15 tests. |
| 2026-10-20 | Phase B NIH CXR locked pilot | Verified a patient-disjoint 112,120-image manifest, trained and selected an epoch-4 DenseNet121 on validation macro AP, fitted four affine logistic calibrators and sensitivity-constrained thresholds on 8,831 calibration images, and evaluated 25,596 official-test images with 2,000 patient-grouped bootstrap replicates per target. Transferred evidence hashes and point estimates reproduce locally. Added a verified four-output adapter and an explicit pneumothorax-first research worklist policy. |

Next task after M1.1: inventory and migrate obsolete test/entry-point interfaces (M1.2).
