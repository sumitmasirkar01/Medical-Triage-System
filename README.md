# STAT-Triage AI

STAT-Triage AI is a research prototype for evidence-backed medical-imaging
worklist prioritization. The supported backend routes three independently
validated research adapters: slice-level RSNA head CT, NIH ChestX-ray14, and
three-plane Stanford MRNet knee MRI. It does not perform cross-modality neural
fusion, and the modality-specific scores are not directly comparable.

## Current evidence

The locked RSNA pilot used 2,400 enriched slices with patient-separated train,
validation, calibration, and test partitions. Its EfficientNet-B0 model has six
independent outputs in this order:

```text
epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any
```

The first locked test produced an `any` hemorrhage ROC-AUC of 0.8417 and
sensitivity of 0.8831 at the calibration-selected threshold. A subsequent locked
retrospective study-level evaluation, on a patient-disjoint cohort of 240 held-out
patients, produced ROC-AUC 0.9078 (95% patient-bootstrap CI 0.8679–0.9411) and
sensitivity 0.9565 at the previously locked study threshold. Neither result
establishes prospective or clinical performance.

The NIH ChestX-ray14 adapter has a locked official-test evaluation across four
targets and 2,797 patients. The Stanford MRNet adapter has a locked official-valid
evaluation treated as test across three targets and 120 exams; patient identifiers
are unavailable for that dataset. These modalities use separate endpoints,
thresholds, and evaluation units, so their scores and metrics must not be ranked
against each other as if they were a common probability scale.

The locked evidence bundles are under `artifacts/`. Dataset semantics and
limitations are documented in `docs/DATASET_CONTRACTS.md`. The frozen research
protocol is in `docs/RESEARCH_PROTOCOL.md`, with machine-readable experiment
provenance under `research/`.

## Collaborator setup

For a fresh laptop checkout, use the tested handoff instructions in
`docs/COLLABORATOR_HANDOFF.md`. On Windows with Python 3.12 installed:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_collaborator.ps1
```

This creates an isolated environment, installs the package, verifies the locked
artifact hashes, and runs the supported test suite. It does not download or include
the original medical-image datasets.

## Authoritative report

Generate the report from the locked artifact bundle with:

```powershell
.\.venv\Scripts\python.exe generate_pilot_report.py
```

The generator verifies the bundle manifest and fails if required evidence is
missing or altered. Its default output is:

```text
docs/STAT_Triage_RSNA_Pilot_Report.docx
```

## Tests

The active suite covers all three adapters, evidence bundles, upload boundaries,
modality-safe worklist behavior, release provenance, and the Streamlit render
contract:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Run the research worklist

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app accepts one single-frame CT DICOM instance up to 32 MiB, runs the verified
slice-level adapter, and adds its evidence to the session-local worklist. It
provides explicit override and completion controls with audit history. The safe
command-line boundary in `scripts/infer_multimodal.py` supports CT slices, complete
CT series, frontal chest radiographs, and three-plane knee MRI exams. Every command
returns the same versioned triage envelope and requires human review. The graphical
interface remains intentionally CT-only.

See `docs/CLI_INFERENCE.md` for commands, output fields, resource limits, and exit
codes.

Generate a complete de-identified fixture and rehearse the workflow with:

```powershell
.\.venv\Scripts\python.exe -m scripts.create_demo_dicom --output sample_dicoms\DEMO_CT.dcm
.\.venv\Scripts\python.exe -m scripts.demo_smoke --dicom sample_dicoms\DEMO_CT.dcm
```

See `docs/DEMO_REHEARSAL.md` for the interactive demonstration sequence and claims
that remain outside the evidence.

The checkpoint-backed RSNA adapter is implemented in `stat_triage/rsna.py`. It
verifies checkpoint provenance, applies the recorded preprocessing contract, and
returns named slice-level evidence. `stat_triage/worklist.py` consumes that evidence
with elapsed-time aging, deterministic ordering, explicit overrides, completion,
and retained audit history. `app.py` is the supported CT-only Streamlit entry
point. Older dashboards, benchmarks, reports, and incompatible tests are retained
under `legacy/` for reference and are excluded from the supported test suite.

## Research status

This software is for research and demonstration. It is not a medical device and
must not be used to make clinical decisions. Scores from the enriched pilot are
not population-calibrated probabilities.
