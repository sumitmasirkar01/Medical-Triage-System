# STAT-Triage AI

STAT-Triage AI is a research prototype for evidence-backed medical-imaging
worklist prioritization. The current supported evaluation is a **slice-level
RSNA intracranial hemorrhage pilot**. Chest X-ray, knee MRI, volumetric CT,
conformal prediction, sequential stopping, and counterfactual explanations are
experimental work and are not validated features of the supported path.

## Current evidence

The locked RSNA pilot used 2,400 enriched slices with patient-separated train,
validation, calibration, and test partitions. Its EfficientNet-B0 model has six
independent outputs in this order:

```text
epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any
```

The first locked test produced an `any` hemorrhage ROC-AUC of 0.8417 and
sensitivity of 0.8831 at the calibration-selected threshold. These are pilot,
slice-level results and do not establish study-level or clinical performance.

The evidence bundle is under `artifacts/rsna_baseline/`. Dataset semantics and
limitations are documented in `docs/DATASET_CONTRACTS.md`. The execution plan is
in `PROJECT_PLAN.md`.

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

The active suite covers the RSNA adapter, upload boundary, supported worklist,
and Streamlit render contract:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Run the research worklist

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app accepts one CT DICOM instance at a time, runs the verified slice-level
adapter, and adds its evidence to the session-local worklist. It provides explicit
override and completion controls with audit history.

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
