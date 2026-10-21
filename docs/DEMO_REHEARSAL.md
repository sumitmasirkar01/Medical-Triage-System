# CT worklist demonstration rehearsal

## Preparation

1. Install `requirements-dev.txt` in the project virtual environment.
2. Confirm the locked bundle exists under
   `artifacts/rsna_baseline/rsna_baseline_bundle/`.
3. Run `python -m pytest -q`; do not continue if the supported suite fails.
4. Use de-identified or synthetic DICOM files only during the demonstration.

## Automated rehearsal

```powershell
.\.venv\Scripts\python.exe -m scripts.create_demo_dicom `
  --output sample_dicoms\DEMO_CT.dcm

.\.venv\Scripts\python.exe -m scripts.demo_smoke `
  --dicom sample_dicoms\DEMO_CT.dcm
```

Expected behavior: checkpoint verification, inference, enqueue, manual reclassification,
override clearing, completion, and four retained audit actions. The generated image
is an engineering fixture without clinical ground truth; its model score must not be
presented as diagnostic performance.

## Interactive rehearsal

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. Open **Model provenance** and point out the six labels, preprocessing, threshold,
   epoch, and checkpoint hash.
2. Upload a CT DICOM in **Add DICOM**. Confirm that unsupported modalities and
   malformed files produce an actionable error without entering the worklist.
3. Open **Active worklist** and inspect the six scores. State that these are
   slice-level model outputs and are not population-calibrated probabilities.
4. Apply an override with an actor ID and reason. Confirm the effective zone changes
   and the automatic zone remains visible in the case data.
5. Clear the override and inspect the audit history.
6. Mark the case completed and confirm it leaves the active worklist.

## Claims to avoid

- Do not call this a 3D study-level system.
- Do not claim regulatory readiness, prospective validation, or deployment safety.
- Do not claim validated conformal coverage or measured early-exit compute savings.
- Do not describe the synthetic demonstration image as a diagnosed patient case.
- Do not present the enriched-pilot PPV as a population estimate.
