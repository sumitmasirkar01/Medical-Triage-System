# Reproducible installation and verification

Verified locally on 2026-10-01 with Windows, Python 3.12.7, and CPU inference.
The locked evaluation was performed separately on the Kaggle Tesla T4 described
in `PROJECT_PLAN.md`.

## Supported environment

| Component | Verified version |
| --- | --- |
| Python | 3.12.7 |
| PyTorch | 2.13.0+cpu |
| Torchvision | 0.28.0+cpu |
| Streamlit | 1.62.0 |
| pydicom | 3.0.2 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |

The exact verified dependency pins are in `requirements.txt`. `pyproject.toml`
defines compatible ranges for package installation.

## Fresh environment

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

The PyTorch wheel selected by pip depends on the host platform. Follow official
PyTorch installation guidance first when a CUDA-specific build is required, then
install this project with `--no-deps`.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.create_demo_dicom --output sample_dicoms\DEMO_CT.dcm
.\.venv\Scripts\python.exe -m scripts.demo_smoke --dicom sample_dicoms\DEMO_CT.dcm
```

Expected supported-suite result as of 2026-10-01: `15 passed`.

## Model bundle resolution

By default, the adapter looks for:

```text
artifacts/rsna_baseline/rsna_baseline_bundle
```

When running outside the repository root, set an absolute bundle path:

```powershell
$env:STAT_TRIAGE_BUNDLE_DIR = "E:\Medical_Triage_system\artifacts\rsna_baseline\rsna_baseline_bundle"
```

Loading fails if the checkpoint SHA-256 differs from the bundle manifest.
