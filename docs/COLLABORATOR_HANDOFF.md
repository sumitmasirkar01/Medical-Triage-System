# Collaborator handoff

This repository contains the code, locked research evidence, and three model
checkpoints needed to inspect the reported results and run inference without the
original training datasets. It remains a research demonstration and is not
authorized for clinical use.

## Minimum laptop requirements

- 64-bit Windows, macOS, or Linux
- Python 3.12
- At least 8 GB RAM; 16 GB is recommended for the three-model smoke test
- At least 3 GB free disk space for the checkout, environment, and packages
- Internet access during initial package installation
- A CUDA GPU is optional; CPU execution is supported and slower

The repository is approximately 229 MiB before Git packing. All tracked files are
below GitHub's 100 MB per-file limit. A normal Git clone therefore includes the
locked checkpoints; training datasets and patient DICOM files are not included.

## Windows setup

Install Git and 64-bit Python 3.12, then run:

```powershell
git clone https://github.com/sumitmasirkar01/Medical-Triage-System.git
cd Medical-Triage-System
powershell -ExecutionPolicy Bypass -File .\setup_collaborator.ps1
```

The setup creates `.venv`, installs the project and development dependencies,
verifies all checkpoint and evidence hashes, and runs the supported test suite.
Add `-SkipTests` for a quicker installation check. Add `-RunModelSmoke` to load all
three real checkpoints and run deterministic synthetic, de-identified inputs on
CPU; this takes longer and uses more memory.

## macOS or Linux setup

```bash
git clone https://github.com/sumitmasirkar01/Medical-Triage-System.git
cd Medical-Triage-System
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
python -m scripts.check_collaborator_setup
python -m pytest -q
```

PyTorch installation can vary by GPU and operating system. The default dependency
works on CPU. If a platform-specific PyTorch installation is needed, install the
appropriate PyTorch build first and then repeat `python -m pip install -e '.[dev]'`.

## Reproduce the evidence and tables

These commands do not require the original image datasets:

```powershell
.\.venv\Scripts\python.exe -m scripts.validate_multimodal_evidence --hashes-only
.\.venv\Scripts\python.exe -m scripts.generate_multimodal_results
.\.venv\Scripts\python.exe -m scripts.validate_release
```

The manuscript-ready table is
`artifacts/consolidated_results/locked_multimodal_results.md`. The experiment
registry is `research/experiment_registry.json`, and the evidence hierarchy is
explained in `docs/PROTOCOL_AMENDMENT_PHASE2.md`.

## Run the demonstration

Launch the CT research worklist:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run the three-model synthetic smoke test:

```powershell
.\.venv\Scripts\python.exe -m scripts.multimodal_smoke --device cpu
```

The graphical application is CT-only. The safe CLI supports CT slices, complete CT
series, frontal chest radiographs, and three-plane knee MRI exams; see
`docs/CLI_INFERENCE.md`.

## Information the paper author should use

- Use the CT Phase 2 study-level row as the primary confirmatory result.
- Label the CT slice, chest X-ray, and knee MRI evaluations as retrospective locked
  baselines.
- Do not report ROC-AUC as accuracy or combine results into one system accuracy.
- Do not claim clinical readiness, prospective benefit, external generalization,
  fairness, or direct cross-modality superiority.
- Cite artifact hashes and the experiment registry when transcribing results.

Do not add patient scans, Kaggle credentials, API keys, local paths, or dataset
archives to the repository.
