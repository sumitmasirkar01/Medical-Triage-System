# Stanford MRNet locked evidence bundle

This directory contains the immutable evidence from the three-plane knee MRI
research baseline. `mrnet_bundle_manifest.json` declares the byte count and
SHA-256 digest of every artifact.

The official MRNet validation partition was reserved as a locked 120-exam test
partition. Thresholds and affine probability calibrators were selected using a
separate 113-exam calibration partition. They were not changed after test
evaluation.

Validate the bundle and reproduce every saved point estimate with:

```powershell
.\.venv\Scripts\python.exe -c "from stat_triage.mrnet import validate_mrnet_evidence_bundle; print(validate_mrnet_evidence_bundle(recompute_metrics=True))"
```

Patient identifiers are unavailable in MRNet. Splits and bootstrap intervals
therefore use `exam_id`, and patient-level independence cannot be verified.
This is a research demonstration and is not intended for clinical use.
