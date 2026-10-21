# RSNA baseline artifact handoff

Copy the contents of the Kaggle `rsna_baseline_bundle.zip` into this directory.
Do not recreate metric JSON by hand: the report generator must consume the saved
outputs from the locked run.

Required files:

- `test_metrics.json`
- `test_predictions.csv`
- `calibration_predictions.csv`
- `operating_point.json`
- `training_history.csv`
- `run_config.json`
- `rsna_pilot_manifest_split.csv`
- `rsna_hu_window_preview.png`

Recommended provenance file:

- `best_efficientnet_b0.pt` (expected SHA-256:
  `4a15c605bb1481de7f12a4881f9c9b5adb36d9386ecac0aa6eb3ae7c3ae805b1`)

The manifest contains dataset paths and pseudonymous identifiers. Keep the bundle
within the project workspace and do not publish it automatically. The report should
use aggregate results and the de-identified window preview only.
