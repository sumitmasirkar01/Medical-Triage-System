# Legacy and experimental code

This directory contains earlier prototypes, synthetic demonstrations, historical
reports, and checks written against APIs that are no longer supported.

Files here are retained for traceability. They are not part of the active test
suite or the supported RSNA inference path, and their numerical outputs must not
be presented as clinical validation.

- `apps/`: incompatible dashboard, CLI, and benchmark entry points.
- `reports/`: reports containing hard-coded or synthetic claims superseded by
  `generate_pilot_report.py`.
- `tests/obsolete/`: tests that import removed interfaces.
- `tests/experiments/`: manually invoked engineering experiments that were named
  as tests but were not discoverable pytest regressions.
- `training/` and `scripts/`: older transformer and unified benchmark workflows.
- `docs/`: historical generated reports and figures.
- `core/`: the earlier prototype package for preprocessing, calibration,
  experimental models, and queue logic.
- `models/` and `weights/`: checkpoints without the provenance and artifact
  contracts required by the supported application.

Code should return to the supported tree only after its data contract, checkpoint,
tests, and claims match the current evidence standards.
