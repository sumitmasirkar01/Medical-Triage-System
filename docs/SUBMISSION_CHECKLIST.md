# Submission and demonstration checklist

## Files

- [ ] `README.md` contains only supported commands and current limitations.
- [ ] `release/release_manifest.json` has been regenerated after the final edit.
- [ ] `docs/STAT_Triage_RSNA_Pilot_Report.docx` opens and its tables match the
  locked metrics.
- [ ] The artifact bundle contains the checkpoint, configuration, predictions,
  metrics, split manifest, and bundle manifest.
- [ ] No clinical DICOM or direct patient identifier is included in the submission.
- [ ] `legacy/` is either included with its warning or excluded as a whole; individual
  legacy results are not mixed into supported evidence.

## Verification

- [ ] `python -m pip check` reports no broken requirements.
- [ ] `python -m pytest -q` reports 15 passed.
- [ ] The generated demo DICOM passes the CLI rehearsal.
- [ ] Streamlit accepts the generated DICOM and displays six scores.
- [ ] Override, override clearing, audit history, and completion work interactively.
- [ ] An invalid file and non-CT DICOM produce actionable errors.

## Five-minute demonstration

1. **Problem and scope (30 seconds):** explain worklist prioritization and state that
   this release is a slice-level RSNA research pilot.
2. **Evidence (60 seconds):** show the patient-separated split, locked threshold,
   ROC-AUC and sensitivity with confidence intervals, and enriched-cohort limitation.
3. **Provenance (45 seconds):** show the six labels, HU windows, checkpoint hash,
   selected epoch, and threshold in the Model provenance tab.
4. **Workflow (90 seconds):** upload the generated CT, inspect scores and routing,
   apply an override with actor/reason, clear it, and show the audit entries.
5. **Completion and limitations (45 seconds):** complete the case and state that
   study-level validation, prospective performance, conformal guarantees, and true
   streaming inference remain future work.
6. **Reproducibility (30 seconds):** point to the tests, release manifest, evidence
   bundle, report generator, and clean-environment instructions.

## Questions to prepare for

- Why was the cohort enriched, and which metrics depend on prevalence?
- How was patient leakage prevented?
- Why was the threshold selected on calibration rather than test data?
- Why is this currently slice-level rather than study-level?
- Why are the model scores not described as calibrated clinical probabilities?
- Which legacy mathematical methods will be evaluated later, and what evidence gates
  must they pass?
