# Phase 3 unified inference service

`UnifiedTriageService` is the public Phase 3 boundary for model execution and
workflow disposition. It accepts an explicit `HeadCTRequest`,
`HeadCTStudyRequest`, `ChestXrayRequest`, or `KneeMRIRequest`; it never infers a
modality from a filename or pixel content.

Successful requests are routed to the verified locked adapter and then to
`MultimodalTriagePolicy`. Every call returns a `TriageOutcome` with one of
`RED`, `YELLOW`, `GREEN`, `UNSUPPORTED`, or `FAILED`. `GREEN` means only that no
supported target crossed its locked threshold. It does not mean normal, and
every outcome requires human review.

The service fails closed. Contract violations and unknown request types become
`UNSUPPORTED`. Missing inputs, decoding failures, model failures, and
unexpected runtime errors become `FAILED`. Public outcomes contain stable
reason codes rather than exception messages, paths, decoder output, or other
implementation details. The original exception remains chained inside the
router for local diagnostics.

Head CT series requests pass through the validated single-series ingestion
contract and the locked mean-pooling study protocol. Slice and study requests
remain distinct endpoints. Chest X-ray requests accept one frontal radiograph;
knee MRI requests require all three caller-labelled MRNet planes.

Example:

```python
from stat_triage import HeadCTStudyRequest, UnifiedTriageService

service = UnifiedTriageService(device="cuda")
outcome = service.triage(HeadCTStudyRequest(list_of_dicom_paths))
payload = outcome.to_dict()
```

The service is a research demonstration interface. It does not establish a
diagnosis, certify normality, or replace radiologist review.
