# Phase 3 triage outcome contract

Status: implemented and validated on 2026-10-24.

Phase 3 separates a model's limited finding assessment from the workflow action.
No outcome represents a general diagnosis, and every outcome requires human review.

| Disposition | Finding assessment | Meaning |
| --- | --- | --- |
| `RED` | `supported_finding_detected` | A configured urgent target crossed its locked operating threshold. |
| `YELLOW` | `supported_finding_detected` or `no_supported_finding_detected` | A non-emergency target was positive, a score was near its own locked threshold, or a wait-time rule requires review. |
| `GREEN` | `no_supported_finding_detected` | No supported target crossed its locked threshold and no review-escalation rule fired. This does not mean medically normal. |
| `UNSUPPORTED` | `indeterminate` | The modality, anatomy, format, or study structure is outside the supported input contract. No model evidence is emitted. |
| `FAILED` | `indeterminate` | A supported request could not be processed safely. No model evidence is emitted. |

The immutable implementation is `stat_triage.triage_outcome.TriageOutcome`.
Routed outcomes must contain validated `ModelEvidence`, a finite workflow priority,
and locked-positive targets consistent with the evidence. Unsupported and failed
outcomes cannot contain predictions or positive targets. Human review is mandatory
for all five dispositions because this release is a research demonstration.

The unified CLI emits this contract when `--with-triage` is supplied. The next
Phase 3 step will add study-level input validation and map ingestion failures into
`UNSUPPORTED` or `FAILED` outcomes at the service boundary.
