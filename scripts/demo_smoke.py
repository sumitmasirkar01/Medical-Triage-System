"""Rehearse inference and audited worklist actions on one local CT DICOM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stat_triage import PriorityZone, RSNAInferenceAdapter, WorklistService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom", type=Path, required=True)
    args = parser.parse_args()

    adapter = RSNAInferenceAdapter(device="cpu")
    evidence = adapter.predict_file(args.dicom)
    worklist = WorklistService()
    initial = worklist.enqueue(evidence)
    overridden = worklist.override(
        initial.case_id,
        PriorityZone.GREEN,
        "Demonstration of explicit clinician reclassification",
        actor="demo-radiologist",
    )
    restored = worklist.clear_override(
        initial.case_id,
        actor="demo-radiologist",
        reason="End override demonstration",
    )
    worklist.complete(initial.case_id, actor="demo-radiologist")

    print(
        json.dumps(
            {
                "dicom": str(args.dicom),
                "image_id": evidence.image_id,
                "model_sha256": evidence.model_sha256,
                "any_score": evidence.any_score,
                "threshold": evidence.any_threshold,
                "automatic_zone": initial.automatic_zone.value,
                "overridden_zone": overridden.effective_zone.value,
                "restored_zone": restored.effective_zone.value,
                "active_after_completion": len(worklist.active_worklist()),
                "audit_actions": [
                    event.action for event in worklist.audit_history(initial.case_id)
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
