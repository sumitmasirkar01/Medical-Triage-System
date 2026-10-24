"""Safe command-line contracts for unified multimodal triage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .contracts import ModelEvidence
from .multimodal import (
    ChestXrayRequest,
    HeadCTRequest,
    HeadCTStudyRequest,
    KneeMRIRequest,
    UnifiedInferenceRouter,
)
from .triage_outcome import TriageDisposition, TriageOutcome
from .triage_service import UnifiedTriageService


CLI_SCHEMA_VERSION = "2.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stat-triage-infer",
        description="Run one verified modality-specific model and emit a safe triage outcome.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-triage", action="store_true", help=argparse.SUPPRESS)
    subcommands = parser.add_subparsers(dest="command", required=True)

    ct = subcommands.add_parser("head-ct", help="Infer one head CT DICOM slice")
    ct.add_argument("--dicom", type=Path, required=True)

    ct_series = subcommands.add_parser(
        "head-ct-series", help="Infer one complete head CT DICOM series"
    )
    ct_series.add_argument("--dicom", type=Path, nargs="+", required=True)

    cxr = subcommands.add_parser("chest-xray", help="Infer one frontal chest radiograph")
    cxr.add_argument("--image", type=Path, required=True)
    cxr.add_argument("--exam-id", required=True)
    cxr.add_argument("--patient-id")

    mri = subcommands.add_parser("knee-mri", help="Infer one three-plane MRNet exam")
    mri.add_argument("--axial", type=Path, required=True)
    mri.add_argument("--coronal", type=Path, required=True)
    mri.add_argument("--sagittal", type=Path, required=True)
    mri.add_argument("--exam-id", required=True)
    mri.add_argument("--patient-id")
    return parser


def _resolved(path: Path) -> Path:
    """Normalize a path without probing or disclosing it in an exception."""

    return path.expanduser().resolve()


def request_from_arguments(arguments: argparse.Namespace):
    if arguments.command == "head-ct":
        return HeadCTRequest(_resolved(arguments.dicom))
    if arguments.command == "head-ct-series":
        return HeadCTStudyRequest([_resolved(path) for path in arguments.dicom])
    if arguments.command == "chest-xray":
        return ChestXrayRequest(
            _resolved(arguments.image),
            exam_id=arguments.exam_id,
            patient_id=arguments.patient_id or arguments.exam_id,
        )
    if arguments.command == "knee-mri":
        return KneeMRIRequest(
            axial_path=_resolved(arguments.axial),
            coronal_path=_resolved(arguments.coronal),
            sagittal_path=_resolved(arguments.sagittal),
            exam_id=arguments.exam_id,
            patient_id=arguments.patient_id,
        )
    raise ValueError(f"Unsupported command: {arguments.command}")


def serialize_evidence(evidence: ModelEvidence) -> dict[str, object]:
    input_hashes = {
        f"input_{index:04d}": digest
        for index, digest in enumerate(evidence.input_hashes.values(), start=1)
    }
    return {
        "exam_id": evidence.exam_id,
        "patient_id": evidence.patient_id,
        "modality": evidence.modality.value,
        "endpoint": evidence.endpoint,
        "model_id": evidence.model_id,
        "model_sha256": evidence.model_sha256,
        "preprocess_version": evidence.preprocess_version,
        "raw_scores": dict(evidence.raw_scores),
        "calibrated_scores": dict(evidence.calibrated_scores),
        "thresholds": dict(evidence.thresholds),
        "decisions": dict(evidence.decisions),
        "latency_ms": evidence.latency_ms,
        "input_hashes": input_hashes,
        "created_at": evidence.created_at.isoformat(),
        "warnings": list(evidence.warnings),
        "study_instance_uid": evidence.study_instance_uid,
        "series_instance_uid": evidence.series_instance_uid,
    }


def serialize_outcome(outcome: TriageOutcome) -> dict[str, object]:
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "status": "complete",
        "outcome": outcome.to_dict(),
        "evidence": serialize_evidence(outcome.evidence) if outcome.evidence else None,
    }


def run_inference(
    arguments: argparse.Namespace,
    *,
    service: UnifiedTriageService | None = None,
    router: UnifiedInferenceRouter | None = None,
) -> dict[str, object]:
    if service is not None and router is not None:
        raise ValueError("service and router are mutually exclusive")
    active_service = service or UnifiedTriageService(
        router=router, device=arguments.device
    )
    try:
        request = request_from_arguments(arguments)
    except Exception:
        return serialize_outcome(TriageOutcome.failed("invalid_cli_request"))
    return serialize_outcome(active_service.triage(request))


def exit_code_for_result(result: dict[str, object]) -> int:
    outcome = result.get("outcome")
    disposition = outcome.get("disposition") if isinstance(outcome, dict) else None
    if disposition == TriageDisposition.UNSUPPORTED.value:
        return 3
    if disposition == TriageDisposition.FAILED.value:
        return 2
    return 0


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
