"""Command-line contracts for unified multimodal inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .contracts import ModelEvidence
from .multimodal import (
    ChestXrayRequest,
    HeadCTRequest,
    KneeMRIRequest,
    UnifiedInferenceRouter,
)
from .multimodal_worklist import MultimodalTriagePolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stat-triage-infer",
        description="Run one verified modality-specific model and emit JSON evidence.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-triage", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ct = subcommands.add_parser("head-ct", help="Infer one head CT DICOM slice")
    ct.add_argument("--dicom", type=Path, required=True)

    cxr = subcommands.add_parser("chest-xray", help="Infer one frontal chest radiograph")
    cxr.add_argument("--image", type=Path, required=True)
    cxr.add_argument("--exam-id")
    cxr.add_argument("--patient-id")

    mri = subcommands.add_parser("knee-mri", help="Infer one three-plane MRNet exam")
    mri.add_argument("--axial", type=Path, required=True)
    mri.add_argument("--coronal", type=Path, required=True)
    mri.add_argument("--sagittal", type=Path, required=True)
    mri.add_argument("--exam-id", required=True)
    mri.add_argument("--patient-id")
    return parser


def _required_file(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{role} file does not exist: {resolved}")
    return resolved


def request_from_arguments(arguments: argparse.Namespace):
    if arguments.command == "head-ct":
        return HeadCTRequest(_required_file(arguments.dicom, "DICOM"))
    if arguments.command == "chest-xray":
        image = _required_file(arguments.image, "Chest X-ray")
        return ChestXrayRequest(
            image,
            exam_id=arguments.exam_id or image.stem,
            patient_id=arguments.patient_id,
        )
    if arguments.command == "knee-mri":
        return KneeMRIRequest(
            axial_path=_required_file(arguments.axial, "Axial"),
            coronal_path=_required_file(arguments.coronal, "Coronal"),
            sagittal_path=_required_file(arguments.sagittal, "Sagittal"),
            exam_id=arguments.exam_id,
            patient_id=arguments.patient_id,
        )
    raise ValueError(f"Unsupported command: {arguments.command}")


def serialize_evidence(evidence: ModelEvidence) -> dict[str, object]:
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
        "input_hashes": dict(evidence.input_hashes),
        "created_at": evidence.created_at.isoformat(),
        "warnings": list(evidence.warnings),
        "study_instance_uid": evidence.study_instance_uid,
        "series_instance_uid": evidence.series_instance_uid,
    }


def run_inference(
    arguments: argparse.Namespace,
    *,
    router: UnifiedInferenceRouter | None = None,
) -> dict[str, object]:
    active_router = router or UnifiedInferenceRouter(device=arguments.device)
    evidence = active_router.predict(request_from_arguments(arguments))
    result: dict[str, object] = {
        "schema_version": "1.0",
        "status": "complete",
        "evidence": serialize_evidence(evidence),
    }
    if arguments.with_triage:
        decision = MultimodalTriagePolicy().evaluate(evidence, wait_minutes=0)
        result["triage"] = {
            **asdict(decision),
            "zone": decision.zone.value,
        }
    return result


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
