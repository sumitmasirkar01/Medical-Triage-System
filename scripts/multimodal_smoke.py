"""Run all real checkpoints through the unified router and worklist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from scripts.create_demo_dicom import create_demo_ct
from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal import (
    ChestXrayRequest,
    HeadCTRequest,
    KneeMRIRequest,
    UnifiedInferenceRouter,
)
from stat_triage.multimodal_worklist import UnifiedWorklistService


def create_cxr_fixture(path: Path) -> None:
    """Create a deterministic grayscale radiograph-shaped smoke fixture."""
    height, width = 320, 256
    y, x = np.mgrid[:height, :width]
    image = 35 + 120 * (1 - y / height)
    left_lung = ((x - 78) / 52) ** 2 + ((y - 155) / 115) ** 2 < 1
    right_lung = ((x - 178) / 52) ** 2 + ((y - 155) / 115) ** 2 < 1
    image[left_lung | right_lung] -= 28
    image[:, 124:132] += 34
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def create_mri_fixtures(directory: Path) -> dict[str, Path]:
    """Create deterministic uint8 [slice,height,width] arrays for three planes."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for plane_index, (plane, slices) in enumerate(
        (("axial", 5), ("coronal", 6), ("sagittal", 7))
    ):
        volume = []
        y, x = np.mgrid[:96, :96]
        for slice_index in range(slices):
            radius = 24 + slice_index * 1.5
            center_x = 48 + plane_index * 2
            center_y = 48 - plane_index * 2
            tissue = ((x - center_x) ** 2 + (y - center_y) ** 2) < radius**2
            image = np.full((96, 96), 12 + plane_index * 4, dtype=np.float32)
            image[tissue] = 110 + slice_index * 8
            image += 18 * np.exp(-((x - 48) ** 2 + (y - 48) ** 2) / 500)
            volume.append(np.clip(image, 0, 255).astype(np.uint8))
        path = directory / f"{plane}.npy"
        np.save(path, np.stack(volume))
        paths[plane] = path
    return paths


def evidence_summary(evidence: ModelEvidence) -> dict[str, object]:
    return {
        "exam_id": evidence.exam_id,
        "modality": evidence.modality.value,
        "endpoint": evidence.endpoint,
        "model_id": evidence.model_id,
        "model_sha256": evidence.model_sha256,
        "raw_scores": dict(evidence.raw_scores),
        "calibrated_scores": dict(evidence.calibrated_scores),
        "thresholds": dict(evidence.thresholds),
        "decisions": dict(evidence.decisions),
        "latency_ms": evidence.latency_ms,
        "input_hashes": dict(evidence.input_hashes),
        "warnings": list(evidence.warnings),
    }


def run_smoke(fixture_dir: Path, *, device: str) -> dict[str, object]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    ct_path = fixture_dir / "smoke_head_ct.dcm"
    cxr_path = fixture_dir / "smoke_chest_xray.png"
    create_demo_ct(ct_path)
    create_cxr_fixture(cxr_path)
    mri = create_mri_fixtures(fixture_dir / "smoke_knee_mri")

    router = UnifiedInferenceRouter(device=device)
    worklist = UnifiedWorklistService()
    loading_sequence = [list(modality.value for modality in router.loaded_modalities)]
    requests = (
        HeadCTRequest(ct_path),
        ChestXrayRequest(cxr_path, exam_id="SMOKE-CXR", patient_id="SMOKE-PATIENT-CXR"),
        KneeMRIRequest(
            axial_path=mri["axial"], coronal_path=mri["coronal"],
            sagittal_path=mri["sagittal"], exam_id="SMOKE-MRI",
            patient_id="SMOKE-PATIENT-MRI",
        ),
    )
    outputs = []
    for request in requests:
        evidence = router.predict(request)
        row = worklist.enqueue(evidence)
        outputs.append(
            evidence_summary(evidence)
            | {
                "automatic_zone": row.automatic_zone.value,
                "positive_targets": list(row.positive_targets),
                "worklist_rationale": list(row.rationale),
            }
        )
        loading_sequence.append([m.value for m in router.loaded_modalities])

    active = worklist.active_worklist()
    expected_modalities = {modality.value for modality in Modality}
    observed_modalities = {item["modality"] for item in outputs}
    if observed_modalities != expected_modalities:
        raise RuntimeError("Smoke run did not produce all registered modalities")
    if len(active) != 3 or len(worklist.audit_history()) != 3:
        raise RuntimeError("Smoke worklist did not retain all three audited cases")
    expected_loading = [
        [], ["head_ct"], ["head_ct", "chest_xray"],
        ["head_ct", "chest_xray", "knee_mri"],
    ]
    if loading_sequence != expected_loading:
        raise RuntimeError(f"Unexpected lazy-loading sequence: {loading_sequence}")
    unloaded = [modality.value for modality in router.unload()]
    if router.loaded_modalities:
        raise RuntimeError("Router retained adapters after explicit unload")
    return {
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "synthetic_deidentified_inputs": True,
        "real_locked_checkpoints": True,
        "modalities_exercised": sorted(observed_modalities),
        "lazy_loading_sequence": loading_sequence,
        "explicitly_unloaded_modalities": unloaded,
        "loaded_after_unload": [],
        "active_worklist_order": [row.case_id for row in active],
        "audit_actions": [event.action for event in worklist.audit_history()],
        "cuda_peak_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device == "cuda" and torch.cuda.is_available() else None
        ),
        "results": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.fixture_dir:
        report = run_smoke(args.fixture_dir, device=args.device)
    else:
        with tempfile.TemporaryDirectory(prefix="stat_triage_smoke_") as directory:
            report = run_smoke(Path(directory), device=args.device)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print("Saved:", args.output.resolve())
    print(rendered, end="")


if __name__ == "__main__":
    main()
