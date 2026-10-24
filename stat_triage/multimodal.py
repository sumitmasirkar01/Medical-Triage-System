"""Explicit multimodal registry and lazy inference routing."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import time
from types import MappingProxyType
from typing import Callable, Mapping, TypeAlias

import torch
from pydicom.errors import InvalidDicomError

from .contracts import Modality, ModelEvidence
from .cxr import DEFAULT_CXR_BUNDLE_DIR, NIHChestXrayAdapter
from .datasets.mrnet import MRNET_PLANES, MRNET_TARGETS
from .datasets.nih_training import NIH_TARGET_ORDER
from .evaluation import sha256
from .mrnet import DEFAULT_MRNET_BUNDLE_DIR, MRNetInferenceAdapter
from .rsna import DEFAULT_BUNDLE_DIR, RSNAInferenceAdapter, RSNA_LABELS


@dataclass(frozen=True)
class ModalityDefinition:
    modality: Modality
    display_name: str
    endpoint: str
    evaluation_level: str
    input_roles: tuple[str, ...]
    labels: tuple[str, ...]
    locked_decision_targets: tuple[str, ...]
    limitations: tuple[str, ...]


MODALITY_REGISTRY: Mapping[Modality, ModalityDefinition] = MappingProxyType(
    {
        Modality.HEAD_CT: ModalityDefinition(
            modality=Modality.HEAD_CT,
            display_name="Head CT",
            endpoint="slice_level_any_hemorrhage",
            evaluation_level="slice",
            input_roles=("dicom_slice", "dicom_series"),
            labels=RSNA_LABELS,
            locked_decision_targets=("any",),
            limitations=(
                "slice_and_study_endpoints_are_distinct",
                "study_input_requires_one_complete_series",
                "scores_are_not_population_calibrated",
            ),
        ),
        Modality.CHEST_XRAY: ModalityDefinition(
            modality=Modality.CHEST_XRAY,
            display_name="Chest X-ray",
            endpoint="nih_cxr14_four_target_triage",
            evaluation_level="image",
            input_roles=("frontal_radiograph",),
            labels=NIH_TARGET_ORDER,
            locked_decision_targets=NIH_TARGET_ORDER,
            limitations=(
                "frontal_view_cannot_be_verified_from_pixels_alone",
                "nih_labels_are_report_derived",
            ),
        ),
        Modality.KNEE_MRI: ModalityDefinition(
            modality=Modality.KNEE_MRI,
            display_name="Knee MRI",
            endpoint="mrnet_three_target_exam_triage",
            evaluation_level="exam",
            input_roles=MRNET_PLANES,
            labels=MRNET_TARGETS,
            locked_decision_targets=MRNET_TARGETS,
            limitations=(
                "plane_identity_is_caller_supplied",
                "patient_identifiers_are_unavailable",
                "high_sensitivity_thresholds_have_low_specificity",
            ),
        ),
    }
)


@dataclass(frozen=True)
class HeadCTRequest:
    dicom_path: Path

    def __init__(self, dicom_path: str | Path) -> None:
        object.__setattr__(self, "dicom_path", Path(dicom_path))


@dataclass(frozen=True)
class ChestXrayRequest:
    image_path: Path
    exam_id: str | None = None
    patient_id: str | None = None

    def __init__(
        self,
        image_path: str | Path,
        *,
        exam_id: str | None = None,
        patient_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "image_path", Path(image_path))
        object.__setattr__(self, "exam_id", exam_id)
        object.__setattr__(self, "patient_id", patient_id)


@dataclass(frozen=True)
class KneeMRIRequest:
    axial_path: Path
    coronal_path: Path
    sagittal_path: Path
    exam_id: str
    patient_id: str | None = None

    def __init__(
        self,
        *,
        axial_path: str | Path,
        coronal_path: str | Path,
        sagittal_path: str | Path,
        exam_id: str,
        patient_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "axial_path", Path(axial_path))
        object.__setattr__(self, "coronal_path", Path(coronal_path))
        object.__setattr__(self, "sagittal_path", Path(sagittal_path))
        object.__setattr__(self, "exam_id", exam_id)
        object.__setattr__(self, "patient_id", patient_id)


InferenceRequest: TypeAlias = HeadCTRequest | ChestXrayRequest | KneeMRIRequest


class UnifiedInferenceError(ValueError):
    """A modality request could not be routed or processed."""


class UnifiedInferenceRouter:
    """Lazily load the verified adapter selected by an explicit request type."""

    def __init__(
        self,
        *,
        device: str | None = None,
        bundle_dirs: Mapping[Modality, str | Path] | None = None,
        adapter_factories: Mapping[Modality, Callable[[], object]] | None = None,
    ) -> None:
        defaults = {
            Modality.HEAD_CT: DEFAULT_BUNDLE_DIR,
            Modality.CHEST_XRAY: DEFAULT_CXR_BUNDLE_DIR,
            Modality.KNEE_MRI: DEFAULT_MRNET_BUNDLE_DIR,
        }
        self.bundle_dirs = {
            modality: Path(path).resolve()
            for modality, path in (defaults | dict(bundle_dirs or {})).items()
        }
        self.device = device
        self.adapter_factories = dict(adapter_factories or {})
        unknown = set(self.adapter_factories).difference(MODALITY_REGISTRY)
        if unknown:
            raise UnifiedInferenceError(f"Unregistered adapter modalities: {unknown}")
        self._adapters: dict[Modality, object] = {}

    @property
    def registered_modalities(self) -> tuple[ModalityDefinition, ...]:
        return tuple(MODALITY_REGISTRY.values())

    @property
    def loaded_modalities(self) -> tuple[Modality, ...]:
        return tuple(self._adapters)

    def unload(self, modality: Modality | None = None) -> tuple[Modality, ...]:
        """Drop one or all cached adapters and release reclaimable accelerator memory."""
        if modality is None:
            removed = tuple(self._adapters)
            self._adapters.clear()
        else:
            resolved = Modality(modality)
            removed = (resolved,) if self._adapters.pop(resolved, None) is not None else ()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return removed

    def _adapter(self, modality: Modality) -> object:
        if modality not in self._adapters:
            if modality in self.adapter_factories:
                adapter = self.adapter_factories[modality]()
            elif modality is Modality.HEAD_CT:
                adapter = RSNAInferenceAdapter(self.bundle_dirs[modality], device=self.device)
            elif modality is Modality.CHEST_XRAY:
                adapter = NIHChestXrayAdapter(self.bundle_dirs[modality], device=self.device)
            elif modality is Modality.KNEE_MRI:
                adapter = MRNetInferenceAdapter(self.bundle_dirs[modality], device=self.device)
            else:  # pragma: no cover - guarded by the closed Modality enum
                raise UnifiedInferenceError(f"Unsupported modality: {modality}")
            self._adapters[modality] = adapter
        return self._adapters[modality]

    def predict(self, request: InferenceRequest) -> ModelEvidence:
        try:
            if isinstance(request, HeadCTRequest):
                adapter = self._adapter(Modality.HEAD_CT)
                started = time.perf_counter()
                evidence = adapter.predict_file(request.dicom_path)
                return evidence.to_model_evidence(
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    input_hashes={request.dicom_path.name: sha256(request.dicom_path)},
                )
            if isinstance(request, ChestXrayRequest):
                adapter = self._adapter(Modality.CHEST_XRAY)
                return adapter.predict_file(
                    request.image_path,
                    patient_id=request.patient_id,
                    exam_id=request.exam_id,
                )
            if isinstance(request, KneeMRIRequest):
                adapter = self._adapter(Modality.KNEE_MRI)
                return adapter.predict_exam(
                    axial=request.axial_path,
                    coronal=request.coronal_path,
                    sagittal=request.sagittal_path,
                    exam_id=request.exam_id,
                    patient_id=request.patient_id,
                )
        except (OSError, ValueError, InvalidDicomError) as exc:
            raise UnifiedInferenceError(str(exc)) from exc
        raise UnifiedInferenceError(
            f"Unsupported inference request type: {type(request).__name__}"
        )
