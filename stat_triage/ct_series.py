"""Fail-closed validation and deterministic ordering for head CT DICOM series."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from statistics import median
import time
from typing import Iterable
import warnings
import zipfile

import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

from .ingestion import MAX_DICOM_UPLOAD_BYTES
from .contracts import Modality, ModelEvidence
from .evaluation import sha256
from .rsna import DEFAULT_BUNDLE_DIR, RSNAInferenceAdapter


MIN_HEAD_CT_SLICES = 10
MAX_HEAD_CT_SLICES = 512
MAX_HEAD_CT_SERIES_BYTES = 512 * 1024 * 1024


def _default_study_protocol_bundle() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "rsna_phase2"
        / "rsna_phase2_locked_protocol_bundle.zip"
    )


DEFAULT_CT_STUDY_PROTOCOL_BUNDLE = _default_study_protocol_bundle()


class CTSeriesIssue(str, Enum):
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class CTSeriesInputError(ValueError):
    """A CT series cannot safely enter study-level inference."""

    def __init__(self, issue: CTSeriesIssue, code: str, message: str) -> None:
        super().__init__(message)
        self.issue = issue
        self.code = code


@dataclass(frozen=True)
class CTSeriesSlice:
    path: Path
    sop_instance_uid: str
    position: float
    instance_number: int | None
    rows: int
    columns: int
    bytes: int


@dataclass(frozen=True)
class ValidatedHeadCTSeries:
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    slices: tuple[CTSeriesSlice, ...]
    ordering_basis: str
    total_bytes: int

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.slices)


def load_locked_ct_study_protocol(
    bundle_path: str | Path = DEFAULT_CT_STUDY_PROTOCOL_BUNDLE,
) -> dict[str, object]:
    """Load and validate the frozen Phase 2 deployment pooling contract."""

    bundle = Path(bundle_path).expanduser().resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    declaration_path = bundle.parent / "validation_report.json"
    try:
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        expected_hash = declaration["outer_bundle_sha256"]["locked_protocol"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("CT study protocol has no valid hash declaration") from exc
    if sha256(bundle) != expected_hash:
        raise ValueError("CT study protocol bundle SHA-256 mismatch")
    try:
        with zipfile.ZipFile(bundle) as archive:
            protocol = json.loads(archive.read("locked_study_protocol.json"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read locked CT study protocol") from exc
    if not isinstance(protocol, dict):
        raise ValueError("Locked CT study protocol must be a JSON object")
    if protocol.get("status") != "locked":
        raise ValueError("CT study protocol is not locked")
    if protocol.get("endpoint") != "study_level_any_hemorrhage":
        raise ValueError("CT study protocol endpoint is unsupported")
    if protocol.get("selected_method") != "mean" or protocol.get("top_k") is not None:
        raise ValueError("CT study protocol must use the locked mean pooling rule")
    if protocol.get("test_rows_present") != 0 or protocol.get("test_evaluated") is not False:
        raise ValueError("CT study protocol was not locked before test access")
    try:
        threshold = float(protocol["operating_point"]["threshold"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CT study protocol has no valid threshold") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("CT study protocol threshold must be finite and in [0, 1]")
    return protocol


def _unsupported(code: str, message: str) -> CTSeriesInputError:
    return CTSeriesInputError(CTSeriesIssue.UNSUPPORTED, code, message)


def _failed(code: str, message: str) -> CTSeriesInputError:
    return CTSeriesInputError(CTSeriesIssue.FAILED, code, message)


def _required_text(dataset: Dataset, name: str) -> str:
    value = str(getattr(dataset, name, "")).strip()
    if not value:
        raise _unsupported("missing_required_metadata", f"DICOM is missing {name}.")
    return value


def _header(path: Path) -> Dataset:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"pydicom.*")
            return pydicom.dcmread(path, stop_before_pixels=True)
    except (InvalidDicomError, OSError, ValueError) as exc:
        raise _failed("dicom_header_read_failed", f"Cannot read DICOM header {path.name}: {exc}") from exc


def validate_head_ct_series(
    paths: Iterable[str | Path],
    *,
    minimum_slices: int = MIN_HEAD_CT_SLICES,
    maximum_slices: int = MAX_HEAD_CT_SLICES,
) -> ValidatedHeadCTSeries:
    """Validate one complete, single-series head CT request before pixel decoding."""

    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be an iterable of DICOM files, not one path")
    if (
        isinstance(minimum_slices, bool)
        or isinstance(maximum_slices, bool)
        or not isinstance(minimum_slices, int)
        or not isinstance(maximum_slices, int)
        or minimum_slices < 1
        or maximum_slices < minimum_slices
    ):
        raise ValueError("slice limits must be positive integers with maximum >= minimum")

    resolved = tuple(Path(path).expanduser().resolve() for path in paths)
    if len(resolved) < minimum_slices:
        raise _unsupported(
            "incomplete_series",
            f"Head CT requires at least {minimum_slices} slices; received {len(resolved)}.",
        )
    if len(resolved) > maximum_slices:
        raise _unsupported(
            "excessive_slice_count",
            f"Head CT supports at most {maximum_slices} slices; received {len(resolved)}.",
        )
    if len(set(resolved)) != len(resolved):
        raise _unsupported("duplicate_file_path", "The CT series contains duplicate file paths.")

    missing = [path.name for path in resolved if not path.is_file()]
    if missing:
        raise _failed("missing_dicom_file", f"CT series contains missing files: {', '.join(missing[:3])}")
    sizes = tuple(path.stat().st_size for path in resolved)
    if any(size <= 0 for size in sizes):
        raise _failed("empty_dicom_file", "The CT series contains an empty DICOM file.")
    if any(size > MAX_DICOM_UPLOAD_BYTES for size in sizes):
        raise _unsupported("slice_too_large", "A CT slice exceeds the 32 MiB per-file limit.")
    total_bytes = sum(sizes)
    if total_bytes > MAX_HEAD_CT_SERIES_BYTES:
        raise _unsupported(
            "series_too_large",
            f"The CT series exceeds the 512 MiB limit ({total_bytes:,} bytes received).",
        )

    records: list[dict[str, object]] = []
    for path, size in zip(resolved, sizes, strict=True):
        dataset = _header(path)
        if str(getattr(dataset, "Modality", "")).upper() != "CT":
            raise _unsupported("unsupported_modality", f"{path.name} is not a CT DICOM instance.")
        frames = int(getattr(dataset, "NumberOfFrames", 1))
        if frames != 1:
            raise _unsupported("multiframe_dicom", "Multi-frame DICOM instances are unsupported.")
        try:
            rows = int(dataset.Rows)
            columns = int(dataset.Columns)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise _unsupported(
                "missing_geometry", f"{path.name} has invalid Rows or Columns."
            ) from exc
        if rows <= 0 or columns <= 0:
            raise _unsupported("invalid_geometry", "DICOM Rows and Columns must be positive.")

        image_position = getattr(dataset, "ImagePositionPatient", None)
        instance_value = getattr(dataset, "InstanceNumber", None)
        try:
            z_position = (
                float(image_position[2]) if image_position is not None else None
            )
            instance_number = int(instance_value) if instance_value is not None else None
        except (IndexError, TypeError, ValueError, OverflowError) as exc:
            raise _unsupported(
                "invalid_slice_position", f"{path.name} has invalid slice-order metadata."
            ) from exc
        if z_position is None and instance_number is None:
            raise _unsupported(
                "missing_slice_position",
                f"{path.name} has neither ImagePositionPatient nor InstanceNumber.",
            )
        if z_position is not None and not math.isfinite(z_position):
            raise _unsupported("invalid_slice_position", "Slice positions must be finite.")

        records.append(
            {
                "path": path,
                "bytes": size,
                "patient_id": _required_text(dataset, "PatientID"),
                "study_uid": _required_text(dataset, "StudyInstanceUID"),
                "series_uid": _required_text(dataset, "SeriesInstanceUID"),
                "sop_uid": _required_text(dataset, "SOPInstanceUID"),
                "z_position": z_position,
                "instance_number": instance_number,
                "rows": rows,
                "columns": columns,
            }
        )

    for field, code, label in (
        ("patient_id", "mixed_patient", "patient"),
        ("study_uid", "mixed_study", "study"),
        ("series_uid", "mixed_series", "series"),
    ):
        if len({record[field] for record in records}) != 1:
            raise _unsupported(code, f"The request contains more than one {label}.")
    if len({record["sop_uid"] for record in records}) != len(records):
        raise _unsupported("duplicate_sop_instance", "SOPInstanceUID values must be unique.")
    if len({(record["rows"], record["columns"]) for record in records}) != 1:
        raise _unsupported("inconsistent_geometry", "All CT slices must have identical dimensions.")

    has_positions = [record["z_position"] is not None for record in records]
    if any(has_positions) and not all(has_positions):
        raise _unsupported(
            "inconsistent_slice_position",
            "ImagePositionPatient must be present on every slice or none of them.",
        )
    if all(has_positions):
        ordering_basis = "image_position_patient_z"
        order_values = [float(record["z_position"]) for record in records]
    else:
        ordering_basis = "instance_number"
        if any(record["instance_number"] is None for record in records):
            raise _unsupported("missing_slice_position", "InstanceNumber is required for ordering.")
        order_values = [float(record["instance_number"]) for record in records]
    if len(set(order_values)) != len(order_values):
        raise _unsupported("duplicate_slice_position", "Slice-order values must be unique.")

    sorted_values = sorted(order_values)
    gaps = [right - left for left, right in zip(sorted_values, sorted_values[1:])]
    if ordering_basis == "instance_number":
        regular_spacing = all(math.isclose(gap, 1.0) for gap in gaps)
    else:
        typical_gap = median(gaps)
        regular_spacing = typical_gap > 0 and all(
            math.isclose(gap, typical_gap, rel_tol=0.10, abs_tol=0.10) for gap in gaps
        )
    if not regular_spacing:
        raise _unsupported(
            "irregular_slice_spacing",
            "Slice spacing is irregular; the series may be incomplete or mixed.",
        )

    ordered = sorted(zip(order_values, records, strict=True), key=lambda item: (item[0], str(item[1]["path"])))
    slices = tuple(
        CTSeriesSlice(
            path=record["path"],
            sop_instance_uid=str(record["sop_uid"]),
            position=float(order_value),
            instance_number=record["instance_number"],
            rows=int(record["rows"]),
            columns=int(record["columns"]),
            bytes=int(record["bytes"]),
        )
        for order_value, record in ordered
    )
    return ValidatedHeadCTSeries(
        patient_id=str(records[0]["patient_id"]),
        study_instance_uid=str(records[0]["study_uid"]),
        series_instance_uid=str(records[0]["series_uid"]),
        slices=slices,
        ordering_basis=ordering_basis,
        total_bytes=total_bytes,
    )


class HeadCTStudyInferenceAdapter:
    """Run the slice model over a validated series and apply locked mean pooling."""

    def __init__(
        self,
        slice_adapter: RSNAInferenceAdapter | None = None,
        *,
        slice_bundle: str | Path = DEFAULT_BUNDLE_DIR,
        protocol: dict[str, object] | None = None,
        protocol_bundle: str | Path = DEFAULT_CT_STUDY_PROTOCOL_BUNDLE,
        device: str | None = None,
    ) -> None:
        self.slice_adapter = slice_adapter or RSNAInferenceAdapter(slice_bundle, device=device)
        self.protocol = protocol or load_locked_ct_study_protocol(protocol_bundle)
        if self.protocol.get("selected_method") != "mean":
            raise ValueError("Head CT study inference requires locked mean pooling")
        try:
            self.threshold = float(self.protocol["operating_point"]["threshold"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Head CT study protocol has no valid threshold") from exc
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Head CT study threshold must be finite and in [0, 1]")

    def predict_series(self, paths: Iterable[str | Path]) -> ModelEvidence:
        validated = validate_head_ct_series(paths)
        started = time.perf_counter()
        slice_scores: list[float] = []
        try:
            for item in validated.slices:
                result = self.slice_adapter.predict_file(item.path)
                if result.patient_id != validated.patient_id:
                    raise ValueError("slice inference patient identifier changed")
                if result.study_instance_uid != validated.study_instance_uid:
                    raise ValueError("slice inference study identifier changed")
                slice_scores.append(float(result.any_score))
        except (OSError, ValueError, RuntimeError) as exc:
            raise _failed(
                "slice_inference_failed", f"Head CT slice inference failed: {exc}"
            ) from exc
        study_score = float(sum(slice_scores) / len(slice_scores))
        if not math.isfinite(study_score) or not 0 <= study_score <= 1:
            raise _failed("invalid_study_score", "Mean-pooled CT study score is invalid.")
        input_hashes = {
            f"{item.sop_instance_uid}.dcm": sha256(item.path)
            for item in validated.slices
        }
        return ModelEvidence.created_now(
            exam_id=validated.study_instance_uid,
            patient_id=validated.patient_id,
            modality=Modality.HEAD_CT,
            endpoint="study_level_any_hemorrhage",
            model_id=(
                f"efficientnet_b0_epoch_{self.slice_adapter.model_epoch}_mean_study_pooling"
            ),
            model_sha256=self.slice_adapter.model_sha256,
            preprocess_version="rsna_hu_windows_v1_mean_pooling_v1",
            raw_scores={"any": study_score},
            calibrated_scores={"any": study_score},
            thresholds={"any": self.threshold},
            decisions={"any": study_score >= self.threshold},
            latency_ms=(time.perf_counter() - started) * 1000.0,
            input_hashes=input_hashes,
            warnings=(
                "mean_pooled_sigmoid_score_not_population_calibrated",
                "research_triage_pilot_not_for_clinical_use",
            ),
            study_instance_uid=validated.study_instance_uid,
            series_instance_uid=validated.series_instance_uid,
        )
