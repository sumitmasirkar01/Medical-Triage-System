from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pydicom
from pydicom.dataset import Dataset
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

from .contracts import Modality, ModelEvidence


RSNA_LABELS = (
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
)

def _default_bundle_dir() -> Path:
    configured = os.environ.get("STAT_TRIAGE_BUNDLE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    relative = Path("artifacts") / "rsna_baseline" / "rsna_baseline_bundle"
    working_copy = Path.cwd() / relative
    if working_copy.exists():
        return working_copy.resolve()
    return (Path(__file__).resolve().parents[1] / relative).resolve()


DEFAULT_BUNDLE_DIR = _default_bundle_dir()


class RSNAInputError(ValueError):
    """The supplied DICOM does not satisfy the locked RSNA input contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RSNAEvidence:
    """Auditable output for one CT slice."""

    image_id: str
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    scores: Mapping[str, float]
    any_threshold: float
    predicted_any: bool
    model_sha256: str
    model_epoch: int
    endpoint: str = "slice_level_any_hemorrhage"
    score_interpretation: str = "sigmoid_model_output_not_population_calibrated"

    def __post_init__(self) -> None:
        if tuple(self.scores) != RSNA_LABELS:
            raise ValueError(f"scores must use label order {RSNA_LABELS}")
        if not all(0.0 <= value <= 1.0 for value in self.scores.values()):
            raise ValueError("all scores must be finite probabilities in [0, 1]")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    @property
    def any_score(self) -> float:
        return self.scores["any"]

    def to_model_evidence(
        self,
        *,
        latency_ms: float,
        input_hashes: Mapping[str, str],
    ) -> ModelEvidence:
        """Convert the legacy slice-shaped result to the multimodal contract."""

        return ModelEvidence.created_now(
            exam_id=self.image_id,
            patient_id=self.patient_id,
            modality=Modality.HEAD_CT,
            endpoint=self.endpoint,
            model_id=f"efficientnet_b0_epoch_{self.model_epoch}",
            model_sha256=self.model_sha256,
            preprocess_version="rsna_hu_windows_v1",
            raw_scores=self.scores,
            calibrated_scores=self.scores,
            thresholds={"any": self.any_threshold},
            decisions={"any": self.predicted_any},
            latency_ms=latency_ms,
            input_hashes=input_hashes,
            warnings=(self.score_interpretation,),
            study_instance_uid=self.study_instance_uid,
            series_instance_uid=self.series_instance_uid,
        )


class RSNAPreprocessor:
    """Reproduce the pilot's three HU windows and ImageNet normalization."""

    def __init__(self, run_config: Mapping[str, object]):
        labels = tuple(run_config.get("labels", ()))
        if labels != RSNA_LABELS:
            raise ValueError(f"unexpected label order in run configuration: {labels}")

        size = run_config.get("input_size")
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError("run configuration must contain a two-element input_size")
        self.input_size = (int(size[0]), int(size[1]))

        configured_windows = run_config.get("windows")
        if not isinstance(configured_windows, dict):
            raise ValueError("run configuration does not define HU windows")
        self.windows = tuple(
            (
                float(configured_windows[name]["center"]),
                float(configured_windows[name]["width"]),
            )
            for name in ("brain", "subdural", "bone")
        )

        normalization = run_config.get("normalization")
        if not isinstance(normalization, dict):
            raise ValueError("run configuration does not define normalization")
        self.mean = torch.tensor(normalization["mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(normalization["std"], dtype=torch.float32).view(3, 1, 1)
        if self.mean.numel() != 3 or self.std.numel() != 3 or torch.any(self.std <= 0):
            raise ValueError("normalization must define three positive channel scales")

    @staticmethod
    def _window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
        if width <= 0:
            raise ValueError("HU window width must be positive")
        lower = center - width / 2.0
        return np.clip((hu - lower) / width, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def to_hounsfield_units(dataset: Dataset) -> np.ndarray:
        if str(getattr(dataset, "Modality", "")).upper() != "CT":
            raise RSNAInputError("Expected a CT DICOM instance; other modalities are unsupported.")
        required = (
            "PatientID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "Rows",
            "Columns",
            "PhotometricInterpretation",
            "RescaleSlope",
            "RescaleIntercept",
            "PixelData",
        )
        missing = [name for name in required if not hasattr(dataset, name)]
        if missing:
            raise RSNAInputError(f"DICOM is missing required fields: {', '.join(missing)}")
        photometric = str(dataset.PhotometricInterpretation).upper()
        if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
            raise RSNAInputError(f"Unsupported PhotometricInterpretation: {photometric}")
        if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
            raise RSNAInputError("Expected a single-channel CT image.")
        try:
            pixels = dataset.pixel_array.astype(np.float32)
        except Exception as exc:
            raise RSNAInputError(f"Pixel data could not be decoded: {exc}") from exc
        if pixels.ndim != 2:
            raise RSNAInputError(
                f"Expected one two-dimensional CT slice; received shape {pixels.shape}."
            )
        expected_shape = (int(dataset.Rows), int(dataset.Columns))
        if pixels.shape != expected_shape:
            raise RSNAInputError(
                f"Decoded pixel shape {pixels.shape} does not match DICOM rows/columns {expected_shape}."
            )
        if photometric == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
        slope = float(dataset.RescaleSlope)
        intercept = float(dataset.RescaleIntercept)
        if not np.isfinite(slope) or not np.isfinite(intercept) or slope == 0.0:
            raise RSNAInputError("RescaleSlope and RescaleIntercept must be finite; slope cannot be zero.")
        hu = pixels * slope + intercept
        if not np.isfinite(hu).all():
            raise RSNAInputError("DICOM rescaling produced non-finite HU values.")
        return hu

    def transform_dataset(self, dataset: Dataset) -> Tensor:
        hu = self.to_hounsfield_units(dataset)
        channels = np.stack(
            [self._window(hu, center, width) for center, width in self.windows],
            axis=0,
        )
        tensor = torch.from_numpy(channels).unsqueeze(0)
        if tuple(tensor.shape[-2:]) != self.input_size:
            tensor = F.interpolate(
                tensor,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        tensor = tensor.squeeze(0)
        return (tensor - self.mean) / self.std


class RSNAInferenceAdapter:
    """Load and execute the verified six-output EfficientNet-B0 checkpoint."""

    def __init__(
        self,
        bundle_dir: str | Path = DEFAULT_BUNDLE_DIR,
        device: str | torch.device | None = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.run_config = self._read_json("run_config.json")
        self.operating_point = self._read_json("operating_point.json")
        self.manifest = self._read_json("bundle_manifest.json")
        self._validate_bundle_contract()
        self.preprocessor = RSNAPreprocessor(self.run_config)
        self.model, self.model_epoch = self._load_model()

    def _read_json(self, name: str) -> dict:
        path = self.bundle_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _validate_bundle_contract(self) -> None:
        if tuple(self.run_config.get("labels", ())) != RSNA_LABELS:
            raise ValueError("run configuration label order does not match the RSNA contract")
        if self.run_config.get("model") != "torchvision_efficientnet_b0":
            raise ValueError("unsupported checkpoint architecture")
        if self.operating_point.get("endpoint") != "slice_level_any_hemorrhage":
            raise ValueError("unsupported operating-point endpoint")

        checkpoint = self.bundle_dir / "best_efficientnet_b0.pt"
        expected = self.manifest.get("files", {}).get(checkpoint.name, {})
        expected_hash = expected.get("sha256")
        if not expected_hash or _sha256(checkpoint) != expected_hash:
            raise ValueError("checkpoint SHA-256 does not match the bundle manifest")
        if self.operating_point.get("model_checkpoint_sha256") != expected_hash:
            raise ValueError("operating point refers to a different checkpoint")
        self.model_sha256 = expected_hash
        self.any_threshold = float(self.operating_point["threshold"])
        if not 0.0 <= self.any_threshold <= 1.0:
            raise ValueError("operating threshold must be in [0, 1]")

    def _load_model(self) -> tuple[nn.Module, int]:
        checkpoint = torch.load(
            self.bundle_dir / "best_efficientnet_b0.pt",
            map_location="cpu",
            weights_only=True,
        )
        if tuple(checkpoint.get("labels", ())) != RSNA_LABELS:
            raise ValueError("checkpoint label order does not match the RSNA contract")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint has no model_state_dict")

        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(RSNA_LABELS))
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        return model, int(checkpoint["epoch"])

    @torch.inference_mode()
    def predict_dataset(self, dataset: Dataset, image_id: str | None = None) -> RSNAEvidence:
        tensor = self.preprocessor.transform_dataset(dataset).unsqueeze(0).to(self.device)
        scores_array = torch.sigmoid(self.model(tensor)).squeeze(0).cpu().tolist()
        scores = {label: float(score) for label, score in zip(RSNA_LABELS, scores_array)}
        resolved_image_id = image_id or str(getattr(dataset, "SOPInstanceUID", "UNKNOWN_IMAGE"))
        return RSNAEvidence(
            image_id=resolved_image_id,
            patient_id=str(getattr(dataset, "PatientID", "UNKNOWN_PATIENT")),
            study_instance_uid=str(getattr(dataset, "StudyInstanceUID", "UNKNOWN_STUDY")),
            series_instance_uid=str(getattr(dataset, "SeriesInstanceUID", "UNKNOWN_SERIES")),
            scores=scores,
            any_threshold=self.any_threshold,
            predicted_any=scores["any"] >= self.any_threshold,
            model_sha256=self.model_sha256,
            model_epoch=self.model_epoch,
        )

    def predict_file(self, path: str | Path) -> RSNAEvidence:
        source = Path(path)
        dataset = pydicom.dcmread(source)
        return self.predict_dataset(dataset, image_id=source.stem)
