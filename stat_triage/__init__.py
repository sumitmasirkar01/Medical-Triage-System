"""Supported STAT-Triage research components."""

from .contracts import EvidenceContractError, Modality, ModelEvidence
from .calibration import (
    AffineLogisticCalibration,
    CalibrationMetrics,
    TemperatureCalibration,
    calibration_metrics,
    fit_temperature_scaling,
    fit_affine_logistic_scaling,
)
from .bootstrap import (
    BOOTSTRAP_METRICS,
    ConfidenceInterval,
    GroupedBootstrapResult,
    grouped_bootstrap_confidence_intervals,
)
from .orchestration import (
    ReleaseValidationError,
    evaluate_rsna_release,
    validate_project_release_manifest,
)
from .losses import (
    MaskedAsymmetricLoss,
    MaskedWeightedBCEWithLogitsLoss,
    masked_asymmetric_loss_with_logits,
    masked_weighted_bce_with_logits,
)

from .evaluation import (
    BinaryMetrics,
    EvidenceBundleError,
    binary_metrics,
    evaluate_locked_predictions,
    validate_evidence_bundle,
)
from .thresholds import OperatingPoint, select_sensitivity_threshold

from .rsna import (
    RSNA_LABELS,
    RSNAEvidence,
    RSNAInferenceAdapter,
    RSNAInputError,
    RSNAPreprocessor,
)
from .worklist import AuditEvent, CaseStatus, PriorityZone, WorklistRow, WorklistService
from .cxr import (
    CXREvidenceError,
    DEFAULT_CXR_BUNDLE_DIR,
    NIHChestXrayAdapter,
    validate_cxr_evidence_bundle,
)
from .cxr_policy import CXRPriorityDecision, CXRWorklistPolicy
from .mrnet import (
    DEFAULT_MRNET_BUNDLE_DIR,
    MRNetEvidenceError,
    MRNetInferenceAdapter,
    MRNetInputError,
    validate_mrnet_evidence_bundle,
)
from .multimodal import (
    MODALITY_REGISTRY,
    ChestXrayRequest,
    HeadCTRequest,
    KneeMRIRequest,
    ModalityDefinition,
    UnifiedInferenceError,
    UnifiedInferenceRouter,
)
from .multimodal_worklist import (
    MultimodalTriagePolicy,
    TriagePolicyDecision,
    UnifiedWorklistRow,
    UnifiedWorklistService,
)
from .artifacts import (
    DEFAULT_MULTIMODAL_REGISTRY,
    MultimodalRegistryError,
    validate_multimodal_evidence_registry,
)
from .cli import build_parser as build_inference_cli_parser, serialize_evidence

__all__ = [
    "EvidenceContractError",
    "Modality",
    "ModelEvidence",
    "CalibrationMetrics",
    "AffineLogisticCalibration",
    "TemperatureCalibration",
    "calibration_metrics",
    "fit_temperature_scaling",
    "fit_affine_logistic_scaling",
    "BOOTSTRAP_METRICS",
    "ConfidenceInterval",
    "GroupedBootstrapResult",
    "grouped_bootstrap_confidence_intervals",
    "ReleaseValidationError",
    "evaluate_rsna_release",
    "validate_project_release_manifest",
    "MaskedAsymmetricLoss",
    "MaskedWeightedBCEWithLogitsLoss",
    "masked_asymmetric_loss_with_logits",
    "masked_weighted_bce_with_logits",
    "BinaryMetrics",
    "EvidenceBundleError",
    "binary_metrics",
    "evaluate_locked_predictions",
    "validate_evidence_bundle",
    "OperatingPoint",
    "select_sensitivity_threshold",
    "RSNA_LABELS",
    "RSNAEvidence",
    "RSNAInferenceAdapter",
    "RSNAInputError",
    "RSNAPreprocessor",
    "AuditEvent",
    "CaseStatus",
    "PriorityZone",
    "WorklistRow",
    "WorklistService",
    "CXREvidenceError",
    "DEFAULT_CXR_BUNDLE_DIR",
    "NIHChestXrayAdapter",
    "validate_cxr_evidence_bundle",
    "CXRPriorityDecision",
    "CXRWorklistPolicy",
    "DEFAULT_MRNET_BUNDLE_DIR",
    "MRNetEvidenceError",
    "MRNetInferenceAdapter",
    "MRNetInputError",
    "validate_mrnet_evidence_bundle",
    "MODALITY_REGISTRY",
    "ChestXrayRequest",
    "HeadCTRequest",
    "KneeMRIRequest",
    "ModalityDefinition",
    "UnifiedInferenceError",
    "UnifiedInferenceRouter",
    "MultimodalTriagePolicy",
    "TriagePolicyDecision",
    "UnifiedWorklistRow",
    "UnifiedWorklistService",
    "DEFAULT_MULTIMODAL_REGISTRY",
    "MultimodalRegistryError",
    "validate_multimodal_evidence_registry",
    "build_inference_cli_parser",
    "serialize_evidence",
]
