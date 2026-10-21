# core/__init__.py
"""
STAT-Triage AI: Core Engine Package
"""

from .base import (
    ModalityType,
    TriagePriorityZone,
    ModalityTriageOutput,
    DynamicQueueItem,
    BaseTriageAdapter,
)

# Aliases for backward compatibility
TriageZone = TriagePriorityZone
TriageRecord = DynamicQueueItem

from .gateway import (
    DynamicPriorityGateway,
)

TriageGateway = DynamicPriorityGateway

from .open_clip_engine import (
    OpenBiomedicalCLIPEngine,
)

from .axial_transformer import (
    VolumetricAxialTransformer,
    AsymmetricFocalLoss,
)

from .conformal import (
    SplitConformalCalibrator,
)

from .sequential_sprt import (
    SequentialProbabilityRatioTest,
    SPRTDecisionStatus,
    SPRTStreamResult,
)

from .counterfactual import (
    CausalCounterfactualEngine,
)

__all__ = [
    "ModalityType",
    "TriagePriorityZone",
    "TriageZone",
    "ModalityTriageOutput",
    "DynamicQueueItem",
    "TriageRecord",
    "BaseTriageAdapter",
    "DynamicPriorityGateway",
    "TriageGateway",
    "OpenBiomedicalCLIPEngine",
    "VolumetricAxialTransformer",
    "AsymmetricFocalLoss",
    "SplitConformalCalibrator",
    "SequentialProbabilityRatioTest",
    "SPRTDecisionStatus",
    "SPRTStreamResult",
    "CausalCounterfactualEngine",
]