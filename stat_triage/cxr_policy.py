"""Explicit workflow routing for the validated NIH Chest X-ray pilot."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import Modality, ModelEvidence
from .datasets.nih_training import NIH_TARGET_ORDER
from .worklist import PriorityZone


@dataclass(frozen=True)
class CXRPriorityDecision:
    zone: PriorityZone
    within_zone_score: float
    rule_id: str
    reason: str


class CXRWorklistPolicy:
    """Route pneumothorax above other pilot findings without pooling modalities."""

    rule_id = "nih_cxr14_pilot_v1"

    def route(self, evidence: ModelEvidence) -> CXRPriorityDecision:
        if evidence.modality is not Modality.CHEST_XRAY:
            raise ValueError("CXR policy requires chest_xray evidence")
        expected = set(NIH_TARGET_ORDER)
        if set(evidence.decisions) != expected or set(evidence.calibrated_scores) != expected:
            raise ValueError("CXR evidence does not contain the four locked targets")

        if evidence.decisions["pneumothorax"]:
            return CXRPriorityDecision(
                PriorityZone.RED,
                evidence.calibrated_scores["pneumothorax"],
                self.rule_id,
                "pneumothorax_positive_at_locked_operating_point",
            )
        secondary = tuple(target for target in NIH_TARGET_ORDER if target != "pneumothorax")
        positives = tuple(target for target in secondary if evidence.decisions[target])
        if positives:
            return CXRPriorityDecision(
                PriorityZone.YELLOW,
                max(evidence.calibrated_scores[target] for target in positives),
                self.rule_id,
                "other_pilot_finding_positive:" + ",".join(positives),
            )
        return CXRPriorityDecision(
            PriorityZone.GREEN,
            max(evidence.calibrated_scores.values()),
            self.rule_id,
            "no_pilot_finding_positive",
        )
