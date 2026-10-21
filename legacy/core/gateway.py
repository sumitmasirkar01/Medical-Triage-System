# core/gateway.py
"""
STAT-Triage AI: Dynamic Anti-Starvation Gateway & Prioritization Engine.
Incorporates Conformal Risk Cardinality and SPRT Emergency Halts into S(t).
"""

from typing import Dict, List, Optional
import time
import numpy as np
from core.base import DynamicQueueItem, ModalityTriageOutput, ModalityType, TriagePriorityZone


class DynamicPriorityGateway:
    """
    Manages clinical worklists using starvation-free dynamic priority scoring:
    S(t) = P_crit * 2|P_crit - 0.5| + λ*t + γ*(1 - PSI) + μ*(|C(X)| - 1) + I_SPRT * 2.0
    """

    def __init__(
        self,
        lambda_aging: float = 0.01,         # +0.6 priority points per hour of waiting
        gamma_instability: float = 0.35,     # Penalty for epistemic decision instability
        mu_conformal_ambiguity: float = 0.20,# Penalty for expanding conformal sets
        sprt_escalation_boost: float = 2.0,  # Immediate boost when SPRT detects catastrophic bleed
        red_threshold: float = 0.70,
        yellow_threshold: float = 0.40,
    ):
        self.lambda_aging = lambda_aging
        self.gamma_instability = gamma_instability
        self.mu_conformal_ambiguity = mu_conformal_ambiguity
        self.sprt_escalation_boost = sprt_escalation_boost
        self.red_threshold = red_threshold
        self.yellow_threshold = yellow_threshold
        self._queue: Dict[str, DynamicQueueItem] = {}

    def compute_priority_score(
        self,
        triage_output: ModalityTriageOutput,
        t_wait_minutes: float,
    ) -> float:
        """Calculates dynamic priority score S(t)."""
        p_crit = triage_output.p_critical
        certainty = 2.0 * abs(p_crit - 0.5) # Minimum at 0.5 (maximum ambiguity), 1.0 at extremes
        clinical_severity = p_crit * certainty

        # Anti-starvation linear wait-time aging
        aging_term = self.lambda_aging * t_wait_minutes

        # Decision instability surcharge
        instability_term = self.gamma_instability * max(0.0, 1.0 - triage_output.psi_score)

        # Conformal ambiguity surcharge: expands if prediction set size > 1
        set_size = len(triage_output.conformal_set) if triage_output.conformal_set else 1
        conformal_term = self.mu_conformal_ambiguity * max(0, set_size - 1)

        # SPRT early-exit interrupt boost
        sprt_boost = self.sprt_escalation_boost if triage_output.sprt_halted else 0.0

        return float(clinical_severity + aging_term + instability_term + conformal_term + sprt_boost)

    def route_zone(self, dynamic_score: float, sprt_halted: bool = False) -> TriagePriorityZone:
        """Maps continuous priority score to triage zone."""
        if sprt_halted or dynamic_score >= self.red_threshold:
            return TriagePriorityZone.RED
        elif dynamic_score >= self.yellow_threshold:
            return TriagePriorityZone.YELLOW
        return TriagePriorityZone.GREEN

    def enqueue(
        self,
        study_uid: str,
        patient_id: str,
        modality: ModalityType,
        t_wait_minutes: float,
        triage_output: ModalityTriageOutput,
    ) -> DynamicQueueItem:
        score = self.compute_priority_score(triage_output, t_wait_minutes)
        zone = self.route_zone(score, sprt_halted=triage_output.sprt_halted)

        item = DynamicQueueItem(
            study_uid=study_uid,
            patient_id=patient_id,
            modality=modality,
            arrival_time=time.time() - t_wait_minutes * 60.0,
            triage_output=triage_output,
            urgency_zone=zone,
            priority_score=score,
        )
        self._queue[study_uid] = item
        return item

    def get_sorted_worklist(self) -> List[DynamicQueueItem]:
        """Returns active worklist sorted descending by dynamic score S(t)."""
        return sorted(self._queue.values(), key=lambda item: item.dynamic_score, reverse=True)

    def override_item(self, study_uid: str, target_zone: TriagePriorityZone, reason: str) -> bool:
        """Enforces clinician override authority over algorithmic rankings."""
        if study_uid not in self._queue:
            return False
        item = self._queue[study_uid]
        item.priority_zone = target_zone
        item.clinician_overridden = True
        item.override_reason = reason
        return True
