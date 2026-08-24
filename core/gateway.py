# core/gateway.py
"""
Clinical Triage Gateway & Starvation-Free Dynamic Aging Queue.
Routes incoming Head CT volumes into RED, YELLOW, or GREEN operational zones
and computes dynamic priority scores to prevent patient starvation in emergency queues.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


class TriageZone(Enum):
    """Operational clinical prioritization zones."""
    RED = "RED"        # Emergent / Immediate Acute Intervention Required (P_crit >= 0.70)
    YELLOW = "YELLOW"  # Indeterminate / Elevated Risk Monitoring (0.30 <= P_crit < 0.70)
    GREEN = "GREEN"    # Routine / Low Risk Baseline (P_crit < 0.30)


@dataclass
class TriageRecord:
    """Standardized patient triage metadata and inferential audit record."""
    patient_id: str
    study_uid: str
    p_critical: float
    confidence: float = 1.0
    psi_score: float = 1.0
    arrival_time: float = field(default_factory=time.time)
    assigned_zone: Optional[TriageZone] = None
    priority_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TriageGateway:
    """
    Deterministic Triage Router with Starvation-Free Aging Dynamics:
    
    Priority Score Formulation:
        S(t) = P_critical * Confidence + lambda_aging * (t / 60.0) + gamma_psi * (1.0 - PSI)
        
    Where:
        - P_critical * Confidence: Calibrated acute risk magnitude
        - lambda_aging: Starvation prevention rate per minute in queue (default: 0.005/min)
        - gamma_psi * (1.0 - PSI): Saliency instability uncertainty penalty
    """

    def __init__(
        self,
        red_threshold: float = 0.70,
        yellow_lower: float = 0.30,
        yellow_upper: float = 0.70,
        aging_rate_per_min: float = 0.005,
        psi_penalty_weight: float = 0.05
    ):
        self.red_threshold = red_threshold
        self.yellow_lower = yellow_lower
        self.yellow_upper = yellow_upper
        self.aging_rate_per_min = aging_rate_per_min
        self.psi_penalty_weight = psi_penalty_weight

    def route_record(self, record: TriageRecord) -> TriageZone:
        """Assigns the operational triage zone based on calibrated critical hemorrhage probability."""
        p = float(np.clip(record.p_critical, 0.0, 1.0))
        
        if p >= self.red_threshold:
            zone = TriageZone.RED
        elif self.yellow_lower <= p < self.yellow_upper:
            zone = TriageZone.YELLOW
        else:
            zone = TriageZone.GREEN

        record.assigned_zone = zone
        return zone

    def compute_priority_score(self, record: TriageRecord, elapsed_seconds: Optional[float] = None) -> float:
        """
        Calculates monotonic dynamic priority score S(t) for queue sorting.
        Ensures patients with lower initial risk cannot starve indefinitely behind newer arrivals.
        """
        if elapsed_seconds is None:
            elapsed_seconds = max(0.0, time.time() - record.arrival_time)
        else:
            elapsed_seconds = max(0.0, float(elapsed_seconds))

        elapsed_minutes = elapsed_seconds / 60.0
        
        # Base risk weighted by model confidence
        base_risk = record.p_critical * record.confidence
        
        # Linear aging term to prevent queue starvation
        aging_boost = self.aging_rate_per_min * elapsed_minutes
        
        # Explainability uncertainty penalty (unstable Grad-CAM bumps priority for human verification)
        psi_penalty = self.psi_penalty_weight * max(0.0, 1.0 - record.psi_score)
        
        total_score = base_risk + aging_boost + psi_penalty
        record.priority_score = float(total_score)
        return record.priority_score

    def sort_queue(self, queue: List[TriageRecord]) -> List[TriageRecord]:
        """Sorts emergency worklist by dynamic priority score S(t) descending."""
        now = time.time()
        for rec in queue:
            self.compute_priority_score(rec, elapsed_seconds=now - rec.arrival_time)
        return sorted(queue, key=lambda r: r.priority_score, reverse=True)