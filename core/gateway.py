import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any


class TriageZone(Enum):
    ZONE_1_ESCALATED = "Zone 1: Immediate Escalation"
    ZONE_2_SECOND_READ = "Zone 2: Second-Read Audit"
    ZONE_3_ROUTINE = "Zone 3: Normal Worklist"


@dataclass
class TriageRecord:
    study_uid: str
    pseudonym_token: str
    modality: str
    p_critical: float
    confidence: float
    subtypes: List[float]
    assigned_zone: TriageZone
    ingress_monotonic: float
    ingress_wall_time: float
    heatmap_iou: float
    is_explainability_verified: bool

    def current_priority_score(self, current_mono: float, alpha: float = 0.25, k_half: float = 1800.0) -> float:
        """
        S(t) = P(critical) * C + alpha * (t / (t + K))
        Using monotonic clock to guarantee resistance against NTP clock shifts.
        """
        wait_seconds = max(0.0, current_mono - self.ingress_monotonic)
        aging_boost = alpha * (wait_seconds / (wait_seconds + k_half))
        return float((self.p_critical * self.confidence) + aging_boost)


class TriageGateway:
    """Thread-safe operational triage controller."""
    def __init__(self, p_high: float = 0.85, c_high: float = 0.80, p_low: float = 0.45):
        self.p_high = p_high
        self.c_high = c_high
        self.p_low = p_low
        self._lock = threading.RLock()
        self._worklist: List[TriageRecord] = []

    def classify_zone(self, p_crit: float, confidence: float) -> TriageZone:
        if p_crit >= self.p_high and confidence >= self.c_high:
            return TriageZone.ZONE_1_ESCALATED
        elif (self.p_low <= p_crit < self.p_high) or (confidence < self.c_high):
            return TriageZone.ZONE_2_SECOND_READ
        else:
            return TriageZone.ZONE_3_ROUTINE

    def register_study(
        self,
        study_uid: str,
        pseudonym_token: str,
        modality: str,
        p_crit: float,
        confidence: float,
        subtypes: List[float],
        iou_score: float,
        is_verified: bool,
        arrival_offset_sec: float = 0.0
    ) -> TriageRecord:
        with self._lock:
            zone = self.classify_zone(p_crit, confidence)
            now_mono = time.monotonic() - arrival_offset_sec
            now_wall = time.time() - arrival_offset_sec
            
            record = TriageRecord(
                study_uid=study_uid,
                pseudonym_token=pseudonym_token,
                modality=modality,
                p_critical=p_crit,
                confidence=confidence,
                subtypes=subtypes,
                assigned_zone=zone,
                ingress_monotonic=now_mono,
                ingress_wall_time=now_wall,
                heatmap_iou=iou_score,
                is_explainability_verified=is_verified
            )
            self._worklist.append(record)
            return record

    def get_dynamic_ranked_queue(self) -> List[Dict[str, Any]]:
        with self._lock:
            curr_mono = time.monotonic()
            ranked = []
            for record in self._worklist:
                score = record.current_priority_score(curr_mono)
                ranked.append({
                    "record": record,
                    "score": score,
                    "wait_minutes": (curr_mono - record.ingress_monotonic) / 60.0
                })
            ranked.sort(key=lambda item: item["score"], reverse=True)
            return ranked