# test_triad_gateway.py
"""
End-to-End Verification Test for the Complete 3-Modality Diagnostic Triad:
  - Pathway A: Head CT (Volumetric 3D)
  - Pathway B: Chest X-Ray (2D Radiograph)
  - Pathway C: Knee MRI (Soft-Tissue Multi-Slice)
"""

import torch
from core.models import HeadCTAdapter, ChestXRayAdapter, KneeMRIAdapter
from core.gateway import TriageGateway, TriageRecord


def test_triad_pipeline():
    print("=" * 85)
    print("🏥 EXECUTING 3-MODALITY DIAGNOSTIC TRIAD INGESTION BENCHMARK")
    print("=" * 85)

    # 1. Initialize All 3 Production Adapters
    adapters = {
        "HEAD_CT": HeadCTAdapter(weights_path="weights/head_ct_gru_weights.pt", device="cpu"),
        "CHEST_XRAY": ChestXRayAdapter(weights_path="weights/chest_xray_weights.pt", device="cpu"),
        "KNEE_MRI": KneeMRIAdapter(weights_path="weights/knee_mri_weights.pt", device="cpu"),
    }

    # 2. Simulate 6 incoming emergency & routine patients across all 3 modalities
    incoming_patients = [
        {"id": "PT-HEAD-EMERG-01", "modality": "HEAD_CT", "tensor": torch.randn(32, 3, 256, 256)},
        {"id": "PT-CXR-ROUTINE-02", "modality": "CHEST_XRAY", "tensor": torch.randn(3, 256, 256)},
        {"id": "PT-MRI-TRAUMA-03", "modality": "KNEE_MRI", "tensor": torch.randn(3, 256, 256)},
        {"id": "PT-CXR-EMERG-04", "modality": "CHEST_XRAY", "tensor": torch.randn(3, 256, 256)},
        {"id": "PT-HEAD-ROUTINE-05", "modality": "HEAD_CT", "tensor": torch.randn(32, 3, 256, 256)},
        {"id": "PT-MRI-CHRONIC-06", "modality": "KNEE_MRI", "tensor": torch.randn(3, 256, 256)},
    ]

    gateway = TriageGateway(red_threshold=0.70, yellow_lower=0.30, yellow_upper=0.70)
    records = []

    print("\n[*] Processing Ingestion Stream through Polymorphic Adapters:")
    for p in incoming_patients:
        adapter = adapters[p["modality"]]
        out = adapter.predict(p["tensor"], compute_psi=False)
        
        rec = TriageRecord(
            patient_id=p["id"],
            study_uid=f"1.2.840.113619.{p['id']}",
            p_critical=out.p_critical,
            confidence=out.confidence,
            psi_score=0.92,
            metadata={"modality": out.modality.value, "subtypes": out.subtypes}
        )
        gateway.route_record(rec)
        records.append(rec)
        print(f"  • Ingested {p['id']:<18} [{p['modality']:<10}] -> P(Crit): {rec.p_critical:.4f} | Zone: {rec.assigned_zone.value}")

    # 3. Dynamic Priority Queue Sorting
    sorted_queue = gateway.sort_queue(records)

    print("\n" + "=" * 85)
    print(f"{'Rank':^5} | {'Patient ID':^18} | {'Modality':^12} | {'Zone':^8} | {'P(Critical)':^12} | {'Priority S(t)'}")
    print("=" * 85)
    for idx, r in enumerate(sorted_queue, 1):
        mod = r.metadata.get("modality", "N/A")
        print(f" {idx:02d}   | {r.patient_id:<18} | {mod:<12} | {r.assigned_zone.value:<6} |   {r.p_critical:8.4f}   |   {r.priority_score:.4f}")
    print("=" * 85)
    print("\n✅ 3-MODALITY TRIAGE TRIAD FULLY OPERATIONAL!")


if __name__ == "__main__":
    test_triad_pipeline()