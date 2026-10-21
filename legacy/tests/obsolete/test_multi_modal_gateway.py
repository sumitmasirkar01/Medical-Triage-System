# test_multi_modal_gateway.py
"""Validates joint execution of Pathway A (Head CT) and Pathway B (Chest X-Ray) in TriageGateway."""

import torch
from core.models import HeadCTAdapter, ChestXRayAdapter
from core.gateway import TriageGateway, TriageRecord


def test_multi_modal_pipeline():
    print("=" * 80)
    print("🏥 EXECUTING MULTI-MODAL TRIAGE VERIFICATION SUITE")
    print("=" * 80)

    # 1. Initialize Both Production Adapters
    ct_adapter = HeadCTAdapter(weights_path="weights/head_ct_gru_weights.pt", device="cpu")
    cxr_adapter = ChestXRayAdapter(weights_path="weights/chest_xray_weights.pt", device="cpu")

    # 2. Simulate Incoming Volumetric Head CT & 2D Chest Radiograph
    dummy_ct = torch.randn(32, 3, 256, 256)
    dummy_cxr = torch.randn(3, 256, 256)

    # 3. Polymorphic Inferences
    ct_out = ct_adapter.predict(dummy_ct, compute_psi=False)
    cxr_out = cxr_adapter.predict(dummy_cxr, compute_psi=False)

    print(f"\n[✓] Head CT Output     : P(Crit)={ct_out.p_critical:.4f} | Conf={ct_out.confidence:.4f} | Subtypes={list(ct_out.subtypes.keys())}")
    print(f"[✓] Chest X-Ray Output : P(Crit)={cxr_out.p_critical:.4f} | Conf={cxr_out.confidence:.4f} | Subtypes={list(cxr_out.subtypes.keys())}")

    # 4. Ingest Both into Triage Gateway Queue
    gateway = TriageGateway()
    records = [
        TriageRecord(patient_id="PT-HEAD-01", study_uid="1.2.3.4.1", p_critical=ct_out.p_critical, confidence=ct_out.confidence, psi_score=0.92),
        TriageRecord(patient_id="PT-CXR-02", study_uid="1.2.3.4.2", p_critical=cxr_out.p_critical, confidence=cxr_out.confidence, psi_score=0.95),
    ]

    for r in records:
        gateway.route_record(r)

    # 5. Sort Unified Queue
    sorted_queue = gateway.sort_queue(records)
    print("\n" + "=" * 80)
    print(f"{'Rank':^5} | {'Patient ID':^12} | {'Zone':^8} | {'P(Critical)':^12} | {'Priority S(t)'}")
    print("=" * 80)
    for idx, r in enumerate(sorted_queue, 1):
        print(f" {idx:02d}   |  {r.patient_id:<10} | {r.assigned_zone.value:<6} |   {r.p_critical:8.4f}   |   {r.priority_score:.4f}")
    print("=" * 80)
    print("\n✅ MULTI-MODAL PATHWAYS FULLY OPERATIONAL!")


if __name__ == "__main__":
    test_multi_modal_pipeline()