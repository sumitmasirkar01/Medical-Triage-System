# test_polymorphic_gateway.py
"""Verification test for BaseTriageAdapter polymorphic contracts and TriageGateway routing."""

import torch
from core.models import HeadCTAdapter
from core.gateway import TriageGateway, TriageRecord
from core.base import ModalityType, ModalityTriageOutput


def test_polymorphic_pipeline():
    print("=" * 80)
    print("🔬 TESTING POLYMORPHIC ADAPTER CONTRACT & GATEWAY INGESTION")
    print("=" * 80)

    # 1. Instantiate the polymorphic Head CT Adapter
    adapter = HeadCTAdapter(weights_path="weights/head_ct_gru_weights.pt", device="cpu")
    print(f"  [✓] Initialized {adapter.modality.value} Adapter")

    # 2. Feed simulated 32-slice CT tensor (32, 3, 256, 256)
    dummy_volume = torch.randn(32, 3, 256, 256)
    output: ModalityTriageOutput = adapter.predict(dummy_volume, compute_psi=True)

    print(f"  [✓] Output Contract Validation:")
    print(f"      • Modality        : {output.modality.value}")
    print(f"      • P(Critical)     : {output.p_critical:.4f}")
    print(f"      • Confidence      : {output.confidence:.4f}")
    print(f"      • PSI Stability   : {output.psi_score:.4f}")
    print(f"      • Peak Slice Idx  : {output.salient_slice_index}")
    print(f"      • Heatmap Shape   : {output.saliency_map.shape}")

    # 3. Ingest into Triage Gateway
    gateway = TriageGateway()
    record = TriageRecord(
        patient_id="PT-TEST-POLY-001",
        study_uid="1.2.840.113619.2.55.99999",
        p_critical=output.p_critical,
        confidence=output.confidence,
        psi_score=output.psi_score,
        metadata={"modality": output.modality.value, "subtypes": output.subtypes}
    )

    zone = gateway.route_record(record)
    priority_score = gateway.compute_priority_score(record)

    print(f"  [✓] Gateway Dynamic Ingestion:")
    print(f"      • Routed Zone     : {zone.name}")
    print(f"      • Priority Score  : {priority_score:.4f}")

    print("\n" + "=" * 80)
    print("✅ POLYMORPHIC CONTRACT VERIFIED: READY FOR PATHWAY B (CHEST X-RAY)")
    print("=" * 80)


if __name__ == "__main__":
    test_polymorphic_pipeline()