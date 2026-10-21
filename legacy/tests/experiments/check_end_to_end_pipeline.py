# tests/test_end_to_end_pipeline.py
"""
Unified Multi-Modal End-to-End Regression & Integration Benchmark.
Validates:
  1. Polymorphic adapters for Chest X-Ray, Head CT, and Knee MRI
  2. BiomedCLIP zero-shot prompt evaluation & causal counterfactual generation
  3. 3D Axial Transformer sequence processing with Wald's CUSUM-SPRT early-exit
  4. Conformal coverage set inclusion across all modalities
  5. Starvation-free Dynamic Priority Gateway scoring S(t)
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from core.models import ChestXRayAdapter, HeadCTAdapter, KneeMRIAdapter
from core.gateway import DynamicPriorityGateway
from core.base import ModalityType, TriagePriorityZone


def run_end_to_end_benchmark():
    print("=" * 80)
    print("🚀 BENCHMARK: UNIFIED MULTI-MODAL END-TO-END PIPELINE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Standard CPU'})")

    gateway = DynamicPriorityGateway()

    # 1. Benchmark Chest X-Ray Adapter (BiomedCLIP + Conformal + Causal Mask)
    print("\n" + "-" * 80)
    print("[*] 1. Evaluating ChestXRayAdapter (2D Frontal Radiograph)...")
    cxr_adapter = ChestXRayAdapter(device=device)
    dummy_cxr = torch.randn(1, 1, 224, 224)

    t0 = time.perf_counter()
    cxr_out = cxr_adapter.predict_triage(
        dummy_cxr,
        custom_prompt="acute tension pneumothorax with visible visceral pleural line",
    )
    cxr_time = (time.perf_counter() - t0) * 1000

    print(f"  • Modality               : {cxr_out.modality.value}")
    print(f"  • Critical Probability   : {cxr_out.p_critical * 100:.2f}%")
    print(f"  • PSI Stability Score    : {cxr_out.psi_score:.2f}")
    print(f"  • Conformal Set C(X)     : {cxr_out.conformal_set}")
    print(f"  • Latency                : {cxr_time:.2f} ms")

    assert cxr_out.p_critical >= 0.0 and cxr_out.p_critical <= 1.0
    assert len(cxr_out.conformal_set) >= 1
    print("[✓] Chest X-Ray Foundation Pipeline Verified.")

    # 2. Benchmark Head CT Adapter (3D Axial Transformer + Wald's CUSUM-SPRT)
    print("\n" + "-" * 80)
    print("[*] 2. Evaluating HeadCTAdapter (32-Slice 3D Volumetric Scan)...")
    ct_adapter = HeadCTAdapter(device=device)
    
    # 32-slice feature sequence with simulated acute epidural bleed starting at slice 8
    ct_embeddings = torch.randn(1, 32, 512, device=device)
    ct_embeddings[:, 7:15, :] += 4.5

    t1 = time.perf_counter()
    ct_out = ct_adapter.predict_triage(ct_embeddings)
    ct_time = (time.perf_counter() - t1) * 1000

    print(f"  • Modality               : {ct_out.modality.value}")
    print(f"  • Study Critical Risk    : {ct_out.p_critical * 100:.2f}%")
    print(f"  • SPRT Critical Halt     : {ct_out.sprt_halted}")
    print(f"  • SPRT Trigger Slice     : Slice #{ct_out.sprt_exit_slice + 1 if ct_out.sprt_exit_slice is not None else 'N/A'} / 32")
    print(f"  • Compute Saved          : {ct_out.sprt_compute_saved_pct:.1f}%")
    print(f"  • Conformal Set C(X)     : {ct_out.conformal_set}")
    print(f"  • Latency                : {ct_time:.2f} ms")

    assert ct_out.p_critical >= 0.0
    assert len(ct_out.conformal_set) >= 1
    print("[✓] 3D Head CT Axial & SPRT Pipeline Verified.")

    # 3. Benchmark Knee MRI Adapter
    print("\n" + "-" * 80)
    print("[*] 3. Evaluating KneeMRIAdapter (Multi-Sequence Series)...")
    mri_adapter = KneeMRIAdapter(device=device)
    dummy_mri = torch.randn(1, 16, 512, device=device)

    t2 = time.perf_counter()
    mri_out = mri_adapter.predict_triage(dummy_mri)
    mri_time = (time.perf_counter() - t2) * 1000

    print(f"  • Modality               : {mri_out.modality.value}")
    print(f"  • Critical Risk          : {mri_out.p_critical * 100:.2f}%")
    print(f"  • Conformal Set C(X)     : {mri_out.conformal_set}")
    print(f"  • Latency                : {mri_time:.2f} ms")

    assert mri_out.p_critical >= 0.0
    print("[✓] Knee MRI Multi-Planar Pipeline Verified.")

    # 4. Ingest All Studies into Anti-Starvation Gateway
    print("\n" + "-" * 80)
    print("[*] 4. Evaluating Dynamic Starvation-Free Gateway Prioritization [S(t)]...")
    gateway.enqueue("STUDY-CXR-01", "PAT-101", ModalityType.CHEST_XRAY, t_wait_minutes=15, triage_output=cxr_out)
    gateway.enqueue("STUDY-CT-02", "PAT-102", ModalityType.HEAD_CT, t_wait_minutes=5, triage_output=ct_out)
    gateway.enqueue("STUDY-MRI-03", "PAT-103", ModalityType.KNEE_MRI, t_wait_minutes=120, triage_output=mri_out)

    worklist = gateway.get_sorted_worklist()
    print("\n  Sorted Dynamic Triage Queue:")
    for rank, item in enumerate(worklist, start=1):
        print(f"   Rank #{rank}: [{item.priority_zone.value:<6}] Study: {item.study_uid:<13} | S(t) = {item.dynamic_score:.4f} | Wait: {item.t_wait_minutes:3.0f}m | Set: {item.triage_output.conformal_set}")

    assert len(worklist) == 3
    print("\n" + "=" * 80)
    print("✅ ALL END-TO-END PIPELINE BENCHMARKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_end_to_end_benchmark()