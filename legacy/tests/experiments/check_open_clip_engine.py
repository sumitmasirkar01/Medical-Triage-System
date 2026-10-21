# tests/test_open_clip_engine.py
"""
Multi-Metric Verification & Benchmark Suite for Open-Vocabulary Foundation Engine.
"""

import os
import sys
import time

# Ensure project root is always in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from core.open_clip_engine import OpenBiomedicalCLIPEngine


def run_foundation_benchmarks():
    print("=" * 80)
    print("🚀 BENCHMARK: OPEN-VOCABULARY BIOMEDICAL FOUNDATION ENGINE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Standard CPU'})")

    # 1. Initialize Engine
    t0 = time.perf_counter()
    engine = OpenBiomedicalCLIPEngine(device=device)
    init_time = (time.perf_counter() - t0) * 1000
    print(f"[✓] Engine Initialized in {init_time:.2f} ms | Weights Loaded: {engine.is_loaded}")

    # 2. Test Open-Vocabulary Clinical Queries
    clinical_prompts = [
        "healthy clear lungs with no pneumothorax or effusion",
        "acute tension pneumothorax with visible visceral pleural line",
        "dense pulmonary consolidation and lobar pneumonia",
        "bilateral pleural effusion with fluid blunting costophrenic angles",
        "severe cardiomegaly with enlarged cardiac silhouette",
    ]

    print(f"\n[*] Evaluating Zero-Shot Semantic Grounding on {len(clinical_prompts)} Clinical Prompts:")
    sample_tensor = torch.randn(1, 1, 224, 224)

    t_infer_start = time.perf_counter()
    predictions = engine.zero_shot_triage(sample_tensor, candidate_queries=clinical_prompts, temperature=0.7)
    infer_time = (time.perf_counter() - t_infer_start) * 1000

    for query, prob in predictions.items():
        bar = "█" * int(prob * 30)
        print(f"  • {query[:45]:<47} : {prob:.4f} | {bar}")

    assert np.isclose(sum(predictions.values()), 1.0, atol=1e-4), "Softmax probability sum violation!"
    print(f"[✓] Mathematical Probability Axiom Verified (Σ P(y) = {sum(predictions.values()):.4f})")
    print(f"[✓] Zero-Shot Forward Pass Latency: {infer_time:.2f} ms")

    # 3. Benchmark Throughput (FPS) over 20 Iterations
    print("\n[*] Stress-Testing Latency & Throughput over 20 Forward Cycles...")
    latencies = []
    for _ in range(20):
        t_start = time.perf_counter()
        _ = engine.zero_shot_triage(sample_tensor, candidate_queries=clinical_prompts)
        latencies.append((time.perf_counter() - t_start) * 1000)

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    fps = 1000.0 / mean_lat

    print(f"  • Mean Latency  : {mean_lat:.2f} ms")
    print(f"  • P95 Latency   : {p95_lat:.2f} ms")
    print(f"  • Throughput    : {fps:.1f} FPS")

    # 4. Test Direct Spatial Patch Saliency Generation
    print("\n[*] Testing Gradient-Free Spatial Patch Saliency Extraction...")
    target_condition = "acute tension pneumothorax with visible visceral pleural line"
    t_cam_start = time.perf_counter()
    saliency_map = engine.extract_patch_saliency(sample_tensor, target_query=target_condition)
    cam_time = (time.perf_counter() - t_cam_start) * 1000

    assert saliency_map.shape == (224, 224), f"Saliency shape mismatch: {saliency_map.shape}"
    assert 0.0 <= saliency_map.min() and saliency_map.max() <= 1.0, "Saliency normalization bounded violation!"
    print(f"[✓] Patch Saliency Generated in {cam_time:.2f} ms | Shape: {saliency_map.shape} | Range: [{saliency_map.min():.2f}, {saliency_map.max():.2f}]")

    print("\n" + "=" * 80)
    print("✅ PHASE 1 FOUNDATION BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_foundation_benchmarks()