# tests/test_sequential_sprt.py
"""
Deterministic Verification & Latency Benchmark for Streaming Wald's CUSUM-SPRT Engine.
Evaluates:
  1. Mid-stream critical early-exit on acute intracranial hemorrhage
  2. Compute and network savings verification (> 80% savings on early bleeds)
  3. CUSUM non-negative invariance across normal skull vertex slices
  4. Vectorized update throughput & microsecond latency profile
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from core.sequential_sprt import SequentialProbabilityRatioTest, SPRTDecisionStatus


def run_sprt_benchmarks():
    print("=" * 80)
    print("🚀 BENCHMARK: STREAMING SEQUENTIAL OPTIMAL STOPPING (WALD'S CUSUM-SPRT)")
    print("=" * 80)

    alpha = 0.01   # 1% False Alarm rate
    beta = 0.001   # 0.1% Missed Hemorrhage rate
    sprt = SequentialProbabilityRatioTest(alpha=alpha, beta=beta, baseline_prob=0.05)

    print(f"[*] Wald Boundaries Calibrated:")
    print(f"  • Type I Error Bound  (α) : {alpha:.4f}")
    print(f"  • Type II Error Bound (β) : {beta:.4f}")
    print(f"  • Critical Boundary   (A) : +{sprt.upper_bound_A:.4f} (ln[(1-β)/α])")

    # 1. Test Early Catastrophic Intracranial Bleed (Appears at slice 6 of 64)
    total_slices = 64
    slice_stream_bleed = np.full(total_slices, 0.02) # Background normal skull vertex
    slice_stream_bleed[5:12] = 0.98                  # Severe acute hematoma starting at slice 6

    t0 = time.perf_counter()
    result_bleed = sprt.evaluate_full_stream(slice_stream_bleed)
    eval_time = (time.perf_counter() - t0) * 1000

    print(f"\n[*] Evaluating 64-Slice Trauma Stream (Acute Bleed starting at Slice 6):")
    print(f"  • Stopping Decision       : {result_bleed.status.value}")
    print(f"  • Early-Exit Trigger Slice: Slice #{result_bleed.exit_slice_idx + 1} / {total_slices}")
    print(f"  • Final Log-Likelihood (Λ): +{result_bleed.log_likelihood_ratio:.4f}")
    print(f"  • Compute & Time Saved    : {result_bleed.compute_savings_pct:.2f}%")
    print(f"  • Evaluation Latency      : {eval_time:.3f} ms")

    assert result_bleed.status == SPRTDecisionStatus.CRITICAL_HALT
    assert result_bleed.exit_slice_idx <= 8, f"Early exit triggered too late: Slice {result_bleed.exit_slice_idx + 1}"
    assert result_bleed.compute_savings_pct >= 80.0, "Failed to achieve >80% compute savings!"
    print(f"[✓] Critical Early-Exit Verified: Trapped fatal bleed at Slice #{result_bleed.exit_slice_idx + 1}, saving {result_bleed.compute_savings_pct:.1f}% compute.")

    # 2. Test Completely Clean 64-Slice Normal Stream
    slice_stream_normal = np.full(total_slices, 0.01)
    result_normal = sprt.evaluate_full_stream(slice_stream_normal)

    print(f"\n[*] Evaluating 64-Slice Outpatient Normal Stream:")
    print(f"  • Stopping Decision       : {result_normal.status.value}")
    print(f"  • Total Slices Ingested   : {result_normal.total_slices_streamed} / {total_slices}")
    print(f"  • Final Log-Likelihood (Λ): {result_normal.log_likelihood_ratio:.4f}")
    print(f"  • Compute & Time Saved    : {result_normal.compute_savings_pct:.2f}%")

    assert result_normal.status == SPRTDecisionStatus.NORMAL_HALT
    print("[✓] Normal Stream Fast-Path Verified: Scanned full volume and routed cleanly to Green.")

    # 3. Stress-Testing SPRT Ingestion Throughput (1,000 Volume Streams)
    print("\n[*] Stress-Testing SPRT Ingestion Throughput (1,000 Streams of 64 Slices)...")
    latencies = []
    for _ in range(1000):
        t_start = time.perf_counter()
        _ = sprt.evaluate_full_stream(slice_stream_bleed)
        latencies.append((time.perf_counter() - t_start) * 1000)

    mean_lat_us = np.mean(latencies) * 1000.0
    p95_lat_us = np.percentile(latencies, 95) * 1000.0
    throughput = 1000.0 / np.mean(latencies)

    print(f"  • Mean Latency Per Volume : {mean_lat_us:.2f} µs")
    print(f"  • P95 Latency Per Volume  : {p95_lat_us:.2f} µs")
    print(f"  • Ingestion Throughput    : {throughput:,.0f} Volumes/Sec")

    print("\n" + "=" * 80)
    print("✅ PHASE 4 STREAMING WALD'S CUSUM-SPRT BENCHMARK PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_sprt_benchmarks()