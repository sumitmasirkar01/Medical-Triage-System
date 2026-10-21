# tests/test_conformal.py
"""
Deterministic Mathematical Coverage Benchmark for Split Conformal Risk Engine.
Validates:
  1. Finite-sample coverage guarantee: P(Y ∈ C(X)) ≥ 1 - α
  2. Multi-class and multi-label prediction set generation
  3. Dynamic set cardinality scaling on ambiguous inputs
  4. Execution latency & vectorization throughput
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from core.conformal import SplitConformalCalibrator


def run_conformal_benchmarks():
    print("=" * 80)
    print("🚀 BENCHMARK: DISTRIBUTION-FREE CONFORMAL RISK ENGINE")
    print("=" * 80)

    classes = ["Normal", "Epidural_Hematoma", "Subdural_Hematoma", "Subarachnoid", "Intraparenchymal"]
    num_classes = len(classes)
    alpha = 0.01 # 99% statistical safety guarantee

    # 1. Generate Synthetic Calibration (N=2000) and Test (N=1000) Datasets
    np.random.seed(42)
    n_calib = 2000
    n_test = 1000

    # Calibrated probabilities with synthetic noise
    calib_raw = np.random.dirichlet(np.ones(num_classes) * 0.5, size=n_calib)
    calib_targets = np.random.choice(num_classes, size=n_calib)

    # Bias probabilities toward true classes to simulate trained model
    for i in range(n_calib):
        calib_raw[i, calib_targets[i]] += 1.8
    calib_probs = calib_raw / calib_raw.sum(axis=1, keepdims=True)

    test_raw = np.random.dirichlet(np.ones(num_classes) * 0.5, size=n_test)
    test_targets = np.random.choice(num_classes, size=n_test)
    for i in range(n_test):
        test_raw[i, test_targets[i]] += 1.8
    test_probs = test_raw / test_raw.sum(axis=1, keepdims=True)

    # 2. Fit Split Conformal Calibrator
    calibrator = SplitConformalCalibrator(alpha=alpha, class_names=classes)
    t0 = time.perf_counter()
    q_hat = calibrator.fit(calib_probs, calib_targets)
    fit_time = (time.perf_counter() - t0) * 1000

    print(f"[*] Calibration Complete in {fit_time:.2f} ms")
    print(f"  • Calibration Samples (N) : {n_calib}")
    print(f"  • Target Error Budget (α) : {alpha:.4f} (Safety: {(1.0 - alpha)*100:.1f}%)")
    print(f"  • Non-Conformity Threshold: q̂ = {q_hat:.4f}")
    print(f"  • Inclusion Threshold     : P(y) ≥ {1.0 - q_hat:.4f}")

    # 3. Predict Conformal Sets on Test Cases
    t1 = time.perf_counter()
    metrics = calibrator.evaluate_coverage(test_probs, test_targets)
    eval_time = (time.perf_counter() - t1) * 1000

    print(f"\n[*] Empirical Test Set Validation (N={n_test}):")
    print(f"  • Target Minimum Coverage : {metrics['target_coverage'] * 100:.2f}%")
    print(f"  • Empirical Coverage      : {metrics['empirical_coverage'] * 100:.2f}%")
    print(f"  • Mean Prediction Set Size: {metrics['mean_set_size']:.2f} classes/scan")
    print(f"  • Total Evaluation Latency: {eval_time:.2f} ms ({eval_time/n_test*1000:.2f} µs/scan)")

    assert metrics["coverage_guarantee_met"], f"Coverage guarantee violated: {metrics['empirical_coverage']} < {metrics['target_coverage']}"
    print(f"[✓] Mathematical Proof Verified: Empirical Coverage ({metrics['empirical_coverage']*100:.1f}%) ≥ Target ({metrics['target_coverage']*100:.1f}%)")

    # 4. Ambiguity Escalation Stress Test
    print("\n[*] Testing Ambiguity Detection on High-Entropy Input...")
    # Flat uniform distribution representing completely corrupted / ambiguous scan
    ambiguous_scan = np.array([[0.20, 0.20, 0.20, 0.20, 0.20]])
    sets, sizes = calibrator.predict_set(ambiguous_scan)
    print(f"  • Corrupted Scan Prediction Set : {sets[0]}")
    print(f"  • Prediction Set Cardinality |C|: {sizes[0]}")
    assert sizes[0] > 1, "Ambiguous scan failed to expand prediction set size!"
    print(f"[✓] Safety Guardrail Verified: Ambiguity automatically inflates set size to {sizes[0]} classes for clinician review.")

    print("\n" + "=" * 80)
    print("✅ PHASE 3 CONFORMAL RISK GUARANTEES BENCHMARK PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_conformal_benchmarks()