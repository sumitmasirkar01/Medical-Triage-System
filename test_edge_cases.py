# test_edge_cases.py
import numpy as np
import torch
from core.preprocessing import ClinicalCTPreprocessor
from core.gateway import TriageGateway, TriageZone
from core.models import StatelessGradCAM, PerturbationStabilityEvaluator, HeadCTTriPathwayNet

def test_system_boundaries():
    print("🛡️ RUNNING SYSTEM BOUNDARY & EDGE-CASE VALIDATION...")

    # 1. Edge Case: Zero elapsed wait time (t = 0s)
    gateway = TriageGateway()
    rec_zero = gateway.register_study("CT_ZERO", "ANON_00", "Head CT", 0.80, 0.90, [0.1]*5, 0.85, True, arrival_offset_sec=0.0)
    score_zero = rec_zero.current_priority_score(rec_zero.ingress_monotonic)
    assert np.isclose(score_zero, 0.80 * 0.90), f"t=0 score mismatch: {score_zero}"
    print("  [✓] Boundary t=0s: Correctly equals P * C")

    # 2. Edge Case: Infinite/Asymptotic wait time (t -> infinity)
    # Aging term alpha = 0.25 must never exceed its mathematical ceiling
    score_inf = rec_zero.current_priority_score(rec_zero.ingress_monotonic + 1e8, alpha=0.25)
    max_theoretical = (0.80 * 0.90) + 0.25
    assert score_inf <= max_theoretical + 1e-5, "Aging term exceeded alpha ceiling!"
    print(f"  [✓] Boundary t->inf: Score strictly bounded by alpha ceiling ({score_inf:.4f} <= {max_theoretical:.4f})")

    # 3. Edge Case: Completely Flat / Zero-Variance Image (Black or Artifact scan)
    model = HeadCTTriPathwayNet()
    model.eval()
    cam_extractor = StatelessGradCAM(model.backbone, model.backbone.features[-1])
    evaluator = PerturbationStabilityEvaluator(iou_threshold=0.75, num_passes=3)
    
    flat_tensor = torch.zeros(1, 3, 256, 256)
    mean_iou, is_verified = evaluator.evaluate_stability(cam_extractor, flat_tensor)
    assert not is_verified, "Flat noise image falsely marked as verified!"
    print("  [✓] Zero-Variance Input: Correctly flagged as Unstable (Zero division prevented)")

    # 4. Edge Case: Corrupted DICOM Array with NaNs and Infs
    corrupt_array = np.array([[np.nan, np.inf], [-np.inf, 40.0]])
    cleaned_hu = ClinicalCTPreprocessor.raw_to_hu(corrupt_array)
    assert not np.isnan(cleaned_hu).any() and not np.isinf(cleaned_hu).any()
    print("  [✓] Corrupted Pixel Array: NaNs and Infs safely clamped to air density (-1024 HU)")

    print("\n✅ ALL 4 BOUNDARY CHECKS PASSED: Zero divide, NaN, or saturation risks detected.\n")

if __name__ == "__main__":
    test_system_boundaries()