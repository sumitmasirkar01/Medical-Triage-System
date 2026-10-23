"""Lock and evaluate the prespecified CT Phase 2 selective policy locally."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pandas as pd

from stat_triage.ct_selective import (
    evaluate_locked_ct_selective_policy,
    lock_ct_selective_policy,
)


ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def member(bundle: Path, name: str) -> bytes:
    with zipfile.ZipFile(bundle) as archive:
        return archive.read(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", type=Path, default=ROOT / "artifacts" / "rsna_phase2"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts" / "rsna_phase2" / "selective_analysis",
    )
    parser.add_argument("--minimum-coverage", type=float, default=0.80)
    parser.add_argument("--maximum-automatic-fnr", type=float, default=0.05)
    parser.add_argument("--exploratory-minimum-coverage", type=float, default=0.50)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    locked_bundle = args.artifact_dir / "rsna_phase2_locked_protocol_bundle.zip"
    final_bundle = args.artifact_dir / "rsna_phase2_final_evidence_bundle.zip"
    protocol_bytes = member(locked_bundle, "locked_study_protocol.json")
    calibration_bytes = member(locked_bundle, "study_calibration_predictions.csv")
    protocol = json.loads(protocol_bytes)
    calibration = pd.read_csv(BytesIO(calibration_bytes))
    primary_policy = {
        "minimum_coverage": args.minimum_coverage,
        "maximum_automatic_false_negative_rate": args.maximum_automatic_fnr,
        "feasible": True,
    }
    try:
        policy = lock_ct_selective_policy(
            calibration, protocol, minimum_coverage=args.minimum_coverage,
            maximum_automatic_false_negative_rate=args.maximum_automatic_fnr,
        )
        policy["analysis_status"] = "prespecified_secondary"
    except ValueError as exc:
        primary_policy["feasible"] = False
        primary_policy["failure"] = str(exc)
        policy = lock_ct_selective_policy(
            calibration, protocol,
            minimum_coverage=args.exploratory_minimum_coverage,
            maximum_automatic_false_negative_rate=args.maximum_automatic_fnr,
        )
        policy["analysis_status"] = "post_test_exploratory"
    policy["prespecified_primary_policy"] = primary_policy
    policy["provenance"] = {
        "study_protocol_sha256": digest(protocol_bytes),
        "calibration_predictions_sha256": digest(calibration_bytes),
    }
    # The policy is complete before the test member is loaded.
    test_bytes = member(final_bundle, "study_test_predictions.csv")
    test = pd.read_csv(BytesIO(test_bytes))
    report = evaluate_locked_ct_selective_policy(
        test, policy, bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["analysis_status"] = policy["analysis_status"]
    report["prespecified_primary_policy"] = primary_policy
    report["provenance"] = {
        "selective_policy_sha256": digest(
            (json.dumps(policy, indent=2) + "\n").encode("utf-8")
        ),
        "study_test_predictions_sha256": digest(test_bytes),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.output_dir / "locked_selective_policy.json"
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    # Recompute after serialization so the reported hash is the exact saved policy.
    report["provenance"]["selective_policy_sha256"] = hashlib.sha256(
        policy_path.read_bytes()
    ).hexdigest()
    report_path = args.output_dir / "selective_test_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"policy": policy, "test": report}, indent=2))


if __name__ == "__main__":
    main()
