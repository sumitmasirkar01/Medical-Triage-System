import pandas as pd
import pytest

from stat_triage.ct_study import (
    StudyAggregationError, aggregate_study_predictions,
    evaluate_study_pooling_protocol, lock_study_pooling_protocol,
)


def _cohort(slices=3):
    manifest, predictions = [], []
    index = 0
    for split in ("validation", "calibration", "test"):
        for study_number, target in enumerate((0, 0, 0, 0, 0, 1, 1, 1, 1, 1)):
            for z in range(slices):
                image = f"i{index}"
                manifest.append({"image_id": image, "patient_id": f"p-{split}-{study_number}",
                                 "study_uid": f"s-{split}-{study_number}", "z_position": z, "split": split})
                base = 0.1 if target == 0 else 0.3
                score = base + (0.6 if target and z == slices - 1 else 0.0)
                predictions.append({"image_id": image, "target_any": target, "score_any": score})
                index += 1
    return pd.DataFrame(predictions), pd.DataFrame(manifest)


def test_study_pooling_methods_and_order_independence():
    predictions, manifest = _cohort()
    maximum, audit = aggregate_study_predictions(
        predictions.sample(frac=1, random_state=2), manifest, method="max",
        minimum_median_slices=3,
    )
    means, _ = aggregate_study_predictions(
        predictions, manifest.sample(frac=1, random_state=3), method="mean",
        minimum_median_slices=3,
    )
    assert audit["median_slices_per_study"] == 3
    assert maximum.sort_values("study_uid")["score_any"].max() == pytest.approx(0.9)
    assert means.sort_values("study_uid")["score_any"].max() == pytest.approx(0.5)


def test_protocol_selects_on_validation_and_evaluates_test():
    predictions, manifest = _cohort(4)
    report, test = evaluate_study_pooling_protocol(
        predictions, manifest, methods=("mean", "max"), minimum_median_slices=4,
        bootstrap_replicates=20,
    )
    assert report["selected_method"] in {"max", "mean"}
    assert report["test_not_used_for_selection"] is True
    assert report["test_metrics"]["roc_auc"] == 1.0
    assert len(test) == 10


def test_sparse_slice_pilot_is_rejected():
    predictions, manifest = _cohort(1)
    with pytest.raises(StudyAggregationError, match="Sparse study coverage"):
        aggregate_study_predictions(predictions, manifest, method="max")


def test_development_lock_never_loads_or_evaluates_test():
    predictions, manifest = _cohort(4)
    split_by_image = manifest.set_index("image_id")["split"]
    predictions["split"] = predictions["image_id"].map(split_by_image)
    development_predictions = predictions[predictions["split"] != "test"].copy()
    development_manifest = manifest[manifest["split"] != "test"].copy()
    report, calibration = lock_study_pooling_protocol(
        development_predictions, development_manifest, methods=("mean", "max"),
        minimum_median_slices=4,
    )
    assert report["status"] == "locked"
    assert report["test_rows_present"] == 0
    assert report["test_evaluated"] is False
    assert set(calibration["split"]) == {"calibration"}


def test_development_lock_rejects_any_test_partition():
    predictions, manifest = _cohort(4)
    predictions["split"] = predictions["image_id"].map(
        manifest.set_index("image_id")["split"]
    )
    with pytest.raises(StudyAggregationError, match="exactly validation and calibration"):
        lock_study_pooling_protocol(predictions, manifest, minimum_median_slices=4)


def test_development_lock_never_loads_or_evaluates_test():
    predictions, manifest = _cohort(4)
    split_by_image = manifest.set_index("image_id")["split"]
    predictions["split"] = predictions["image_id"].map(split_by_image)
    development_predictions = predictions[predictions["split"] != "test"].copy()
    development_manifest = manifest[manifest["split"] != "test"].copy()
    report, calibration = lock_study_pooling_protocol(
        development_predictions, development_manifest, methods=("mean", "max"),
        minimum_median_slices=4,
    )
    assert report["status"] == "locked"
    assert report["test_rows_present"] == 0
    assert report["test_evaluated"] is False
    assert set(calibration["split"]) == {"calibration"}


def test_development_lock_rejects_any_test_partition():
    predictions, manifest = _cohort(4)
    predictions["split"] = predictions["image_id"].map(
        manifest.set_index("image_id")["split"]
    )
    with pytest.raises(StudyAggregationError, match="exactly validation and calibration"):
        lock_study_pooling_protocol(predictions, manifest, minimum_median_slices=4)


def test_development_lock_never_loads_or_evaluates_test():
    predictions, manifest = _cohort(4)
    split_by_image = manifest.set_index("image_id")["split"]
    predictions["split"] = predictions["image_id"].map(split_by_image)
    development_predictions = predictions[predictions["split"] != "test"].copy()
    development_manifest = manifest[manifest["split"] != "test"].copy()
    report, calibration = lock_study_pooling_protocol(
        development_predictions, development_manifest, methods=("mean", "max"),
        minimum_median_slices=4,
    )
    assert report["status"] == "locked"
    assert report["test_rows_present"] == 0
    assert report["test_evaluated"] is False
    assert set(calibration["split"]) == {"calibration"}


def test_development_lock_rejects_any_test_partition():
    predictions, manifest = _cohort(4)
    predictions["split"] = predictions["image_id"].map(
        manifest.set_index("image_id")["split"]
    )
    with pytest.raises(StudyAggregationError, match="exactly validation and calibration"):
        lock_study_pooling_protocol(predictions, manifest, minimum_median_slices=4)
