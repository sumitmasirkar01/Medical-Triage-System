"""Head CT series ingestion rejects incomplete or internally mixed requests."""

from pathlib import Path

import pydicom
import pytest

import stat_triage.ct_series as ct_series
from scripts.create_demo_dicom import create_demo_ct
from stat_triage.ct_series import (
    CTSeriesInputError,
    CTSeriesIssue,
    HeadCTStudyInferenceAdapter,
    load_locked_ct_study_protocol,
    validate_head_ct_series,
)


def series(tmp_path: Path, count: int = 10) -> list[Path]:
    paths = []
    for index in range(count):
        path = tmp_path / f"slice-{index:03d}.dcm"
        create_demo_ct(path)
        dataset = pydicom.dcmread(path)
        dataset.SOPInstanceUID = f"1.2.826.0.1.3680043.10.999.1.1.{index + 1}"
        dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
        dataset.InstanceNumber = index + 1
        dataset.ImagePositionPatient = [0.0, 0.0, float(20 - index * 2)]
        dataset.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


def test_valid_series_is_deterministically_ordered(tmp_path):
    result = validate_head_ct_series(reversed(series(tmp_path)))

    assert len(result.slices) == 10
    assert result.ordering_basis == "image_position_patient_z"
    assert [item.position for item in result.slices] == sorted(
        item.position for item in result.slices
    )
    assert result.patient_id == "DEMO-001"


def test_incomplete_series_is_unsupported(tmp_path):
    with pytest.raises(CTSeriesInputError) as captured:
        validate_head_ct_series(series(tmp_path, 9))
    assert captured.value.issue is CTSeriesIssue.UNSUPPORTED
    assert captured.value.code == "incomplete_series"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("PatientID", "OTHER", "mixed_patient"),
        ("StudyInstanceUID", "1.2.3.4", "mixed_study"),
        ("SeriesInstanceUID", "1.2.3.5", "mixed_series"),
    ],
)
def test_mixed_identity_is_rejected(tmp_path, field, value, code):
    paths = series(tmp_path)
    dataset = pydicom.dcmread(paths[-1])
    setattr(dataset, field, value)
    dataset.save_as(paths[-1], enforce_file_format=True)

    with pytest.raises(CTSeriesInputError) as captured:
        validate_head_ct_series(paths)
    assert captured.value.code == code


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("SOPInstanceUID", "duplicate_sop_instance"),
        ("ImagePositionPatient", "duplicate_slice_position"),
    ],
)
def test_duplicate_identity_or_position_is_rejected(tmp_path, field, code):
    paths = series(tmp_path)
    first = pydicom.dcmread(paths[0])
    last = pydicom.dcmread(paths[-1])
    setattr(last, field, getattr(first, field))
    if field == "SOPInstanceUID":
        last.file_meta.MediaStorageSOPInstanceUID = last.SOPInstanceUID
    last.save_as(paths[-1], enforce_file_format=True)

    with pytest.raises(CTSeriesInputError) as captured:
        validate_head_ct_series(paths)
    assert captured.value.code == code


def test_wrong_modality_is_unsupported(tmp_path):
    paths = series(tmp_path)
    dataset = pydicom.dcmread(paths[0])
    dataset.Modality = "MR"
    dataset.save_as(paths[0], enforce_file_format=True)

    with pytest.raises(CTSeriesInputError) as captured:
        validate_head_ct_series(paths)
    assert captured.value.code == "unsupported_modality"


def test_missing_interior_slice_is_detected_from_irregular_spacing(tmp_path):
    paths = series(tmp_path)
    dataset = pydicom.dcmread(paths[-1])
    dataset.ImagePositionPatient = [0.0, 0.0, -20.0]
    dataset.save_as(paths[-1], enforce_file_format=True)

    with pytest.raises(CTSeriesInputError) as captured:
        validate_head_ct_series(paths)
    assert captured.value.code == "irregular_slice_spacing"


def test_total_series_resource_limit_is_enforced_before_header_reads(tmp_path, monkeypatch):
    paths = series(tmp_path)
    monkeypatch.setattr(ct_series, "MAX_HEAD_CT_SERIES_BYTES", 1)

    with pytest.raises(CTSeriesInputError) as captured:
        validate_head_ct_series(paths)
    assert captured.value.code == "series_too_large"


def test_repository_locked_study_protocol_is_hash_verified():
    protocol = load_locked_ct_study_protocol()
    assert protocol["selected_method"] == "mean"
    assert protocol["operating_point"]["threshold"] == pytest.approx(0.20713453119969927)


def test_study_adapter_mean_pools_slices_into_common_evidence(tmp_path):
    paths = series(tmp_path)

    class SliceResult:
        patient_id = "DEMO-001"
        study_instance_uid = "1.2.826.0.1.3680043.10.999.1.2"
        any_score = 0.3

    class SliceAdapter:
        model_epoch = 4
        model_sha256 = "a" * 64

        def predict_file(self, path):
            return SliceResult()

    protocol = {
        "selected_method": "mean",
        "operating_point": {"threshold": 0.2},
    }
    result = HeadCTStudyInferenceAdapter(SliceAdapter(), protocol=protocol).predict_series(paths)

    assert result.endpoint == "study_level_any_hemorrhage"
    assert result.raw_scores["any"] == pytest.approx(0.3)
    assert result.decisions["any"] is True
    assert result.study_instance_uid == "1.2.826.0.1.3680043.10.999.1.2"
    assert len(result.input_hashes) == 10
