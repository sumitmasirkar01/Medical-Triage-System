from datetime import datetime, timezone

from stat_triage.contracts import Modality, ModelEvidence
from stat_triage.multimodal import (
    MODALITY_REGISTRY,
    ChestXrayRequest,
    HeadCTRequest,
    KneeMRIRequest,
    UnifiedInferenceRouter,
)


def evidence(modality: Modality, exam_id: str) -> ModelEvidence:
    return ModelEvidence(
        exam_id=exam_id,
        patient_id="P1",
        modality=modality,
        endpoint="test_endpoint",
        model_id="test_model",
        model_sha256="a" * 64,
        preprocess_version="test_v1",
        raw_scores={"finding": 0.8},
        calibrated_scores={"finding": 0.7},
        thresholds={"finding": 0.5},
        decisions={"finding": True},
        latency_ms=1.0,
        input_hashes={"input": "b" * 64},
        created_at=datetime.now(timezone.utc),
    )


class CTStub:
    class Legacy:
        def to_model_evidence(self, **_):
            return evidence(Modality.HEAD_CT, "CT1")

    def predict_file(self, _):
        return self.Legacy()


class CXRStub:
    def predict_file(self, _, *, patient_id=None, exam_id=None):
        return evidence(Modality.CHEST_XRAY, exam_id or "CXR1")


class MRIStub:
    def predict_exam(self, **values):
        return evidence(Modality.KNEE_MRI, values["exam_id"])


def router():
    return UnifiedInferenceRouter(
        adapter_factories={
            Modality.HEAD_CT: CTStub,
            Modality.CHEST_XRAY: CXRStub,
            Modality.KNEE_MRI: MRIStub,
        }
    )


def test_registry_declares_three_distinct_input_contracts():
    assert set(MODALITY_REGISTRY) == set(Modality)
    assert MODALITY_REGISTRY[Modality.HEAD_CT].evaluation_level == "slice"
    assert MODALITY_REGISTRY[Modality.KNEE_MRI].input_roles == (
        "axial", "coronal", "sagittal"
    )


def test_router_lazily_routes_head_ct(tmp_path):
    source = tmp_path / "slice.dcm"
    source.write_bytes(b"dicom")
    instance = router()
    result = instance.predict(HeadCTRequest(source))
    assert result.modality is Modality.HEAD_CT
    assert instance.loaded_modalities == (Modality.HEAD_CT,)


def test_router_routes_chest_xray_without_loading_other_models(tmp_path):
    result = router().predict(
        ChestXrayRequest(tmp_path / "cxr.png", exam_id="CXR-7")
    )
    assert result.modality is Modality.CHEST_XRAY
    assert result.exam_id == "CXR-7"


def test_router_routes_complete_three_plane_mri_request(tmp_path):
    result = router().predict(
        KneeMRIRequest(
            axial_path=tmp_path / "axial.npy",
            coronal_path=tmp_path / "coronal.npy",
            sagittal_path=tmp_path / "sagittal.npy",
            exam_id="MRI-9",
        )
    )
    assert result.modality is Modality.KNEE_MRI
    assert result.exam_id == "MRI-9"


def test_router_can_unload_one_or_all_cached_models(tmp_path):
    instance = router()
    instance.predict(ChestXrayRequest(tmp_path / "cxr.png"))
    instance.predict(
        KneeMRIRequest(
            axial_path=tmp_path / "a.npy", coronal_path=tmp_path / "c.npy",
            sagittal_path=tmp_path / "s.npy", exam_id="MRI",
        )
    )
    assert instance.unload(Modality.CHEST_XRAY) == (Modality.CHEST_XRAY,)
    assert instance.loaded_modalities == (Modality.KNEE_MRI,)
    assert instance.unload() == (Modality.KNEE_MRI,)
    assert instance.loaded_modalities == ()
