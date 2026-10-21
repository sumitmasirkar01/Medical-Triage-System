import torch

from stat_triage.datasets.mrnet import MRNET_PLANES
from stat_triage.datasets.mrnet_training import collate_mrnet_exams
from stat_triage.models.mrnet_multiplane import MRNetMultiPlaneModel


def _item(exam_id: str, slices: int) -> dict:
    return {
        "exam_id": exam_id,
        "volumes": {
            plane: torch.ones(slices + index, 3, 32, 32)
            for index, plane in enumerate(MRNET_PLANES)
        },
        "targets": torch.tensor([1.0, 0.0, 1.0]),
        "label_mask": torch.ones(3),
    }


def test_collate_mrnet_exams_pads_slices_and_builds_masks():
    batch = collate_mrnet_exams([_item("0001", 2), _item("0002", 4)])
    assert batch["volumes"]["axial"].shape == (2, 4, 3, 32, 32)
    assert batch["slice_masks"]["axial"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert batch["targets"].shape == (2, 3)


def test_mrnet_model_returns_one_logit_vector_per_exam():
    batch = collate_mrnet_exams([_item("0001", 2), _item("0002", 3)])
    model = MRNetMultiPlaneModel(pretrained=False, slice_chunk_size=4).eval()
    with torch.no_grad():
        logits = model(batch["volumes"], batch["slice_masks"])
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
