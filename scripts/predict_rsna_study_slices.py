"""Export batched locked-checkpoint predictions for every Phase 2 study slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pydicom
import torch

from stat_triage.rsna import DEFAULT_BUNDLE_DIR, RSNAInferenceAdapter, RSNA_LABELS
from stat_triage.evaluation import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    rows = pd.read_csv(args.manifest)
    required = {"image_id", "dicom_path", "patient_id", "study_uid", "split", *RSNA_LABELS}
    if missing := required.difference(rows.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    adapter = RSNAInferenceAdapter(args.bundle, device=args.device)
    output = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows.iloc[start:start + args.batch_size]
        tensors = []
        for path in batch_rows["dicom_path"]:
            tensors.append(adapter.preprocessor.transform_dataset(pydicom.dcmread(path)))
        images = torch.stack(tensors).to(adapter.device)
        with torch.inference_mode():
            logits = adapter.model(images).cpu()
            scores = torch.sigmoid(logits)
        for offset, (_, row) in enumerate(batch_rows.iterrows()):
            item = {"image_id": row.image_id, "patient_id": row.patient_id,
                    "study_uid": row.study_uid, "split": row.split,
                    "target_any": int(row["any"])}
            for index, label in enumerate(RSNA_LABELS):
                item[f"target_{label}"] = int(row[label])
                item[f"logit_{label}"] = float(logits[offset, index])
                item[f"score_{label}"] = float(scores[offset, index])
            output.append(item)
        print(f"Predicted {min(start + args.batch_size, len(rows))}/{len(rows)} slices", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "rsna_phase2_slice_predictions.csv"
    pd.DataFrame(output).to_csv(destination, index=False)
    report = {"slices": len(output), "checkpoint_sha256": adapter.model_sha256,
              "manifest_sha256": sha256(args.manifest), "predictions_sha256": sha256(destination),
              "test_evaluated": False}
    (args.output_dir / "rsna_phase2_prediction_export.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
