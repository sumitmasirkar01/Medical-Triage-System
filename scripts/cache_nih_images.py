"""Decode train/validation PNGs once into a contiguous uint8 NumPy cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from stat_triage.datasets.nih_training import NIHCacheSourceDataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/nih_cache"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(args.manifest)
    selected = full[full["split"].isin(["train", "validation"])].reset_index(drop=True)
    selected.insert(0, "cache_index", np.arange(len(selected), dtype=np.int64))
    dataset = NIHCacheSourceDataset(selected, size=args.size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    cache_path = args.output_dir / "train_validation_images_uint8.npy"
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.uint8,
        shape=(len(selected), args.size, args.size),
    )
    started = time.perf_counter()
    written = 0
    for batch_number, batch in enumerate(loader, start=1):
        images = batch["image"].numpy()
        indices = batch["cache_index"].numpy()
        cache[indices] = images
        written += len(indices)
        if batch_number == 1 or batch_number % 50 == 0 or written == len(selected):
            elapsed = time.perf_counter() - started
            rate = written / elapsed if elapsed else 0.0
            print(
                f"Cached {written:,}/{len(selected):,} images "
                f"({rate:.1f} images/s, {elapsed / 60:.1f} min)",
                flush=True,
            )
    cache.flush()
    del cache
    manifest_path = args.output_dir / "train_validation_cache_manifest.csv"
    selected.to_csv(manifest_path, index=False)
    report = {
        "status": "complete",
        "included_splits": ["train", "validation"],
        "calibration_included": False,
        "test_included": False,
        "images": len(selected),
        "shape": [len(selected), args.size, args.size],
        "dtype": "uint8",
        "bytes": cache_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
    }
    report_path = args.output_dir / "cache_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
