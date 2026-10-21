# scripts/extract_slice_embeddings.py
"""
STAT-Triage AI: High-Throughput Volumetric Feature Extraction & Caching Pipeline.
Converts raw physical DICOM scans into frozen 512-D BiomedCLIP feature sequences
for rapid training of the 3D Axial Cross-Attention Transformer.
"""

from pathlib import Path
from typing import List, Optional
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from core.dicom_parser import PhysicalDICOMParser
from core.open_clip_engine import OpenBiomedicalCLIPEngine

console = Console()


class VolumetricFeatureExtractor:
    """
    Batch extracts and caches 512-D slice embedding vectors from DICOM series.
    """

    def __init__(self, batch_size: int = 16, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parser = PhysicalDICOMParser()
        self.clip_engine = OpenBiomedicalCLIPEngine(device=self.device)
        self.batch_size = batch_size

    def process_study_series(self, dicom_paths: List[Path]) -> torch.Tensor:
        """
        Parses a sequence of slice files, resizes to 224x224, passes them through
        the frozen BiomedCLIP visual backbone, and returns a (num_slices, 512) tensor.
        """
        slice_tensors = []
        for path in dicom_paths:
            tensor, _ = self.parser.parse_file(path)  # (1, 3, H, W)
            # Bilinear interpolation down to ViT patch input resolution (224x224)
            tensor_resized = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
            slice_tensors.append(tensor_resized.squeeze(0))

        # Stack into batch: (num_slices, 3, 224, 224)
        vol_batch = torch.stack(slice_tensors, dim=0).to(self.device)
        num_slices = vol_batch.shape[0]

        # Extract features in mini-batches to prevent memory saturation
        embeddings_list = []
        with torch.no_grad():
            for i in range(0, num_slices, self.batch_size):
                chunk = vol_batch[i : i + self.batch_size]
                emb = self.clip_engine.encode_image(chunk)  # (chunk_size, 512)
                embeddings_list.append(emb.cpu())

        return torch.cat(embeddings_list, dim=0)  # (num_slices, 512)

    def cache_dataset(self, input_dir: Path, output_dir: Path):
        """
        Recursively discovers study directories, extracts embeddings, and saves .pt files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        study_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

        if not study_dirs:
            console.print(f"[bold yellow]No study subdirectories found in {input_dir}.[/bold yellow]")
            return

        console.print(f"[bold cyan]Found {len(study_dirs)} studies. Starting feature caching...[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Extracting embeddings...", total=len(study_dirs))

            for s_dir in study_dirs:
                dicom_files = sorted(list(s_dir.glob("*.dcm")))
                if not dicom_files:
                    progress.advance(task)
                    continue

                study_uid = s_dir.name
                embeddings = self.process_study_series(dicom_files)
                
                # Save as PyTorch tensor payload
                out_path = output_dir / f"{study_uid}.pt"
                torch.save(
                    {
                        "study_uid": study_uid,
                        "num_slices": embeddings.shape[0],
                        "embeddings": embeddings,  # (N, 512)
                    },
                    out_path,
                )
                progress.advance(task)

        console.print(f"[bold green]✓ Feature caching complete. Tensors saved to {output_dir}[/bold green]")


def run_pipeline_test():
    """Generates synthetic DICOM studies and tests the feature caching pipeline."""
    from tests.test_dicom_parser import create_synthetic_dicom_stream
    import shutil

    temp_input = Path("temp_test_dicoms")
    temp_output = Path("temp_cached_features")

    # Create dummy study directory with 8 slices
    study_dir = temp_input / "TEST_STUDY_001"
    study_dir.mkdir(parents=True, exist_ok=True)

    console.print("[*] Generating synthetic study with 8 DICOM slices...")
    for idx in range(8):
        raw_bytes = create_synthetic_dicom_stream(shape=(512, 512))
        with open(study_dir / f"slice_{idx:02d}.dcm", "wb") as f:
            f.write(raw_bytes)

    extractor = VolumetricFeatureExtractor(batch_size=4)
    extractor.cache_dataset(temp_input, temp_output)

    # Verify saved tensor payload
    cached_file = temp_output / "TEST_STUDY_001.pt"
    assert cached_file.exists(), "Feature cache file was not created!"
    payload = torch.load(cached_file)

    console.print(f"\n[bold green]Cached Payload Verification:[/bold green]")
    console.print(f"  • Study UID      : {payload['study_uid']}")
    console.print(f"  • Slices Cached  : {payload['num_slices']}")
    console.print(f"  • Embedding Dim  : {list(payload['embeddings'].shape)} (Expected: [8, 512])")

    assert payload["embeddings"].shape == (8, 512)
    assert not torch.isnan(payload["embeddings"]).any(), "Embedding tensor contains NaNs!"

    # Clean up temporary test files
    shutil.rmtree(temp_input)
    shutil.rmtree(temp_output)
    console.print("[bold green]✓ Caching pipeline self-test passed and verified.[/bold green]")


if __name__ == "__main__":
    run_pipeline_test()