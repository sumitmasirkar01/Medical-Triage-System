"""Create a de-identified CT DICOM fixture that satisfies the demo contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian


def create_demo_ct(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.10.999.1.1.1"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.999.1"

    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.PatientID = "DEMO-001"
    dataset.PatientName = "DEIDENTIFIED^DEMO"
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.10.999.1.2"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.10.999.1.3"
    dataset.Modality = "CT"
    dataset.InstanceNumber = 1
    dataset.ImagePositionPatient = [0.0, 0.0, 0.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.Rows = dataset.Columns = 512
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = -1024.0
    dataset.PixelSpacing = [0.5, 0.5]
    dataset.SliceThickness = 1.0

    y, x = np.ogrid[:512, :512]
    radius = np.sqrt((x - 256) ** 2 + (y - 256) ** 2)
    hu = np.full((512, 512), -1000, dtype=np.int16)
    hu[radius < 190] = 40
    hu[(radius >= 190) & (radius < 205)] = 1000
    raw = (hu.astype(np.int32) + 1024).clip(0, 65535).astype(np.uint16)
    dataset.PixelData = raw.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_demo_ct(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
