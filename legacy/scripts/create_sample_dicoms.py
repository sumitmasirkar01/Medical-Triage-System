# create_sample_dicoms.py
"""Generates physical binary .dcm files with compliant DICOM metadata headers."""

import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import datetime

os.makedirs("sample_dicoms", exist_ok=True)

def save_as_dicom(filename: str, pixel_array_2d: np.ndarray, modality: str, patient_id: str, is_critical: bool):
    file_path = os.path.join("sample_dicoms", filename)
    
    # 1. Create DICOM file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2" if modality == "CT" else "1.2.840.10008.5.1.4.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    # 2. Build Dataset
    ds = FileDataset(file_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Patient & Study Attributes
    ds.PatientName = f"Simulated^{patient_id}"
    ds.PatientID = patient_id
    ds.Modality = modality
    ds.StudyDate = datetime.date.today().strftime("%Y%m%d")
    ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
    ds.StudyDescription = "STAT Emergency Triage Benchmark Scan"
    ds.SeriesDescription = f"Emergency {modality} Protocol"
    
    # Spatial Attributes
    ds.Rows, ds.Columns = pixel_array_2d.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [0.5, 0.5]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # Signed integer for Hounsfield Units
    
    # CT Physics Rescale Calibration
    ds.RescaleIntercept = -1024.0
    ds.RescaleSlope = 1.0
    ds.WindowCenter = 40.0
    ds.WindowWidth = 80.0
    
    # Convert normalized [0, 1] back into raw 16-bit CT detector values (Hounsfield Units + 1024)
    # 0.0 -> -1024 HU (Air), 0.42 -> ~40 HU (Brain), 0.92 -> ~80 HU (Bleed)
    raw_hu = (pixel_array_2d * 1200.0) - 1000.0
    raw_detector_values = (raw_hu + 1024.0).astype(np.int16)
    
    ds.PixelData = raw_detector_values.tobytes()
    ds.save_as(file_path, write_like_original=False)
    print(f"✅ Generated Physical DICOM: {file_path} ({os.path.getsize(file_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    from dashboard import generate_head_ct_phantom, generate_chest_xray_phantom, generate_knee_mri_phantom
    
    print("=" * 70)
    print("🩻 GENERATING PHYSICAL CLINICAL DICOM FILES (.dcm)")
    print("=" * 70)
    
    # 1. Acute Subdural/ICH Head CT Slice
    ct_vol, _ = generate_head_ct_phantom(is_critical=True)
    save_as_dicom("STAT_BLEED_HEAD_CT.dcm", ct_vol[16, 0], "CT", "PT-STAT-001", is_critical=True)
    
    # 2. Normal Routine Head CT Slice
    ct_norm, _ = generate_head_ct_phantom(is_critical=False)
    save_as_dicom("NORMAL_HEAD_CT.dcm", ct_norm[16, 0], "CT", "PT-NORM-002", is_critical=False)
    
    # 3. STAT Pneumothorax Chest X-Ray
    cxr_crit, _ = generate_chest_xray_phantom(is_critical=True)
    save_as_dicom("STAT_PNEUMO_CHEST_XRAY.dcm", cxr_crit[0], "CR", "PT-STAT-003", is_critical=True)

    # 4. Normal Routine Chest X-Ray
    cxr_norm, _ = generate_chest_xray_phantom(is_critical=False)
    save_as_dicom("NORMAL_CHEST_XRAY.dcm", cxr_norm[0], "CR", "PT-NORM-004", is_critical=False)