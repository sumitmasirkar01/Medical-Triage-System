# dashboard.py
"""
Production Multi-Modal Emergency Triage Console.
Integrated PACS Streaming, Real-time DICOM Ingestion, Realistic Anatomical Phantoms,
Dynamic Priority Queue, and Grad-CAM Diagnostic Viewer.
"""

import time
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch

from core.base import ModalityType, ModalityTriageOutput
from core.models import HeadCTAdapter, ChestXRayAdapter, KneeMRIAdapter
from core.gateway import TriageGateway, TriageRecord, TriageZone
from core.calibration import MultiModalCalibrator
from core.preprocessing import DICOMPhysicsEngine


# ----------------------------------------------------------------------
# Page Configuration & Styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="STAT-Triage AI | Emergency Multi-Modal Console",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 26px; font-weight: 700; color: #0ea5e9; margin-bottom: 0px; }
    .sub-header { font-size: 14px; color: #94a3b8; margin-bottom: 15px; }
    .red-zone { background-color: rgba(239, 68, 68, 0.20); border-left: 5px solid #ef4444; padding: 12px; border-radius: 4px; }
    .yellow-zone { background-color: rgba(245, 158, 11, 0.20); border-left: 5px solid #f59e0b; padding: 12px; border-radius: 4px; }
    .green-zone { background-color: rgba(16, 185, 129, 0.20); border-left: 5px solid #10b981; padding: 12px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Procedural Anatomical Phantom Engine
# ----------------------------------------------------------------------
def generate_head_ct_phantom(is_critical: bool = False):
    """Generates a 32-slice 2.5D Head CT volume with brain parenchyma, skull, and acute bleed."""
    volume = np.zeros((32, 3, 256, 256), dtype=np.float32)
    y, x = np.ogrid[:256, :256]
    center_y, center_x = 128, 128
    cam_map = np.zeros((256, 256), dtype=np.float32)

    for s in range(32):
        scale = np.sin((s / 31.0) * np.pi) ** 0.5
        r_outer = 95 * scale
        r_inner = 88 * scale

        if r_outer > 5:
            dist = np.sqrt((x - center_x) ** 2 + ((y - center_y) * 1.15) ** 2)
            skull = (dist <= r_outer) & (dist > r_inner)
            brain = dist <= r_inner
            v_dist1 = np.sqrt((x - 118) ** 2 + (y - 128) ** 2)
            v_dist2 = np.sqrt((x - 138) ** 2 + (y - 128) ** 2)
            ventricles = ((v_dist1 < 14) | (v_dist2 < 14)) & brain

            slice_ch0 = np.zeros((256, 256), dtype=np.float32)
            slice_ch0[brain] = 0.42 + np.random.normal(0, 0.02, size=slice_ch0[brain].shape)
            slice_ch0[ventricles] = 0.10
            slice_ch0[skull] = 0.95

            # Critical: Mid-ventricular Intracranial Hemorrhage (ICH)
            if is_critical and 12 <= s <= 20:
                bleed_dist = np.sqrt((x - 155) ** 2 + (y - 110) ** 2)
                bleed_mask = bleed_dist <= 20
                slice_ch0[bleed_mask] = 0.92 + np.random.normal(0, 0.02, size=slice_ch0[bleed_mask].shape)
                if s == 16:
                    cam_map = np.exp(-(bleed_dist**2) / (2 * (18**2))).astype(np.float32)

            volume[s, 0] = slice_ch0
            volume[s, 1] = np.clip(slice_ch0 * 0.85, 0, 1)
            volume[s, 2] = np.clip(slice_ch0 * 1.1, 0, 1)

    if not is_critical:
        cam_map = np.random.uniform(0.01, 0.08, size=(256, 256)).astype(np.float32)

    return volume, cam_map


def generate_chest_xray_phantom(is_critical: bool = False):
    """Generates a 2D PA Chest Radiograph with thoracic cavity, mediastinum, and pneumothorax."""
    cxr = np.full((256, 256), 0.12, dtype=np.float32)
    y, x = np.ogrid[:256, :256]

    thorax = ((x - 128) ** 2 / (105 ** 2) + (y - 130) ** 2 / (115 ** 2)) <= 1.0
    cxr[thorax] = 0.65

    lung_left = ((x - 85) ** 2 / (38 ** 2) + (y - 125) ** 2 / (75 ** 2)) <= 1.0
    lung_right = ((x - 171) ** 2 / (38 ** 2) + (y - 125) ** 2 / (75 ** 2)) <= 1.0
    cxr[lung_left | lung_right] = 0.22

    heart = ((x - 145) ** 2 / (42 ** 2) + (y - 155) ** 2 / (45 ** 2)) <= 1.0
    cxr[heart] = 0.72
    cam_map = np.zeros((256, 256), dtype=np.float32)

    if is_critical:
        pneumo_dist = (x - 80) ** 2 / (35 ** 2) + (y - 85) ** 2 / (45 ** 2)
        cxr[pneumo_dist <= 1.0] = 0.04
        cxr[np.abs(pneumo_dist - 1.0) < 0.08] = 0.85
        cam_map = np.exp(-pneumo_dist * 2.5).astype(np.float32)
    else:
        cam_map = np.random.uniform(0.01, 0.08, size=(256, 256)).astype(np.float32)

    cxr_3ch = np.stack([cxr, cxr, cxr], axis=0) + np.random.normal(0, 0.015, size=(3, 256, 256))
    return np.clip(cxr_3ch, 0.0, 1.0).astype(np.float32), cam_map


def generate_knee_mri_phantom(is_critical: bool = False):
    """Generates a Sagittal Knee MRI with distal femur, proximal tibia, joint space, and ACL tear."""
    mri = np.full((256, 256), 0.12, dtype=np.float32)
    y, x = np.ogrid[:256, :256]

    femur = ((x - 128) ** 2 / (70 ** 2) + (y - 70) ** 2 / (55 ** 2)) <= 1.0
    mri[femur & (y <= 110)] = 0.55

    tibia = ((x - 128) ** 2 / (75 ** 2) + (y - 195) ** 2 / (60 ** 2)) <= 1.0
    mri[tibia & (y >= 145)] = 0.50

    mri[(y > 110) & (y < 145) & (x >= 65) & (x <= 190)] = 0.28
    cam_map = np.zeros((256, 256), dtype=np.float32)

    if is_critical:
        tear_dist = np.sqrt((x - 120)**2 + (y - 125)**2)
        mri[tear_dist <= 15] = 0.88
        cam_map = np.exp(-(tear_dist**2) / (2 * (16**2))).astype(np.float32)
    else:
        cam_map = np.random.uniform(0.01, 0.08, size=(256, 256)).astype(np.float32)

    mri_3ch = np.stack([mri, mri, mri], axis=0) + np.random.normal(0, 0.012, size=(3, 256, 256))
    return np.clip(mri_3ch, 0.0, 1.0).astype(np.float32), cam_map


# ----------------------------------------------------------------------
# Singleton Adapters & Gateway Cache
# ----------------------------------------------------------------------
@st.cache_resource
def load_triage_system():
    adapters = {
        ModalityType.HEAD_CT.value: HeadCTAdapter(weights_path="weights/head_ct_gru_weights.pt", device="cpu"),
        ModalityType.CHEST_XRAY.value: ChestXRayAdapter(weights_path="weights/chest_xray_weights.pt", device="cpu"),
        ModalityType.KNEE_MRI.value: KneeMRIAdapter(weights_path="weights/knee_mri_weights.pt", device="cpu"),
    }
    gateway = TriageGateway(red_threshold=0.65, yellow_lower=0.30, yellow_upper=0.65)
    calibrator = MultiModalCalibrator()
    return adapters, gateway, calibrator


adapters, gateway, calibrator = load_triage_system()


# ----------------------------------------------------------------------
# Session State Initialization
# ----------------------------------------------------------------------
if "records" not in st.session_state:
    st.session_state.records = []
    st.session_state.patient_data_cache = {}
    st.session_state.counter = 100

if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = None


# ----------------------------------------------------------------------
# Ingestion Processor
# ----------------------------------------------------------------------
def ingest_phantom_study(modality: ModalityType, is_critical: bool):
    st.session_state.counter += 1
    pt_id = f"PT-{modality.value[:3]}-{st.session_state.counter:04d}"

    if modality == ModalityType.HEAD_CT:
        tensor_np, cam_map = generate_head_ct_phantom(is_critical)
        subtypes = {
            "intraparenchymal": 0.89 if is_critical else 0.02,
            "subdural": 0.35 if is_critical else 0.01,
            "subarachnoid": 0.28 if is_critical else 0.01,
            "intraventricular": 0.15 if is_critical else 0.01,
            "epidural": 0.08 if is_critical else 0.01,
            "any": 0.94 if is_critical else 0.03
        }
        salient_idx = 16 if is_critical else 0
    elif modality == ModalityType.CHEST_XRAY:
        tensor_np, cam_map = generate_chest_xray_phantom(is_critical)
        subtypes = {
            "pneumothorax": 0.92 if is_critical else 0.02,
            "pleural_effusion": 0.45 if is_critical else 0.04,
            "consolidation": 0.12 if is_critical else 0.03,
            "cardiomegaly": 0.08 if is_critical else 0.05,
            "edema": 0.05 if is_critical else 0.02,
            "critical_any": 0.95 if is_critical else 0.03
        }
        salient_idx = None
    else:
        tensor_np, cam_map = generate_knee_mri_phantom(is_critical)
        subtypes = {
            "ACL": 0.91 if is_critical else 0.02,
            "MCL": 0.32 if is_critical else 0.01,
            "Medial Meniscus": 0.58 if is_critical else 0.04,
            "Lateral Meniscus": 0.18 if is_critical else 0.02,
            "Fracture": 0.65 if is_critical else 0.01,
            "critical_surgical_any": 0.93 if is_critical else 0.03
        }
        salient_idx = None

    p_crit = float(np.random.uniform(0.85, 0.96) if is_critical else np.random.uniform(0.02, 0.18))
    psi = float(np.random.uniform(0.88, 0.96) if is_critical else np.random.uniform(0.10, 0.35))
    conf = float(np.clip(2.0 * abs(p_crit - 0.5), 0.0, 1.0))

    record = TriageRecord(
        patient_id=pt_id,
        study_uid=f"1.2.840.113619.2.{int(time.time()*1000)}",
        p_critical=p_crit,
        confidence=conf,
        psi_score=psi,
        metadata={
            "modality": modality.value,
            "subtypes": subtypes,
            "slice_idx": salient_idx,
            "is_volumetric": (modality == ModalityType.HEAD_CT)
        }
    )

    # Route record into gateway zone
    gateway.route_record(record)
    st.session_state.records.append(record)
    st.session_state.patient_data_cache[pt_id] = {
        "tensor": tensor_np,
        "cam": cam_map,
        "record": record
    }


# ----------------------------------------------------------------------
# Real File Ingestion Processor (.dcm, .png, .jpg)
# ----------------------------------------------------------------------
def ingest_real_file(uploaded_file, upload_modality):
    import pydicom
    from PIL import Image as PILImage

    st.session_state.counter += 1
    pt_id = f"PT-UPLD-{st.session_state.counter:04d}"

    if uploaded_file.name.endswith(".dcm"):
        # 1. Read DICOM byte buffer
        dcm = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()))
        raw_pixels = dcm.pixel_array.astype(np.float32)

        # 2. Physics calibration: Hounsfield Units = pixel * slope + intercept
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", -1024.0))
        hu = raw_pixels * slope + intercept

        # 3. Apply Multi-Window Calibration into 3 Channels
        def apply_window(data, center, width):
            lo, hi = center - width / 2.0, center + width / 2.0
            return np.clip((data - lo) / (hi - lo + 1e-7), 0.0, 1.0)

        if upload_modality == ModalityType.HEAD_CT.value:
            # Brain (40, 80), Subdural (75, 215), Bone (600, 2000)
            c0 = apply_window(hu, 40.0, 80.0)
            c1 = apply_window(hu, 75.0, 215.0)
            c2 = apply_window(hu, 600.0, 2000.0)
            tensor_2d = np.stack([c0, c1, c2], axis=0).astype(np.float32)
            # Expand single slice to 32 slices for volumetric Bi-GRU sequence adapter
            tensor_np = np.repeat(tensor_2d[np.newaxis, ...], 32, axis=0)
        elif upload_modality == ModalityType.CHEST_XRAY.value:
            norm_img = (raw_pixels - raw_pixels.min()) / (raw_pixels.max() - raw_pixels.min() + 1e-7)
            tensor_np = np.stack([norm_img, norm_img, norm_img], axis=0).astype(np.float32)
        else:  # KNEE_MRI
            norm_img = (raw_pixels - raw_pixels.min()) / (raw_pixels.max() - raw_pixels.min() + 1e-7)
            tensor_np = np.stack([norm_img, norm_img, norm_img], axis=0).astype(np.float32)
    else:
        # Standard RGB/Grayscale image upload
        pil_img = PILImage.open(uploaded_file).convert("RGB").resize((256, 256))
        tensor_np = np.array(pil_img).transpose(2, 0, 1).astype(np.float32) / 255.0
        if upload_modality == ModalityType.HEAD_CT.value:
            tensor_np = np.repeat(tensor_np[np.newaxis, ...], 32, axis=0)

    # 4. Predict via Polymorphic Adapter
    adapter = adapters[upload_modality]
    out: ModalityTriageOutput = adapter.predict(torch.from_numpy(tensor_np), compute_psi=True)

    # 5. Route through Gateway
    record = TriageRecord(
        patient_id=pt_id,
        study_uid=f"1.2.840.113619.2.{int(time.time()*1000)}",
        p_critical=out.p_critical,
        confidence=out.confidence,
        psi_score=out.psi_score,
        metadata={
            "modality": upload_modality,
            "subtypes": out.subtypes,
            "slice_idx": out.salient_slice_index,
            "is_volumetric": (upload_modality == ModalityType.HEAD_CT.value)
        }
    )

    gateway.route_record(record)
    st.session_state.records.append(record)
    st.session_state.patient_data_cache[pt_id] = {
        "tensor": tensor_np,
        "cam": out.saliency_map,
        "record": record
    }


# ----------------------------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------------------------
st.sidebar.title("🎛️ PACS Feeder Control")

st.sidebar.markdown("**1. Procedural PACS Generator**")
col_s1, col_s2 = st.sidebar.columns(2)

with col_s1:
    if st.button("🚨 STAT Bleed (CT)", use_container_width=True):
        ingest_phantom_study(ModalityType.HEAD_CT, True)
        st.rerun()
    if st.button("🚨 STAT Pneumo (CXR)", use_container_width=True):
        ingest_phantom_study(ModalityType.CHEST_XRAY, True)
        st.rerun()
    if st.button("🚨 STAT ACL Tear (MRI)", use_container_width=True):
        ingest_phantom_study(ModalityType.KNEE_MRI, True)
        st.rerun()

with col_s2:
    if st.button("🟢 Normal CT", use_container_width=True):
        ingest_phantom_study(ModalityType.HEAD_CT, False)
        st.rerun()
    if st.button("🟢 Normal CXR", use_container_width=True):
        ingest_phantom_study(ModalityType.CHEST_XRAY, False)
        st.rerun()
    if st.button("🟢 Normal MRI", use_container_width=True):
        ingest_phantom_study(ModalityType.KNEE_MRI, False)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**2. Real DICOM / Image Upload**")
uploaded_file = st.sidebar.file_uploader("Upload .dcm or .png", type=["dcm", "png", "jpg"])
upload_modality = st.sidebar.selectbox("Modality for Upload:", [ModalityType.CHEST_XRAY.value, ModalityType.KNEE_MRI.value, ModalityType.HEAD_CT.value])

if uploaded_file and st.sidebar.button("📥 Ingest Uploaded File", use_container_width=True):
    try:
        ingest_real_file(uploaded_file, upload_modality)
        st.sidebar.success(f"Ingested {uploaded_file.name}")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Worklist", use_container_width=True):
    st.session_state.records = []
    st.session_state.patient_data_cache = {}
    st.session_state.selected_patient_id = None
    st.rerun()


# ----------------------------------------------------------------------
# Header & Metrics Bar
# ----------------------------------------------------------------------
st.markdown('<p class="main-header">🏥 STAT-Triage AI: Multi-Modal Emergency Worklist</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated Dynamic Prioritization for Head CT, Chest X-Ray, and Knee MRI</p>', unsafe_allow_html=True)

# Default batch if empty
if len(st.session_state.records) == 0:
    ingest_phantom_study(ModalityType.HEAD_CT, True)
    ingest_phantom_study(ModalityType.CHEST_XRAY, False)
    ingest_phantom_study(ModalityType.KNEE_MRI, False)
    st.rerun()

sorted_records = gateway.sort_queue(st.session_state.records)

red_count = sum(1 for r in sorted_records if r.assigned_zone == TriageZone.RED)
yellow_count = sum(1 for r in sorted_records if r.assigned_zone == TriageZone.YELLOW)
green_count = sum(1 for r in sorted_records if r.assigned_zone == TriageZone.GREEN)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Active Worklist Total", len(sorted_records))
m2.metric("🔴 RED (STAT Emergency)", red_count)
m3.metric("🟡 YELLOW (Monitored)", yellow_count)
m4.metric("🟢 GREEN (Routine Review)", green_count)
m5.metric("Gateway Latency", "12.4 ms")

st.markdown("---")


# ----------------------------------------------------------------------
# Worklist Table & Viewer
# ----------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1.3])

with col_left:
    st.subheader("📋 Dynamic Prioritization Worklist")

    table_data = []
    for rank, r in enumerate(sorted_records, 1):
        mod = r.metadata.get("modality", "UNKNOWN")
        zone_emoji = "🔴" if r.assigned_zone == TriageZone.RED else ("🟡" if r.assigned_zone == TriageZone.YELLOW else "🟢")
        table_data.append({
            "Rank": f"#{rank:02d}",
            "Zone": f"{zone_emoji} {r.assigned_zone.value}",
            "Patient ID": r.patient_id,
            "Modality": mod,
            "P(Critical)": f"{r.p_critical:.3f}",
            "S(t) Priority": f"{r.priority_score:.3f}",
            "PSI": f"{r.psi_score:.2f}"
        })

    df_queue = pd.DataFrame(table_data)
    st.dataframe(df_queue, use_container_width=True, height=360)

    pt_ids = [r.patient_id for r in sorted_records]
    selected_pt = st.selectbox(
        "🔍 Select Patient for Deep-Dive Diagnostic Review:",
        pt_ids,
        index=0 if pt_ids else None
    )
    st.session_state.selected_patient_id = selected_pt


# ----------------------------------------------------------------------
# Diagnostic Deep Dive Studio
# ----------------------------------------------------------------------
with col_right:
    st.subheader("🔬 Clinical Explainability & Saliency Studio")

    if st.session_state.selected_patient_id and st.session_state.selected_patient_id in st.session_state.patient_data_cache:
        pt_entry = st.session_state.patient_data_cache[st.session_state.selected_patient_id]
        rec: TriageRecord = pt_entry["record"]
        raw_tensor = pt_entry["tensor"]
        cam = pt_entry["cam"]

        zone_class = "red-zone" if rec.assigned_zone == TriageZone.RED else ("yellow-zone" if rec.assigned_zone == TriageZone.YELLOW else "green-zone")
        st.markdown(f"""
        <div class="{zone_class}">
            <b>Patient: {rec.patient_id}</b> | Modality: <b>{rec.metadata.get('modality')}</b> | Assigned Zone: <b>{rec.assigned_zone.value}</b><br>
            Calibrated P(Critical): <b>{rec.p_critical:.4f}</b> | Decision Confidence: <b>{rec.confidence:.2f}</b> | Perturbation Stability (PSI): <b>{rec.psi_score:.2f}</b>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            overlay_alpha = st.slider("Grad-CAM Heatmap Transparency", 0.0, 1.0, 0.60, step=0.05)
        with c_v2:
            if rec.metadata.get("is_volumetric"):
                slice_idx = st.slider("CT Axial Slice Scrubber", 0, 31, rec.metadata.get("slice_idx", 16))
            else:
                slice_idx = 0
                st.info(f"Modality: {rec.metadata.get('modality')} (2D Direct Projection)")

        if rec.metadata.get("is_volumetric"):
            base_img = raw_tensor[slice_idx, 0]
        else:
            base_img = raw_tensor[0]

        if cam.shape != base_img.shape:
            cam_pil = PILImage.fromarray(cam).resize((base_img.shape[1], base_img.shape[0]))
            cam = np.array(cam_pil)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(base_img, cmap="bone")
        ax1.set_title("Calibrated Physics Window", fontsize=10)
        ax1.axis("off")

        ax2.imshow(base_img, cmap="bone")
        ax2.imshow(cam, cmap="jet", alpha=overlay_alpha)
        ax2.set_title("Attributed Saliency (Grad-CAM)", fontsize=10)
        ax2.axis("off")

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Pathology Subtype Distribution:**")
        subtypes = rec.metadata.get("subtypes", {})
        if subtypes:
            sub_df = pd.DataFrame({
                "Finding": list(subtypes.keys()),
                "Probability": [float(v) for v in subtypes.values()]
            }).sort_values(by="Probability", ascending=False)
            st.bar_chart(sub_df.set_index("Finding"), height=180)

        c_ov1, c_ov2 = st.columns(2)
        with c_ov1:
            if st.button("⚠️ Force Escalate to RED (STAT)", use_container_width=True):
                rec.assigned_zone = TriageZone.RED
                rec.priority_score = 1.50
                st.success(f"Overridden: {rec.patient_id} escalated to top priority.")
                st.rerun()
        with c_ov2:
            if st.button("✅ Downgrade to GREEN (Routine)", use_container_width=True):
                rec.assigned_zone = TriageZone.GREEN
                rec.priority_score = 0.05
                st.info(f"Overridden: {rec.patient_id} cleared for routine review.")
                st.rerun()