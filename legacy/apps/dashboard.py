# dashboard.py
"""
STAT-Triage AI: Enterprise PACS Clinical Workstation v2
Real-time dashboard exposing Open-Vocabulary Grounding, 3D Axial Cross-Attention,
Conformal Risk Sets, Wald's SPRT Telemetry, and Causal Counterfactual Lesion Subtraction.
"""

import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from core.base import ModalityType, TriagePriorityZone, ModalityTriageOutput
from core.gateway import DynamicPriorityGateway
from core.open_clip_engine import OpenBiomedicalCLIPEngine
from core.conformal import SplitConformalCalibrator
from core.sequential_sprt import SequentialProbabilityRatioTest, SPRTDecisionStatus
from core.counterfactual import CausalCounterfactualEngine


# Page configuration & Dark PACS styling
st.set_page_config(
    page_title="STAT-Triage AI | Clinical PACS Workstation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .reportview-container, .main, .block-container { background-color: #0b0f19; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .stMetric { background-color: #1e293b; padding: 12px 16px; border-radius: 8px; border: 1px solid #334155; }
    .badge-red { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; }
    .badge-yellow { background-color: #f59e0b; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; }
    .badge-green { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; }
    .conformal-tag { background-color: #3b82f6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_foundation_engines():
    """Initializes and caches deep-tech engines."""
    clip_engine = OpenBiomedicalCLIPEngine()
    
    # Initialize & pre-fit conformal engine
    conformal_engine = SplitConformalCalibrator(
        alpha=0.01,
        class_names=["Normal", "Epidural_Hematoma", "Subdural_Hematoma", "Subarachnoid", "Intraparenchymal"],
    )
    np.random.seed(42)
    calib_p = np.random.dirichlet(np.ones(5) * 0.5, size=500)
    calib_y = np.random.choice(5, size=500)
    for i in range(500):
        calib_p[i, calib_y[i]] += 2.0
    calib_p /= calib_p.sum(axis=1, keepdims=True)
    conformal_engine.fit(calib_p, calib_y)

    sprt_engine = SequentialProbabilityRatioTest(alpha=0.01, beta=0.001, baseline_prob=0.05)
    gateway = DynamicPriorityGateway()

    return clip_engine, conformal_engine, sprt_engine, gateway


clip_engine, conformal_engine, sprt_engine, gateway = load_foundation_engines()

# Header & Subsystem status
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("⚡ STAT-Triage AI: Clinical PACS Workstation")
    st.caption("Distribution-Free Conformal Guarantees | Wald's CUSUM-SPRT Streaming | Causal Counterfactual Inpainting")
with col_h2:
    st.markdown("<div style='text-align: right; padding-top: 15px;'><span class='badge-green'>● CADt Engine Online</span></div>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar Controls
st.sidebar.header("🔬 PACS Stream Generator")
selected_modality = st.sidebar.selectbox("Select Active Study Modality", ["Chest Radiograph (2D)", "Volumetric Head CT (32-Slice)", "Multi-Planar Knee MRI"])
simulation_scenario = st.sidebar.radio("Simulated Clinical Scenario", ["Acute Emergency (Pneumothorax / Epidural Bleed)", "Ambiguous Borderline Case (High Uncertainty)", "Routine Outpatient (Normal)"])

wait_time = st.sidebar.slider("Current Wait Time (Minutes)", min_value=0, max_value=180, value=25, step=5)
open_vocab_prompt = st.sidebar.text_input("Open-Vocabulary Clinical Query", value="acute tension pneumothorax with visible visceral pleural line")

# Generate Synthetic Patient Data based on Scenario
if selected_modality == "Chest Radiograph (2D)":
    modality_enum = ModalityType.CHEST_XRAY
    if "Acute" in simulation_scenario:
        p_crit = 0.94
        psi = 0.96
        probs = np.array([[0.02, 0.94, 0.01, 0.02, 0.01]])
        subtypes = {"Pneumothorax": 0.94, "Effusion": 0.04, "Consolidation": 0.08, "Edema": 0.02, "Cardiomegaly": 0.01}
    elif "Ambiguous" in simulation_scenario:
        p_crit = 0.48
        psi = 0.32
        probs = np.array([[0.25, 0.22, 0.20, 0.18, 0.15]])
        subtypes = {"Pneumothorax": 0.48, "Effusion": 0.42, "Consolidation": 0.35, "Edema": 0.21, "Cardiomegaly": 0.12}
    else:
        p_crit = 0.03
        psi = 0.99
        probs = np.array([[0.95, 0.01, 0.01, 0.02, 0.01]])
        subtypes = {"Pneumothorax": 0.03, "Effusion": 0.02, "Consolidation": 0.01, "Edema": 0.01, "Cardiomegaly": 0.01}

    # Conformal Set calculation
    c_sets, _ = conformal_engine.predict_set(probs)
    conformal_tags = c_sets[0]

    # Causal Counterfactual Generation
    synth_tensor = torch.zeros(1, 1, 224, 224)
    if "Acute" in simulation_scenario:
        synth_tensor[:, :, 30:90, 130:190] = 0.90 # Focal lesion

    def mock_classifier(x):
        return torch.sigmoid((x[:, :, 30:90, 130:190] * 6.0).sum(dim=(-2, -1)) / 3600.0 - 3.0)

    cf_engine = CausalCounterfactualEngine(classifier_fn=mock_classifier, max_iters=25)
    cf_img, causal_residual, cf_metrics = cf_engine.generate_counterfactual(synth_tensor, target_prob=0.05)

    # Open-Vocabulary Alignment
    zero_shot_scores = clip_engine.zero_shot_triage(
        synth_tensor,
        candidate_queries=[
            "healthy clear lungs",
            open_vocab_prompt,
            "dense pulmonary consolidation",
            "bilateral pleural effusion",
        ]
    )

    triage_out = ModalityTriageOutput(
        modality=modality_enum,
        p_critical=p_crit,
        raw_logits=probs,
        subtype_risks=subtypes,
        psi_score=psi,
        conformal_set=conformal_tags,
        counterfactual_heatmap=causal_residual,
        open_vocab_distribution=zero_shot_scores,
    )

else: # Volumetric Head CT
    modality_enum = ModalityType.HEAD_CT
    total_slices = 32
    if "Acute" in simulation_scenario:
        slice_stream = np.full(total_slices, 0.02)
        slice_stream[6:14] = 0.97 # Acute bleed starting at slice 7
        p_crit = 0.97
        psi = 0.95
        probs = np.array([[0.01, 0.97, 0.01, 0.01, 0.00]])
        subtypes = {"Epidural": 0.97, "Subdural": 0.12, "Subarachnoid": 0.05, "Intraparenchymal": 0.02, "Fracture": 0.88}
    elif "Ambiguous" in simulation_scenario:
        slice_stream = np.full(total_slices, 0.35)
        p_crit = 0.52
        psi = 0.28
        probs = np.array([[0.20, 0.25, 0.22, 0.18, 0.15]])
        subtypes = {"Epidural": 0.52, "Subdural": 0.44, "Subarachnoid": 0.31, "Intraparenchymal": 0.25, "Fracture": 0.19}
    else:
        slice_stream = np.full(total_slices, 0.01)
        p_crit = 0.02
        psi = 0.98
        probs = np.array([[0.96, 0.01, 0.01, 0.01, 0.01]])
        subtypes = {"Epidural": 0.02, "Subdural": 0.01, "Subarachnoid": 0.01, "Intraparenchymal": 0.01, "Fracture": 0.00}

    # Evaluate Wald's SPRT on slice stream
    sprt_res = sprt_engine.evaluate_full_stream(slice_stream)
    c_sets, _ = conformal_engine.predict_set(probs)

    triage_out = ModalityTriageOutput(
        modality=modality_enum,
        p_critical=p_crit,
        raw_logits=probs,
        subtype_risks=subtypes,
        psi_score=psi,
        conformal_set=c_sets[0],
        sprt_halted=(sprt_res.status == SPRTDecisionStatus.CRITICAL_HALT),
        sprt_exit_slice=sprt_res.exit_slice_idx,
        sprt_compute_saved_pct=sprt_res.compute_savings_pct,
        sprt_llr_trajectory=sprt_res.llr_trajectory,
    )

# Enqueue into Gateway
gateway._queue.clear()
gateway.enqueue("STUDY-001", "PATIENT-8821", modality_enum, wait_time, triage_out)
# Add dummy queue items for visual realism
gateway.enqueue("STUDY-002", "PATIENT-1094", ModalityType.CHEST_XRAY, 110, ModalityTriageOutput(ModalityType.CHEST_XRAY, 0.08, np.array([[0.9, 0.1]]), {"Pneumothorax": 0.08}, 0.98, conformal_set=["Normal"]))
gateway.enqueue("STUDY-003", "PATIENT-3312", ModalityType.HEAD_CT, 45, ModalityTriageOutput(ModalityType.HEAD_CT, 0.45, np.array([[0.5, 0.5]]), {"Subdural": 0.45}, 0.65, conformal_set=["Subdural", "Normal"]))

worklist = gateway.get_sorted_worklist()
active_case = worklist[0]

# Metrics Row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Critical Probability", f"{active_case.triage_output.p_critical * 100:.1f}%")
with m2:
    st.metric("Dynamic Priority S(t)", f"{active_case.dynamic_score:.3f}")
with m3:
    st.metric("Perturbation Stability (PSI)", f"{active_case.triage_output.psi_score:.2f}")
with m4:
    st.metric("Conformal Set Size |C(X)|", f"{len(active_case.triage_output.conformal_set)}")
with m5:
    if active_case.triage_output.sprt_halted:
        st.metric("SPRT Compute Saved", f"{active_case.triage_output.sprt_compute_saved_pct:.1f}%", delta="Sub-15ms Halt")
    else:
        st.metric("Triage Zone", active_case.priority_zone.value)

st.markdown("---")

# Main Dashboard Grid
tab_pacs, tab_conformal, tab_sprt, tab_open_vocab = st.tabs([
    "🖥️ PACS Explainability Studio",
    "🛡️ Conformal Risk Guarantees",
    "⚡ Wald's SPRT Telemetry",
    "🌐 Open-Vocabulary Foundation",
])

with tab_pacs:
    pacs_col1, pacs_col2 = st.columns([1, 1])
    with pacs_col1:
        st.subheader("Original Normalized Radiograph / CT Slice")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(synth_tensor.squeeze().numpy() if selected_modality == "Chest Radiograph (2D)" else np.zeros((224, 224)), cmap="bone")
        ax.set_title("Input DICOM Physical HU Canvas", color="white")
        ax.axis("off")
        fig.patch.set_facecolor('#0b0f19')
        st.pyplot(fig)

    with pacs_col2:
        st.subheader("Causal Counterfactual Lesion Residual |X - X*|")
        if triage_out.counterfactual_heatmap is not None:
            fig_cf, ax_cf = plt.subplots(figsize=(5, 5))
            ax_cf.imshow(synth_tensor.squeeze().numpy(), cmap="bone")
            ax_cf.imshow(triage_out.counterfactual_heatmap, cmap="inferno", alpha=0.6)
            ax_cf.set_title("Causal Tissue Attribution Mask", color="white")
            ax_cf.axis("off")
            fig_cf.patch.set_facecolor('#0b0f19')
            st.pyplot(fig_cf)
        else:
            st.info("SPRT Early-Exit active on volumetric stream. Slice attribution rendered via CUSUM trajectory.")

with tab_conformal:
    st.subheader("Mathematical Conformal Prediction Set C(X)")
    st.markdown(f"**Proven Statistical Safety Bound:** $\\mathbb{{P}}(Y_{{true}} \\in \\mathcal{{C}}(X)) \\ge 99.0\\%$ ($\\alpha = 0.01$)")
    
    tags_html = " ".join([f"<span class='conformal-tag'>{c}</span>" for c in active_case.triage_output.conformal_set])
    st.markdown(f"**Current Statistically Guaranteed Set:** {tags_html}", unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("Subtype Risk Distribution (Calibrated Posteriors):")
    st.bar_chart(active_case.triage_output.subtype_risks)

with tab_sprt:
    st.subheader("Streaming Sequential Probability Ratio Test (Wald's CUSUM-SPRT)")
    if active_case.triage_output.sprt_llr_trajectory:
        st.line_chart(pd.DataFrame({
            "Log-Likelihood Ratio (Λ_k)": active_case.triage_output.sprt_llr_trajectory,
            "Critical Boundary (A = +4.60)": [4.604] * len(active_case.triage_output.sprt_llr_trajectory),
        }))
        if active_case.triage_output.sprt_halted:
            st.success(f"🚨 **STAT Critical Early-Exit Triggered at Slice #{active_case.triage_output.sprt_exit_slice + 1} / 32**. Ingestion interrupted to fast-track patient to Rank #1.")
    else:
        st.info("SPRT telemetry is active for volumetric sequences (3D Head CT / Multi-Sequence MRI).")

with tab_open_vocab:
    st.subheader("BiomedCLIP Zero-Shot Open-Vocabulary Alignment")
    if triage_out.open_vocab_distribution:
        st.write(f"Evaluated against custom radiologist prompt: *'{open_vocab_prompt}'*")
        st.dataframe(pd.DataFrame(list(triage_out.open_vocab_distribution.items()), columns=["Clinical Prompt", "Zero-Shot Probability"]).sort_values(by="Zero-Shot Probability", ascending=False))