"""CT-only research worklist for the supported RSNA pilot model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stat_triage.ingestion import (
    DICOMIngestionError,
    MAX_DICOM_UPLOAD_BYTES,
    ingest_dicom_bytes,
)
from stat_triage.rsna import DEFAULT_BUNDLE_DIR, RSNAInferenceAdapter
from stat_triage.worklist import PriorityZone, WorklistService


st.set_page_config(
    page_title="STAT-Triage RSNA Worklist",
    page_icon=":material/radiology:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading verified RSNA checkpoint…")
def load_adapter() -> RSNAInferenceAdapter:
    return RSNAInferenceAdapter(DEFAULT_BUNDLE_DIR)


def get_worklist() -> WorklistService:
    if "worklist" not in st.session_state:
        st.session_state.worklist = WorklistService()
    return st.session_state.worklist


def zone_badge(zone: PriorityZone) -> str:
    return zone.value


def rows_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Zone": zone_badge(row.effective_zone),
                "Case": row.case_id,
                "Patient": row.patient_id,
                "Any score": round(row.any_score, 4),
                "Threshold": round(row.threshold, 4),
                "Wait (min)": round(row.wait_minutes, 1),
                "Priority": round(row.priority_score, 4),
                "Override": row.override_reason or "",
            }
            for row in rows
        ]
    )


def render_header(adapter: RSNAInferenceAdapter, active_count: int) -> None:
    st.title("STAT-Triage · RSNA head CT worklist")
    st.caption(
        "Research demonstration · slice-level intracranial hemorrhage screening · "
        "not for clinical use"
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active cases", active_count)
    col2.metric("Locked threshold", f"{adapter.any_threshold:.4f}")
    col3.metric("Selected epoch", adapter.model_epoch)
    col4.metric("Model hash", f"{adapter.model_sha256[:10]}…")


def render_ingestion(adapter: RSNAInferenceAdapter, worklist: WorklistService) -> None:
    st.subheader("Add a CT slice")
    st.write(
        "Upload one CT DICOM instance. Processing stays in this local application "
        "session; the score is not a population-calibrated probability."
    )
    with st.form("dicom_ingestion", clear_on_submit=True):
        uploaded = st.file_uploader(
            "CT DICOM file",
            type=["dcm", "dicom"],
            max_upload_size=MAX_DICOM_UPLOAD_BYTES // (1024 * 1024),
            help="Single-frame CT DICOM, maximum 32 MiB.",
        )
        submitted = st.form_submit_button("Run inference and enqueue", type="primary")

    if submitted:
        if uploaded is None:
            st.warning("Choose a DICOM file before running inference.")
            return
        try:
            with st.spinner("Running verified preprocessing and inference…"):
                row = ingest_dicom_bytes(
                    uploaded.getvalue(), uploaded.name, adapter, worklist
                )
        except DICOMIngestionError as exc:
            st.error(f"The DICOM could not be processed: {exc}")
            return
        except ValueError as exc:
            st.error(str(exc))
            return
        st.success(
            f"Case {row.case_id} added as {row.effective_zone.value}. "
            f"Any score {row.any_score:.4f} at threshold {row.threshold:.4f}."
        )
        st.rerun()


def render_case(row, worklist: WorklistService) -> None:
    evidence = worklist.evidence_for(row.case_id)
    st.subheader(f"Case details · {row.case_id}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Effective zone", row.effective_zone.value)
    col2.metric("Any score", f"{row.any_score:.4f}")
    col3.metric("Wait", f"{row.wait_minutes:.1f} min")
    col4.metric("Threshold result", "Positive" if row.predicted_any else "Negative")

    st.caption(
        f"Patient {row.patient_id} · Study {row.study_instance_uid} · "
        f"Model {row.model_sha256[:16]}…"
    )
    scores = pd.DataFrame(
        {"Finding": list(evidence.scores), "Score": list(evidence.scores.values())}
    ).set_index("Finding")
    st.bar_chart(scores, horizontal=True)

    action_left, action_right = st.columns(2)
    with action_left:
        st.markdown("**Clinician override**")
        with st.form(f"override-{row.case_id}"):
            actor = st.text_input("Actor ID", key=f"override-actor-{row.case_id}")
            selected_zone = st.selectbox(
                "Target zone", [zone.value for zone in PriorityZone], key=f"zone-{row.case_id}"
            )
            reason = st.text_area("Reason", key=f"reason-{row.case_id}")
            override_submitted = st.form_submit_button("Apply override")
        if override_submitted:
            try:
                worklist.override(
                    row.case_id,
                    PriorityZone(selected_zone),
                    reason,
                    actor=actor,
                )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.rerun()

        if row.overridden:
            with st.form(f"clear-{row.case_id}"):
                clear_actor = st.text_input("Actor ID", key=f"clear-actor-{row.case_id}")
                clear_reason = st.text_input(
                    "Reason for clearing override", key=f"clear-reason-{row.case_id}"
                )
                clear_submitted = st.form_submit_button("Clear override")
            if clear_submitted:
                try:
                    worklist.clear_override(
                        row.case_id, actor=clear_actor, reason=clear_reason
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.rerun()

    with action_right:
        st.markdown("**Complete review**")
        with st.form(f"complete-{row.case_id}"):
            completed_by = st.text_input("Actor ID", key=f"complete-actor-{row.case_id}")
            complete_submitted = st.form_submit_button("Mark completed")
        if complete_submitted:
            try:
                worklist.complete(row.case_id, actor=completed_by)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.rerun()

    with st.expander("Audit history"):
        events = worklist.audit_history(row.case_id)
        audit = pd.DataFrame(
            [
                {
                    "Timestamp": pd.to_datetime(event.timestamp, unit="s", utc=True),
                    "Action": event.action,
                    "Actor": event.actor,
                    "Details": dict(event.details),
                }
                for event in events
            ]
        )
        st.dataframe(audit, width="stretch", hide_index=True)


def main() -> None:
    try:
        adapter = load_adapter()
    except Exception as exc:
        st.error(f"Verified model bundle is unavailable: {exc}")
        st.stop()

    worklist = get_worklist()
    rows = worklist.active_worklist()
    render_header(adapter, len(rows))

    st.info(
        "The locked threshold was selected on an enriched pilot calibration set. "
        "Scores and worklist priority are research outputs, not clinical probabilities."
    )

    ingest_tab, worklist_tab, provenance_tab = st.tabs(
        ["Add DICOM", "Active worklist", "Model provenance"]
    )
    with ingest_tab:
        render_ingestion(adapter, worklist)

    with worklist_tab:
        st.subheader("Active worklist")
        if not rows:
            st.info("No active cases. Add a CT DICOM instance to begin.")
        else:
            st.dataframe(rows_frame(rows), width="stretch", hide_index=True)
            selected = st.selectbox("Inspect case", [row.case_id for row in rows])
            selected_row = next(row for row in rows if row.case_id == selected)
            render_case(selected_row, worklist)

    with provenance_tab:
        st.subheader("Locked model contract")
        st.json(
            {
                "endpoint": adapter.operating_point["endpoint"],
                "architecture": adapter.run_config["model"],
                "labels": adapter.run_config["labels"],
                "input_size": adapter.run_config["input_size"],
                "windows": adapter.run_config["windows"],
                "normalization": adapter.run_config["normalization"],
                "selected_epoch": adapter.model_epoch,
                "threshold": adapter.any_threshold,
                "checkpoint_sha256": adapter.model_sha256,
            }
        )
        st.warning(
            "This interface runs the slice-level pilot only. A separate locked retrospective "
            "study-level evaluation on 240 held-out patients achieved ROC-AUC 0.9078; the "
            "interface does not implement that study aggregation. Prospective performance "
            "remains unknown."
        )


main()
