"""
pages/Campaign_Interface.py
----------------------------
Nurse-facing interface used during village health camps:
register a patient, upload their screening signal, and get an instant
AI cardiac risk assessment + triage recommendation.

Wires directly into the backend's agents.run_full_pipeline(), per the
"What the frontend calls" section of the backend README.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st

import database as db
from agents import run_full_pipeline
from diagnose import list_available_demo_files, load_npy

st.set_page_config(page_title="ARA — Campaign Interface", page_icon="🩺", layout="wide")
db.init_db()

st.title("🩺 Campaign Interface")
st.caption("Nurse / healthcare worker view — village health camp screening")

st.sidebar.page_link("app.py", label="← Back to Home", icon="🏠")
st.sidebar.page_link("pages/Patient_Interface.py", label="Patient Interface", icon="💬")

tab_screen, tab_records = st.tabs(["🆕 New Screening", "📋 Camp Records"])

# ---------------------------------------------------------------------------
# TAB 1 — New Screening
# ---------------------------------------------------------------------------
with tab_screen:
    st.subheader("1. Camp details")
    c1, c2, c3 = st.columns(3)
    with c1:
        campaign_date = st.date_input("Campaign date")
    with c2:
        village = st.text_input("Village / camp location", placeholder="e.g. Perumbakkam")
    with c3:
        nurse_name = st.text_input("Nurse / worker name", placeholder="e.g. N. Kumar")

    st.subheader("2. Patient details")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        name = st.text_input("Patient name*")
    with p2:
        age = st.number_input("Age*", min_value=1, max_value=120, value=45)
    with p3:
        gender = st.selectbox("Gender*", ["Female", "Male", "Other"])
    with p4:
        contact_number = st.text_input("Contact number", placeholder="10-digit mobile")

    st.subheader("3. Screening signal")
    st.caption(
        "Upload a reconstruction array (.npy) for a real stenosis diagnosis. "
        "PPG/ECG waveforms (.csv, one column of values) are optional and only "
        "add supportive vitals (heart rate, HRV) — they don't feed the risk model."
    )

    sig_col1, sig_col2 = st.columns(2)

    with sig_col1:
        recon_file = st.file_uploader(
            "Reconstruction file (.npy) — required for a diagnosis", type=["npy"]
        )
        demo_files = list_available_demo_files()
        demo_choice = None
        if demo_files:
            demo_choice = st.selectbox(
                "...or pick a bundled demo file",
                ["(none)"] + [f.split("/")[-1] for f in demo_files],
            )
        else:
            st.info(
                "No demo files bundled in `cardiac_data/`. Add real Reconstructed "
                "Data `.npy` files there, or upload one above."
            )

    with sig_col2:
        ppg_file = st.file_uploader("PPG waveform (.csv, optional)", type=["csv"])
        ecg_file = st.file_uploader("ECG waveform (.csv, optional)", type=["csv"])
        sampling_rate = st.number_input("Sampling rate (Hz)", min_value=1.0, value=100.0)

    run = st.button("▶ Run AI Screening", type="primary", use_container_width=True)

    if run:
        errors = []
        if not name:
            errors.append("Patient name is required.")
        if not gender:
            errors.append("Gender is required.")

        reconstruction = None
        if recon_file is not None:
            try:
                reconstruction = np.load(io.BytesIO(recon_file.read()), allow_pickle=True)
            except Exception as e:
                errors.append(f"Could not read the uploaded .npy file: {e}")
        elif demo_choice and demo_choice != "(none)":
            match = next((f for f in demo_files if f.endswith(demo_choice)), None)
            if match:
                reconstruction = load_npy(match)

        if reconstruction is None:
            errors.append(
                "No screening signal provided — upload a .npy reconstruction file "
                "or select a demo file."
            )

        ppg_signal = None
        if ppg_file is not None:
            try:
                ppg_signal = pd.read_csv(ppg_file, header=None).values.flatten()
            except Exception as e:
                errors.append(f"Could not read the PPG CSV: {e}")

        ecg_signal = None
        if ecg_file is not None:
            try:
                ecg_signal = pd.read_csv(ecg_file, header=None).values.flatten()
            except Exception as e:
                errors.append(f"Could not read the ECG CSV: {e}")

        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Running Intake → Diagnostic → Triage → Explainer agents..."):
                try:
                    patient_data = {
                        "name": name,
                        "age": int(age),
                        "gender": gender,
                        "contact_number": contact_number,
                        "campaign_date": str(campaign_date),
                        "village": village,
                        "nurse_name": nurse_name,
                        "signal_file": getattr(recon_file, "name", demo_choice or "uploaded"),
                    }
                    patient_input = {
                        "reconstruction": reconstruction,
                        "ppg_signal": ppg_signal,
                        "ecg_signal": ecg_signal,
                        "sampling_rate_hz": sampling_rate,
                    }
                    result = run_full_pipeline(patient_data, patient_input)
                except FileNotFoundError as e:
                    result = None
                    st.error(
                        f"Trained models are missing: {e}\n\n"
                        "Run `python train_model.py` before deploying, or make sure "
                        "`models/*.joblib` were committed to the repo."
                    )
                except RuntimeError as e:
                    result = None
                    st.error(
                        f"Gemini API error: {e}\n\n"
                        "Check that GEMINI_API_KEY is set in Secrets."
                    )
                except Exception as e:
                    result = None
                    st.error(f"Unexpected error while running the pipeline: {e}")

            if result and "error" in result and "stenosis_percentage" not in result:
                st.error(result["error"])
            elif result:
                st.success(f"Screening complete — Patient ID **{result['patient_id']}**")

                triage = result["triage"]["final_triage"]
                triage_color = {
                    "IMMEDIATE SURGERY": "🔴",
                    "URGENT MONITORING": "🟠",
                    "NON-URGENT / STABLE": "🟢",
                }.get(triage, "⚪")

                m1, m2, m3 = st.columns(3)
                m1.metric("Stenosis %", f"{result['stenosis_percentage']}%")
                m2.metric("Triage", f"{triage_color} {triage}")
                m3.metric("Model ↔ Rule Agreement", "✅ Yes" if result.get("model_rule_agreement") else "⚠️ Check")

                if result["triage"].get("flagged_for_review"):
                    st.warning(
                        "⚠️ The model's triage and the rule-based cross-check disagree — "
                        "flagged for manual review before sharing with the patient."
                    )

                st.markdown("#### Plain-language report (shown to patient)")
                st.info(result["plain_language_report"])

                vitals = result.get("vitals", {})
                if vitals.get("ppg") or vitals.get("ecg"):
                    with st.expander("Vitals (from PPG/ECG waveform, supportive only)"):
                        if vitals.get("ppg"):
                            st.write("**PPG:**", vitals["ppg"])
                        if vitals.get("ecg"):
                            st.write("**ECG:**", vitals["ecg"])

                st.success(
                    f"✅ Give the patient their **Patient ID: {result['patient_id']}** — "
                    "they'll need it to open the Patient Interface and chat with their "
                    "AI Care Companion."
                )

# ---------------------------------------------------------------------------
# TAB 2 — Camp Records
# ---------------------------------------------------------------------------
with tab_records:
    st.subheader("Recently registered patients")
    patients = db.list_patients(limit=200)
    if patients:
        st.dataframe(
            pd.DataFrame(patients)[["patient_id", "name", "age", "gender", "contact_number", "created_at"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No patients registered yet.")

    st.subheader("Camp / campaign log")
    camps = db.list_campaign_records(limit=200)
    if camps:
        st.dataframe(pd.DataFrame(camps), use_container_width=True, hide_index=True)
    else:
        st.info("No camp records logged yet.")
