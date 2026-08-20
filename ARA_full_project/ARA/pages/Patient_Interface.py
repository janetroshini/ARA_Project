"""
pages/Patient_Interface.py
----------------------------
Patient-facing interface: view a simplified diagnostic report and chat
with the AI Care Companion for emotional support, diet, and exercise
guidance. Wires into database.py + agents.CareCompanionAgent, per the
"What the frontend calls" section of the backend README.
"""

import pandas as pd
import streamlit as st

import database as db
from agents import CareCompanionAgent

st.set_page_config(page_title="ARA — Patient Interface", page_icon="💬", layout="wide")
db.init_db()

st.title("💬 Patient Interface")
st.caption("Your simplified report + your AI Care Companion")

st.sidebar.page_link("app.py", label="← Back to Home", icon="🏠")
st.sidebar.page_link("pages/Campaign_Interface.py", label="Campaign Interface", icon="🩺")

if "patient_id" not in st.session_state:
    st.session_state.patient_id = None

# --- Patient lookup ---------------------------------------------------------
if st.session_state.patient_id is None:
    st.subheader("Find your report")
    st.write("Enter the **Patient ID** given to you by the nurse at your screening camp.")

    lookup_col, browse_col = st.columns([2, 1])
    with lookup_col:
        pid_input = st.text_input("Patient ID", placeholder="e.g. 7")
        if st.button("Open my report", type="primary"):
            if pid_input.strip().isdigit():
                patient = db.get_patient(int(pid_input.strip()))
                if patient:
                    st.session_state.patient_id = int(pid_input.strip())
                    st.rerun()
                else:
                    st.error("No patient found with that ID. Please check with your nurse.")
            else:
                st.error("Patient ID should be a number.")

    with browse_col:
        with st.expander("Don't have your ID?"):
            patients = db.list_patients(limit=50)
            if patients:
                for p in patients:
                    st.write(f"**{p['patient_id']}** — {p['name']} ({p['age']}, {p['gender']})")
            else:
                st.write("No patients registered yet.")
    st.stop()

# --- Loaded patient view -----------------------------------------------------
patient = db.get_patient(st.session_state.patient_id)
if not patient:
    st.error("Patient not found. Please look up your ID again.")
    if st.button("← Look up again"):
        st.session_state.patient_id = None
        st.rerun()
    st.stop()

top_l, top_r = st.columns([4, 1])
with top_l:
    st.subheader(f"Welcome, {patient['name']} 👋")
    st.caption(f"Patient ID: {patient['patient_id']}  |  Age: {patient['age']}  |  Gender: {patient['gender']}")
with top_r:
    if st.button("Switch patient"):
        st.session_state.patient_id = None
        st.rerun()

st.markdown("---")

report_tab, chat_tab = st.tabs(["📄 My Report", "💬 Care Companion Chat"])

# ---------------------------------------------------------------------------
# TAB 1 — Report
# ---------------------------------------------------------------------------
with report_tab:
    latest = db.get_latest_screening(patient["patient_id"])
    if not latest:
        st.info(
            "No screening on file yet. Please visit a health camp to get "
            "screened first — your nurse will register you there."
        )
    else:
        triage = latest["triage_result"]
        triage_color = {
            "IMMEDIATE SURGERY": "🔴",
            "URGENT MONITORING": "🟠",
            "NON-URGENT / STABLE": "🟢",
        }.get(triage, "⚪")

        m1, m2 = st.columns(2)
        m1.metric("Screening result", f"{latest['risk_score']}%")
        m2.metric("Category", f"{triage_color} {triage}")

        st.markdown("#### What this means for you")
        st.info(latest["diagnosis"] or "Your report is being prepared.")

        st.caption(
            f"Screened on {latest['created_at'][:10]}. This is a screening result, "
            "not a final diagnosis — please follow up with a qualified clinician."
        )

        history = db.get_screenings_for_patient(patient["patient_id"])
        if len(history) > 1:
            with st.expander("Past screenings"):
                st.dataframe(
                    pd.DataFrame(history)[["created_at", "risk_score", "triage_result"]],
                    use_container_width=True,
                    hide_index=True,
                )

# ---------------------------------------------------------------------------
# TAB 2 — Care Companion Chat
# ---------------------------------------------------------------------------
with chat_tab:
    st.write(
        "Chat with your **AI Care Companion** about how you're feeling, diet, "
        "exercise, or general lifestyle questions. "
        "*Note: it can't recommend or discuss medications — please ask your "
        "doctor or clinic about those.*"
    )

    quick1, quick2, quick3 = st.columns(3)
    quick_prompt = None
    if quick1.button("🥗 Diet suggestions", use_container_width=True):
        quick_prompt = "Can you give me some general diet suggestions for my heart health?"
    if quick2.button("🏃 Exercise guidance", use_container_width=True):
        quick_prompt = "What kind of exercise is safe and helpful for someone with my results?"
    if quick3.button("😟 I'm feeling anxious", use_container_width=True):
        quick_prompt = "I'm feeling anxious about my screening results. Can you help me?"

    history = db.get_chat_history(patient["patient_id"], limit=50)
    for turn in history:
        if turn["user_message"]:
            with st.chat_message("user"):
                st.write(turn["user_message"])
        if turn["claude_response"]:
            with st.chat_message("assistant"):
                st.write(turn["claude_response"])

    user_message = st.chat_input("Type a message to your Care Companion...")
    message_to_send = quick_prompt or user_message

    if message_to_send:
        with st.chat_message("user"):
            st.write(message_to_send)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    companion = CareCompanionAgent()
                    reply = companion.chat(
                        patient["patient_id"],
                        message_to_send,
                        patient_context=patient,
                    )
                    st.write(reply)
                except RuntimeError as e:
                    st.error(
                        f"Claude API error: {e}\n\n"
                        "Ask the camp organizer to check the ANTHROPIC_API_KEY setup."
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        st.rerun()
