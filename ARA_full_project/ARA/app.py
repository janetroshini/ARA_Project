"""
app.py
------
ARA — Always Reachable Assistant
Home page / landing screen for the Streamlit multi-page app.

Streamlit auto-discovers pages/Campaign_Interface.py and
pages/Patient_Interface.py and lists them in the sidebar navigation.
This file is just the entry point + a friendly landing screen, plus the
one-time database initialization the whole app relies on.
"""

import streamlit as st

import database as db

st.set_page_config(
    page_title="ARA — Always Reachable Assistant",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize the SQLite schema once per session (idempotent — safe on every rerun).
db.init_db()


def check_api_key():
    try:
        return bool(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        import os
        return bool(os.environ.get("GEMINI_API_KEY"))


# --- Header -----------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
        <h1 style="margin-bottom:0;">🫀 ARA</h1>
        <p style="font-size:1.15rem; color:#6b7280; margin-top:0.2rem;">
            Always Reachable Assistant — AI-powered cardiac screening & care companion
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not check_api_key():
    st.warning(
        "⚠️ **GEMINI_API_KEY is not set.** The Explainer and CareCompanion agents "
        "need it to generate patient-friendly reports and chat replies. Add it to "
        "`.streamlit/secrets.toml` locally, or in your Streamlit Community Cloud "
        "app's **Settings → Secrets** panel.",
        icon="⚠️",
    )

st.markdown("---")

# --- Quick stats --------------------------------------------------------
patients = db.list_patients(limit=1000)
campaigns = db.list_campaign_records(limit=1000)
total_screenings = sum(len(db.get_screenings_for_patient(p["patient_id"])) for p in patients) if patients else 0

col1, col2, col3 = st.columns(3)
col1.metric("Patients Registered", len(patients))
col2.metric("Screenings Completed", total_screenings)
col3.metric("Camps / Villages Logged", len(campaigns))

st.markdown("---")

# --- Two entry points -----------------------------------------------------
st.subheader("Choose your interface")

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown("### 🩺 Campaign Interface")
        st.markdown(
            "For **nurses and healthcare workers** running a village health camp.\n\n"
            "- Register a new patient\n"
            "- Upload PPG/ECG or reconstruction signal\n"
            "- Get an instant AI cardiac risk assessment\n"
            "- View triage recommendation"
        )
        st.page_link(
            "pages/Campaign_Interface.py",
            label="Open Campaign Interface →",
            icon="🩺",
            use_container_width=True,
        )

with right:
    with st.container(border=True):
        st.markdown("### 💬 Patient Interface")
        st.markdown(
            "For **patients** who have already been screened.\n\n"
            "- View your simplified diagnostic report\n"
            "- Chat with your AI Care Companion\n"
            "- Get diet & exercise guidance\n"
            "- Receive emotional support after diagnosis"
        )
        st.page_link(
            "pages/Patient_Interface.py",
            label="Open Patient Interface →",
            icon="💬",
            use_container_width=True,
        )

st.markdown("---")

with st.expander("ℹ️ About ARA"):
    st.markdown(
        """
ARA brings together machine learning, generative AI, and portable PPG/ECG
screening in a single Streamlit application, so a healthcare worker can run
a quick cardiac risk assessment on-site and hand the patient a report they
can actually understand.

**Pipeline:** Intake Agent → Diagnostic Agent (Random Forest) → Triage Agent
→ Explainer Agent (Claude) → CareCompanion Agent (Claude)

**Safety boundary:** ARA never suggests, names, or doses medication.
Medication questions are always redirected to the patient's doctor or clinic.

*Team DYNAMO — National AI Hackathon — Problem Statement: MediMentor*
        """
    )

st.sidebar.title("🫀 ARA")
st.sidebar.caption("Always Reachable Assistant")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Navigation**\n\n"
    "Use the pages above (or the links on this screen) to switch between "
    "the Campaign Interface and the Patient Interface."
)
