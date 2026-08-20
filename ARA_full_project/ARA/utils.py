"""
utils.py
--------
Shared configuration, secrets access, and small helpers used across the
ARA backend (database.py, diagnose.py, agents.py).

Kept dependency-light on purpose: the whole ARA stack is Streamlit +
Python, deployed on Streamlit Community Cloud, so this reads secrets via
`st.secrets` when running inside Streamlit and falls back to environment
variables when running standalone (tests, training scripts, notebooks).
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "ara.db"
CARDIAC_DATA_DIR = BASE_DIR / "cardiac_data"

REGRESSOR_PATH = MODELS_DIR / "stenosis_regressor.joblib"
CLASSIFIER_PATH = MODELS_DIR / "triage_classifier.joblib"

# Image resolution the RF models were trained on (matches the reference
# Cardiac-PAI reconstructions: 100x100 flattened to 10,000 features).
IMG_RES = 100

# Triage thresholds (percent stenosis) — single source of truth so
# diagnose.py (labeling) and agents.py (explanations) never disagree.
TRIAGE_THRESHOLDS = {
    "IMMEDIATE SURGERY": 80,
    "URGENT MONITORING": 50,
    "NON-URGENT / STABLE": 0,
}

# Free-tier Gemini model. "gemini-1.5-flash" is the free/low-cost tier as
# of this writing; check https://ai.google.dev/pricing if Google changes
# their lineup and swap the name here only — nothing else needs to change.
GEMINI_MODEL = "gemini-flash-latest"


def get_secret(key: str, default=None):
    """
    Read a config value from Streamlit secrets if available, otherwise
    from the environment. Safe to call outside a Streamlit runtime
    (e.g. from train_model.py or a pytest run).
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


def get_gemini_model():
    """
    Returns an initialized Gemini GenerativeModel.
    Expects GEMINI_API_KEY in .streamlit/secrets.toml (Streamlit Cloud)
    or as an environment variable (local dev). Get a free key at
    https://aistudio.google.com/app/apikey
    """
    import google.generativeai as genai
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Add it to .streamlit/secrets.toml "
            "(local) or the app's Secrets panel (Streamlit Community Cloud). "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def ensure_dirs():
    """Create local working directories if they don't exist yet."""
    for d in (MODELS_DIR, DB_DIR, CARDIAC_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
