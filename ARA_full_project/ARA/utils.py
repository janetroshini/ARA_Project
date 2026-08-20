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

CLAUDE_MODEL = "claude-sonnet-4-6"


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


def get_claude_client():
    """
    Returns an initialized anthropic.Anthropic client.
    Expects ANTHROPIC_API_KEY in .streamlit/secrets.toml (Streamlit Cloud)
    or as an environment variable (local dev).
    """
    import anthropic
    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Add it to .streamlit/secrets.toml "
            "(local) or the app's Secrets panel (Streamlit Community Cloud)."
        )
    return anthropic.Anthropic(api_key=api_key)


def ensure_dirs():
    """Create local working directories if they don't exist yet."""
    for d in (MODELS_DIR, DB_DIR, CARDIAC_DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
