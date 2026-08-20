"""
diagnose.py
-----------
The ML / signal-processing layer of ARA.

Two input paths are supported, because the hackathon doc talks about
PPG/ECG screening while the actual trained assets in Cardiac-PAI are
100x100 acoustic "reconstruction" arrays:

  1. RECONSTRUCTION INPUT (.npy, 2D array, ~100x100) — the format your
     existing stenosis_regressor.joblib / triage_classifier.joblib were
     actually trained on. This is what the Campaign Interface's signal
     upload should produce after Phase-1/Phase-2 reconstruction, and
     it's what the demo .npy files (new_patient_*.npy) in your repo are.

  2. RAW PPG / ECG WAVEFORM INPUT (1D array / CSV) — used for supportive
     vitals (heart rate, HRV, waveform quality) that the CareCompanion
     and Explainer agents can reference. It does NOT feed the Random
     Forest directly, since the RF was never trained on 1D waveforms —
     doing that would silently produce meaningless predictions.

IMPORTANT FIX vs. the reference repo:
ai_diagnosis.py's train_synced_model() re-trains on freshly generated
*synthetic* circular-blob images every run, which is why the shipped
stenosis_regressor.joblib / triage_classifier.joblib were degenerate
(regressor ~99% for everything, classifier collapsed to one class).
train_model.py in this backend instead follows main.py's approach:
train on the real reconstructed .npy arrays in cardiac_data/, with
ground truth computed from actual pixel intensities (vessel/plaque
area ratio), not a fabricated percentage.
"""

import glob
import os

import cv2
import joblib
import numpy as np
from scipy.signal import find_peaks

from utils import (
    IMG_RES,
    TRIAGE_THRESHOLDS,
    REGRESSOR_PATH,
    CLASSIFIER_PATH,
    CARDIAC_DATA_DIR,
    ensure_dirs,
)

VESSEL_INTENSITY_THRESHOLD = 0.2
PLAQUE_INTENSITY_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Reconstruction-image path (the real, trained path)
# ---------------------------------------------------------------------------

def preprocess_reconstruction(raw_array: np.ndarray) -> np.ndarray:
    """
    Denoise + normalize a raw reconstruction array to the 100x100,
    [0, 1]-range format the models expect. This is the 'Denoising
    Pipeline' step from the architecture diagram.
    """
    img = np.asarray(raw_array, dtype=np.float32)

    if img.shape != (IMG_RES, IMG_RES):
        img = cv2.resize(img, (IMG_RES, IMG_RES), interpolation=cv2.INTER_LINEAR)

    # Gaussian denoise (matches the smoothing already baked into the
    # reference reconstructions, applied again defensively for any
    # signal that hasn't been through Phase-2 reconstruction yet)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Normalize into [0, 1]
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    img = np.clip(img, 0, 1)
    return img


def compute_ground_truth(img: np.ndarray):
    """
    Derive stenosis % and triage label directly from pixel intensities.
    This is the *real* labeling logic from main.py's prepare_medical_dataset
    — used both for training labels and as an explainable sanity check
    alongside the model's prediction at inference time.
    """
    vessel_area = np.sum(img > VESSEL_INTENSITY_THRESHOLD)
    plaque_area = np.sum(img > PLAQUE_INTENSITY_THRESHOLD)
    percentage = float((plaque_area / vessel_area) * 100) if vessel_area > 0 else 0.0
    triage = triage_from_percentage(percentage)
    return percentage, triage


def triage_from_percentage(percentage: float) -> str:
    if percentage > TRIAGE_THRESHOLDS["IMMEDIATE SURGERY"]:
        return "IMMEDIATE SURGERY"
    elif percentage > TRIAGE_THRESHOLDS["URGENT MONITORING"]:
        return "URGENT MONITORING"
    return "NON-URGENT / STABLE"


def extract_features(img: np.ndarray) -> np.ndarray:
    """Flatten to the 10,000-length feature vector the RF models expect."""
    return img.flatten().reshape(1, -1)


_regressor = None
_classifier = None


def load_models(force_reload=False):
    """
    Loads the trained Random Forest models from disk. Raises a clear
    error (instead of silently using a degenerate model) if they're
    missing — run train_model.py first.
    """
    global _regressor, _classifier
    if force_reload or _regressor is None or _classifier is None:
        if not (os.path.exists(REGRESSOR_PATH) and os.path.exists(CLASSIFIER_PATH)):
            raise FileNotFoundError(
                f"Trained models not found at {REGRESSOR_PATH} / {CLASSIFIER_PATH}. "
                "Run `python train_model.py` first (needs cardiac_data/ populated "
                "with the Reconstructed Data .npy files)."
            )
        _regressor = joblib.load(REGRESSOR_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _regressor, _classifier


def diagnose_reconstruction(raw_array: np.ndarray) -> dict:
    """
    Full diagnostic pipeline for a reconstruction array: preprocess ->
    extract features -> predict stenosis % + triage -> cross-check
    against the explainable pixel-ratio ground truth.
    """
    img = preprocess_reconstruction(raw_array)
    features = extract_features(img)

    regressor, classifier = load_models()
    predicted_pct = float(regressor.predict(features)[0])
    predicted_triage = classifier.predict(features)[0]

    # Explainable cross-check (not a second vote, just surfaced so the
    # Explainer agent / QA can catch model drift)
    rule_pct, rule_triage = compute_ground_truth(img)

    return {
        "stenosis_percentage": round(predicted_pct, 2),
        "triage_result": predicted_triage,
        "rule_based_percentage": round(rule_pct, 2),
        "rule_based_triage": rule_triage,
        "model_rule_agree": predicted_triage == rule_triage,
        "preprocessed_image": img,
    }


# ---------------------------------------------------------------------------
# PPG / ECG waveform path (supportive vitals, not fed into the RF)
# ---------------------------------------------------------------------------

def extract_ppg_ecg_features(waveform: np.ndarray, fs: float = 100.0) -> dict:
    """
    Basic vitals extraction from a raw PPG or ECG waveform: heart rate,
    beat-to-beat interval variability, and a simple signal-quality score.
    These feed the CareCompanion/Explainer agents as context — they are
    NOT inputs to the stenosis Random Forest.
    """
    waveform = np.asarray(waveform, dtype=np.float32).flatten()
    if waveform.size < fs:  # need at least ~1s of signal
        return {"error": "waveform too short for reliable feature extraction"}

    # Denoise
    from scipy.signal import savgol_filter
    window = min(11, waveform.size - (1 - waveform.size % 2))
    if window >= 5:
        smoothed = savgol_filter(waveform, window_length=window, polyorder=3)
    else:
        smoothed = waveform

    # Peak detection -> beat intervals -> heart rate
    min_distance = int(fs * 0.4)  # refractory ~ >150bpm cap
    peaks, _ = find_peaks(smoothed, distance=max(min_distance, 1), prominence=np.std(smoothed) * 0.5)

    if len(peaks) < 2:
        return {
            "heart_rate_bpm": None,
            "hrv_ms": None,
            "num_beats_detected": len(peaks),
            "signal_quality": "poor",
        }

    intervals_samples = np.diff(peaks)
    intervals_ms = (intervals_samples / fs) * 1000.0
    heart_rate_bpm = float(60000.0 / np.mean(intervals_ms))
    hrv_ms = float(np.std(intervals_ms))

    quality = "good" if len(peaks) >= 5 and hrv_ms < 200 else "moderate"

    return {
        "heart_rate_bpm": round(heart_rate_bpm, 1),
        "hrv_ms": round(hrv_ms, 1),
        "num_beats_detected": int(len(peaks)),
        "signal_quality": quality,
    }


# ---------------------------------------------------------------------------
# Unified entry point used by the Diagnostic Agent
# ---------------------------------------------------------------------------

def diagnose(patient_input: dict) -> dict:
    """
    Unified diagnostic entry point.

    patient_input:
        {
            "reconstruction": np.ndarray (2D)   # required for a real stenosis diagnosis
            "ppg_signal": np.ndarray (1D)        # optional, supportive vitals
            "ecg_signal": np.ndarray (1D)        # optional, supportive vitals
            "sampling_rate_hz": float            # optional, default 100.0
        }

    Returns a merged dict combining the stenosis diagnosis with any
    available vitals.
    """
    result = {}

    if "reconstruction" in patient_input and patient_input["reconstruction"] is not None:
        result.update(diagnose_reconstruction(patient_input["reconstruction"]))
    else:
        result["error"] = (
            "No reconstruction array provided — cannot compute a stenosis "
            "diagnosis. PPG/ECG-only vitals will still be returned if given."
        )

    fs = patient_input.get("sampling_rate_hz", 100.0)
    if patient_input.get("ppg_signal") is not None:
        result["ppg_vitals"] = extract_ppg_ecg_features(patient_input["ppg_signal"], fs)
    if patient_input.get("ecg_signal") is not None:
        result["ecg_vitals"] = extract_ppg_ecg_features(patient_input["ecg_signal"], fs)

    return result


def load_npy(path: str) -> np.ndarray:
    """Convenience loader for the demo .npy files (new_patient_*.npy etc.)."""
    return np.load(path, allow_pickle=True)


def list_available_demo_files():
    ensure_dirs()
    return sorted(glob.glob(str(CARDIAC_DATA_DIR / "**" / "*.npy"), recursive=True))
