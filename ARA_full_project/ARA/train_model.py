"""
train_model.py
---------------
Trains stenosis_regressor.joblib and triage_classifier.joblib on the
REAL reconstructed .npy arrays (Phase-2 k-Wave reconstructions), using
the ground-truth labeling logic from the reference repo's main.py —
NOT ai_diagnosis.py's synthetic generator, which is what produced the
degenerate models (regressor ~99% for everything, classifier collapsed
to a single class) flagged in earlier testing.

Usage:
    1. Put your Reconstructed Data (the .npy files from Cardiac-PAI's
       "Reconstructed Data.zip" / cardiac_data/ folder) into:
           ARA_backend/cardiac_data/
       (subfolders are fine — this script searches recursively)
    2. Run:
           python train_model.py
    3. Models are written to ARA_backend/models/

If cardiac_data/ is empty, this script falls back to a clearly-labeled
SYNTHETIC dataset so the pipeline is still runnable end-to-end for a
demo — but it prints a loud warning, because a model trained only on
synthetic data should never be presented as clinically meaningful.
"""

import glob
import sys

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from utils import IMG_RES, MODELS_DIR, CARDIAC_DATA_DIR, ensure_dirs
from diagnose import (
    preprocess_reconstruction,
    compute_ground_truth,
    VESSEL_INTENSITY_THRESHOLD,
    PLAQUE_INTENSITY_THRESHOLD,
)


def load_real_dataset():
    """Load every .npy under cardiac_data/, compute real pixel-ratio labels."""
    files = glob.glob(str(CARDIAC_DATA_DIR / "**" / "*.npy"), recursive=True)
    files.sort()
    if not files:
        return None, None, None, None

    X, y_pct, y_triage = [], [], []
    for f in files:
        raw = np.load(f, allow_pickle=True)
        img = preprocess_reconstruction(raw)
        pct, triage = compute_ground_truth(img)
        X.append(img.flatten())
        y_pct.append(pct)
        y_triage.append(triage)

    return np.array(X), np.array(y_pct), np.array(y_triage), files


def synthetic_fallback_dataset(n_samples=500, seed=42):
    """
    Clearly-labeled synthetic fallback ONLY for pipeline testing when no
    real data is present. Anatomically similar to the reference repo's
    generator but exists here purely so the rest of the codebase (agents,
    database, Streamlit pages) can be exercised end-to-end without real
    clinical data on hand.
    """
    rng = np.random.default_rng(seed)
    res = IMG_RES
    X, y_pct, y_triage = [], [], []

    for _ in range(n_samples):
        pct = rng.uniform(5, 95)
        img = np.zeros((res, res), dtype=np.float32)
        center, v_rad = res // 2, res // 3
        yy, xx = np.ogrid[:res, :res]
        dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)

        img[dist <= v_rad] = 0.3
        p_rad = v_rad * np.sqrt(pct / 100)
        img[dist <= p_rad] = 0.9
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = np.clip(img, 0, 1)

        # Label from the SAME pixel-ratio ground truth used for real data,
        # not the fabricated `pct` used to draw the circle — keeps the
        # labeling function consistent between real and synthetic data.
        real_pct, triage = compute_ground_truth(img)

        X.append(img.flatten())
        y_pct.append(real_pct)
        y_triage.append(triage)

    return np.array(X), np.array(y_pct), np.array(y_triage)


def train_and_save():
    ensure_dirs()

    X, y_pct, y_triage, files = load_real_dataset()
    used_synthetic = False

    if X is None or len(X) < 20:
        print(
            "\n[WARNING] No (or too little) real data found in "
            f"{CARDIAC_DATA_DIR}. Falling back to a SYNTHETIC dataset "
            "for pipeline testing only. Do NOT present this model as "
            "clinically validated — drop your real Reconstructed Data "
            ".npy files into cardiac_data/ and re-run this script.\n"
        )
        X, y_pct, y_triage = synthetic_fallback_dataset()
        used_synthetic = True
    else:
        print(f"Loaded {len(files)} real reconstruction files from {CARDIAC_DATA_DIR}")

    print(f"Class distribution: {dict(zip(*np.unique(y_triage, return_counts=True)))}")
    if len(set(y_triage)) < 2:
        print(
            "[WARNING] Only one triage class present in the training data. "
            "The classifier will not be able to distinguish risk levels. "
            "Add more varied reconstructions (a range of stenosis severities)."
        )

    X_train, X_test, p_train, p_test, c_train, c_test = train_test_split(
        X, y_pct, y_triage, test_size=0.20, random_state=42
    )

    print("Training regressor...")
    regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    regressor.fit(X_train, p_train)

    print("Training classifier...")
    classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    classifier.fit(X_train, c_train)

    # Evaluate
    p_pred = regressor.predict(X_test)
    c_pred = classifier.predict(X_test)
    mae = mean_absolute_error(p_test, p_pred)
    acc = accuracy_score(c_test, c_pred)

    print(f"\nRegressor MAE on held-out test set: {mae:.2f} percentage points")
    print(f"Classifier accuracy on held-out test set: {acc:.2%}")
    print(f"Predicted stenosis %% range on test set: {p_pred.min():.1f} - {p_pred.max():.1f}")
    print(f"Predicted triage classes on test set: {sorted(set(c_pred))}")

    if p_pred.max() - p_pred.min() < 5:
        print(
            "[WARNING] Predictions barely vary across the test set — this "
            "is the same failure mode as the original degenerate models. "
            "Check that cardiac_data/ actually contains varied stenosis "
            "severities, not near-duplicate reconstructions."
        )

    reg_path = MODELS_DIR / "stenosis_regressor.joblib"
    clf_path = MODELS_DIR / "triage_classifier.joblib"
    joblib.dump(regressor, reg_path)
    joblib.dump(classifier, clf_path)

    print(f"\nSaved: {reg_path}")
    print(f"Saved: {clf_path}")
    if used_synthetic:
        print("\n>>> Trained on SYNTHETIC data — replace with real data before demo/submission. <<<")


if __name__ == "__main__":
    train_and_save()
