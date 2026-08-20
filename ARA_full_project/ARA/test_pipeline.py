"""
test_pipeline.py
-----------------
Standalone smoke test for the ARA backend — runs the full
Intake -> Diagnostic -> Triage -> Explainer -> CareCompanion pipeline
without any Streamlit UI, so you (backend) can verify everything works
before your teammate wires up the frontend pages.

Requires:
  - models/stenosis_regressor.joblib and triage_classifier.joblib
    (run `python train_model.py` first)
  - GEMINI_API_KEY set as an environment variable, e.g.:
        export GEMINI_API_KEY=AIza...
        python test_pipeline.py
    (Explainer/CareCompanion steps are skipped gracefully if it's not set,
    so you can still verify the ML layer + database layer on their own.)

Run:
    python test_pipeline.py
"""

import os
import sys

import numpy as np

import database as db
from diagnose import list_available_demo_files, load_npy, diagnose_reconstruction
from agents import IntakeAgent, DiagnosticAgent, TriageAgent


def make_synthetic_reconstruction(pct=65):
    """Build one synthetic reconstruction so this script runs even with
    no demo .npy files present (e.g. before cardiac_data/ is populated)."""
    import cv2
    res = 100
    img = np.zeros((res, res), dtype=np.float32)
    center, v_rad = res // 2, res // 3
    yy, xx = np.ogrid[:res, :res]
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    img[dist <= v_rad] = 0.3
    p_rad = v_rad * np.sqrt(pct / 100)
    img[dist <= p_rad] = 0.9
    return cv2.GaussianBlur(img, (5, 5), 0)


def main():
    print("=" * 60)
    print("ARA BACKEND SMOKE TEST")
    print("=" * 60)

    print("\n[1/6] Initializing database...")
    db.init_db()
    print("      OK — tables ready at database/ara.db")

    print("\n[2/6] Loading a screening input...")
    demo_files = list_available_demo_files()
    if demo_files:
        raw = load_npy(demo_files[0])
        print(f"      Using demo file: {demo_files[0]}")
    else:
        raw = make_synthetic_reconstruction(pct=65)
        print("      No demo .npy files found in cardiac_data/ — using a "
              "synthetic reconstruction instead.")

    print("\n[3/6] Intake Agent: validating patient...")
    patient_data = {
        "name": "Test Patient",
        "age": 54,
        "gender": "F",
        "contact_number": "9999999999",
        "reconstruction": raw,
        "village": "Demo Village",
        "nurse_name": "N. Kumar",
    }
    intake = IntakeAgent()
    validation = intake.validate(patient_data)
    print(f"      valid={validation['valid']}  errors={validation['errors']}")
    if not validation["valid"]:
        sys.exit(1)
    patient_id = intake.intake(patient_data)
    print(f"      patient_id={patient_id}")

    print("\n[4/6] Diagnostic Agent: running ML pipeline...")
    diagnostic = DiagnosticAgent()
    try:
        diagnosis_result = diagnostic.run({"reconstruction": raw})
    except FileNotFoundError as e:
        print(f"      MODELS NOT TRAINED YET: {e}")
        print("      Run: python train_model.py")
        sys.exit(1)
    print(f"      stenosis_percentage={diagnosis_result['stenosis_percentage']}%")
    print(f"      triage_result={diagnosis_result['triage_result']}")
    print(f"      rule_based check: {diagnosis_result['rule_based_percentage']}% "
          f"/ {diagnosis_result['rule_based_triage']} "
          f"(agrees with model: {diagnosis_result['model_rule_agree']})")

    print("\n[5/6] Triage Agent: classifying...")
    triage = TriageAgent()
    triage_result = triage.classify(diagnosis_result)
    print(f"      final_triage={triage_result['final_triage']}  "
          f"flagged_for_review={triage_result['flagged_for_review']}")

    db.add_screening(
        patient_id=patient_id,
        signal_file=demo_files[0] if demo_files else "synthetic",
        diagnosis="(see explainer step)",
        risk_score=diagnosis_result["stenosis_percentage"],
        triage_result=triage_result["final_triage"],
    )

    print("\n[6/6] Explainer + CareCompanion Agents (Gemini API)...")
    if not os.environ.get("GEMINI_API_KEY"):
        print("      GEMINI_API_KEY not set — skipping Gemini calls. "
              "Set it and re-run to test the Explainer/CareCompanion agents.")
    else:
        from agents import ExplainerAgent, CareCompanionAgent
        explainer = ExplainerAgent()
        report = explainer.explain(
            {
                "stenosis_percentage": diagnosis_result["stenosis_percentage"],
                "triage_result": triage_result["final_triage"],
            },
            patient_data,
        )
        print(f"      Explainer report:\n      {report}\n")

        companion = CareCompanionAgent()
        reply = companion.chat(patient_id, "I'm a bit scared about my results. What should I eat?")
        print(f"      CareCompanion reply:\n      {reply}")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
