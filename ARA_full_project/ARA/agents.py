"""
agents.py
---------
The agentic AI layer of ARA, matching the pipeline in the project doc:

    Intake Agent -> Diagnostic Agent -> Triage Agent -> Explainer Agent
                                                              |
                                                      CareCompanion Agent

Only the Explainer and CareCompanion agents call an LLM (Google Gemini,
free tier) — Intake, Diagnostic, and Triage are deterministic/rule-based
on purpose (patient data validation and the ML prediction shouldn't
depend on an LLM call that could hallucinate a number). This keeps the
system fast, cheap, and auditable: the LLM is used for what it's
actually good at (turning a structured result into a clear, kind
explanation), not for arithmetic.

Safety boundary (kept intentional, per the doc): CareCompanion NEVER
gives medication names, dosages, or drug interaction advice. Every
agent prompt below enforces that explicitly and instructs the model to
redirect medication questions to the patient's care team.
"""

import datetime

from diagnose import diagnose, triage_from_percentage
from utils import get_gemini_model
import database as db


MEDICATION_BOUNDARY = (
    "You must never suggest, name, dose, or compare any medication or "
    "drug. If the person asks about medication, tell them clearly that "
    "medication decisions must come from their doctor or the clinic that "
    "screened them, and that you can help with everything else "
    "(understanding the report, diet, exercise, and emotional support)."
)


# ---------------------------------------------------------------------------
# 1. Intake Agent
# ---------------------------------------------------------------------------

class IntakeAgent:
    """Validates patient data and uploaded signals before anything downstream runs."""

    REQUIRED_FIELDS = ["name", "age", "gender"]

    def validate(self, patient_data: dict) -> dict:
        errors = []

        for field in self.REQUIRED_FIELDS:
            if not patient_data.get(field):
                errors.append(f"Missing required field: {field}")

        age = patient_data.get("age")
        if age is not None:
            try:
                age_val = int(age)
                if age_val <= 0 or age_val > 120:
                    errors.append(f"Age {age_val} is out of plausible range (1-120).")
            except (ValueError, TypeError):
                errors.append(f"Age '{age}' is not a valid number.")

        contact = patient_data.get("contact_number")
        if contact and not str(contact).replace("+", "").replace(" ", "").isdigit():
            errors.append(f"Contact number '{contact}' doesn't look like a valid number.")

        signal_present = any(
            patient_data.get(k) is not None
            for k in ("reconstruction", "ppg_signal", "ecg_signal")
        )
        if not signal_present:
            errors.append("No screening signal provided (reconstruction, PPG, or ECG).")

        return {"valid": len(errors) == 0, "errors": errors}

    def intake(self, patient_data: dict) -> int:
        """Validate, then persist the patient and return their patient_id."""
        result = self.validate(patient_data)
        if not result["valid"]:
            raise ValueError(f"Intake validation failed: {result['errors']}")

        return db.add_patient(
            name=patient_data["name"],
            age=patient_data.get("age"),
            gender=patient_data.get("gender"),
            contact_number=patient_data.get("contact_number"),
        )


# ---------------------------------------------------------------------------
# 2. Diagnostic Agent
# ---------------------------------------------------------------------------

class DiagnosticAgent:
    """Wraps the ML pipeline in diagnose.py — no LLM involved here on purpose."""

    def run(self, patient_input: dict) -> dict:
        return diagnose(patient_input)


# ---------------------------------------------------------------------------
# 3. Triage Agent
# ---------------------------------------------------------------------------

class TriageAgent:
    """
    Confirms / normalizes the triage category. The RF classifier already
    predicts a category, but this agent is the single place that maps a
    percentage to a category (via diagnose.triage_from_percentage) — so
    if the model and the rule ever disagree, downstream code can flag it
    instead of silently trusting either one.
    """

    def classify(self, diagnosis_result: dict) -> dict:
        model_triage = diagnosis_result.get("triage_result")
        pct = diagnosis_result.get("stenosis_percentage")
        rule_triage = triage_from_percentage(pct) if pct is not None else None

        flagged = model_triage is not None and rule_triage is not None and model_triage != rule_triage

        return {
            "final_triage": model_triage or rule_triage,
            "model_triage": model_triage,
            "rule_triage": rule_triage,
            "flagged_for_review": flagged,
        }


# ---------------------------------------------------------------------------
# 4. Explainer Agent (Gemini)
# ---------------------------------------------------------------------------

class ExplainerAgent:
    """Turns the structured diagnosis into a plain-language patient report."""

    SYSTEM_PROMPT = (
        "You are the Explainer Agent inside ARA, a rural cardiac screening "
        "assistant. You are given a structured AI screening result (a "
        "stenosis percentage and a triage category produced by a trained "
        "model — you did not calculate these numbers, so never change them "
        "or invent new ones). Your job is to explain what the result means "
        "in warm, simple, non-alarming language a patient with no medical "
        "background can understand. Always: (1) state clearly this is a "
        "screening result, not a final diagnosis, (2) tell them what the "
        "triage category means practically (e.g. whether and how soon to "
        "see a doctor), (3) end with encouragement to follow up with a "
        "qualified clinician. " + MEDICATION_BOUNDARY
    )

    def explain(self, diagnosis_result: dict, patient_context: dict) -> str:
        model = get_gemini_model()

        user_prompt = (
            f"Patient: {patient_context.get('name', 'the patient')}, "
            f"age {patient_context.get('age', 'unknown')}.\n"
            f"AI screening result:\n"
            f"- Predicted stenosis: {diagnosis_result.get('stenosis_percentage')}%\n"
            f"- Triage category: {diagnosis_result.get('triage_result')}\n"
            f"Write a short (under 200 words) plain-language explanation "
            f"for the patient."
        )

        response = model.generate_content(
            [self.SYSTEM_PROMPT, user_prompt],
            generation_config={"max_output_tokens": 500},
        )
        return response.text


# ---------------------------------------------------------------------------
# 5. CareCompanion Agent (Gemini)
# ---------------------------------------------------------------------------

class CareCompanionAgent:
    """Ongoing conversational support: emotional check-ins, diet, exercise."""

    SYSTEM_PROMPT = (
        "You are the CareCompanion Agent inside ARA, a supportive AI "
        "companion for patients after a cardiac screening in a rural "
        "health camp. You provide emotional support, anxiety check-ins, "
        "general diet suggestions, and general exercise guidance suited "
        "to a cardiovascular risk patient. Keep responses warm, brief, and "
        "encouraging. You are not a substitute for a doctor: for anything "
        "that sounds urgent (chest pain, breathlessness, fainting), tell "
        "the patient to seek in-person medical care immediately. "
        + MEDICATION_BOUNDARY
    )

    def chat(self, patient_id: int, user_message: str, patient_context: dict = None) -> str:
        patient_context = patient_context or {}

        context_note = ""
        latest = db.get_latest_screening(patient_id)
        if latest:
            context_note = (
                f"\n\n[Context for you, not to repeat verbatim: this patient's "
                f"latest screening showed {latest['risk_score']}% stenosis, "
                f"triage category '{latest['triage_result']}'.]"
            )

        model = get_gemini_model()
        # Gemini takes the system prompt at model-init time, not per-call,
        # so we re-instantiate with it folded in (cheap: no network call
        # happens until generate_content()/send_message() below).
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model.model_name,
            system_instruction=self.SYSTEM_PROMPT + context_note,
        )

        history = db.get_chat_history(patient_id, limit=20)
        gemini_history = []
        for turn in history:
            if turn["user_message"]:
                gemini_history.append({"role": "user", "parts": [turn["user_message"]]})
            if turn["claude_response"]:
                gemini_history.append({"role": "model", "parts": [turn["claude_response"]]})

        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(
            user_message,
            generation_config={"max_output_tokens": 500},
        )
        reply = response.text

        db.add_chat_message(patient_id, user_message, reply)
        return reply


# ---------------------------------------------------------------------------
# Orchestration: the full pipeline, end to end
# ---------------------------------------------------------------------------

def run_full_pipeline(patient_data: dict, patient_input: dict) -> dict:
    """
    Runs Intake -> Diagnostic -> Triage -> Explainer for a new screening,
    persists the patient + screening + writes to campaign_records if
    campaign info is present, and returns everything the frontend needs
    to render both the Campaign Interface result and the Patient Interface
    report.
    """
    intake = IntakeAgent()
    diagnostic = DiagnosticAgent()
    triage = TriageAgent()
    explainer = ExplainerAgent()

    intake_payload = {**patient_data, **patient_input}
    patient_id = intake.intake(intake_payload)

    diagnosis_result = diagnostic.run(patient_input)
    if "error" in diagnosis_result and "stenosis_percentage" not in diagnosis_result:
        return {"patient_id": patient_id, "error": diagnosis_result["error"]}

    triage_result = triage.classify(diagnosis_result)

    plain_language_report = explainer.explain(
        {
            "stenosis_percentage": diagnosis_result["stenosis_percentage"],
            "triage_result": triage_result["final_triage"],
        },
        patient_data,
    )

    screening_id = db.add_screening(
        patient_id=patient_id,
        signal_file=patient_data.get("signal_file"),
        diagnosis=plain_language_report,
        risk_score=diagnosis_result["stenosis_percentage"],
        triage_result=triage_result["final_triage"],
        plaque_type=diagnosis_result.get("plaque_type"),
    )

    if patient_data.get("campaign_date") or patient_data.get("village"):
        db.add_campaign_record(
            campaign_date=patient_data.get("campaign_date", datetime.date.today().isoformat()),
            village=patient_data.get("village"),
            nurse_name=patient_data.get("nurse_name"),
        )

    return {
        "patient_id": patient_id,
        "screening_id": screening_id,
        "stenosis_percentage": diagnosis_result["stenosis_percentage"],
        "triage": triage_result,
        "plain_language_report": plain_language_report,
        "vitals": {
            "ppg": diagnosis_result.get("ppg_vitals"),
            "ecg": diagnosis_result.get("ecg_vitals"),
        },
        "model_rule_agreement": diagnosis_result.get("model_rule_agree"),
    }
