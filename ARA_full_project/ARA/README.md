# ARA — Always Reachable Assistant

**Team DYNAMO — National AI Hackathon — Problem Statement: MediMentor**

ARA is a single Streamlit application that brings AI-assisted cardiac
screening into rural health camps, and keeps patients supported afterwards
through a Claude-powered Care Companion. This repo is the **complete,
deployable project** — backend (ML + agentic AI + database) and frontend
(Campaign Interface + Patient Interface) merged into one app, ready to push
to Streamlit Community Cloud.

## What's in this repo

```
ARA/
├── app.py                          # Home page + navigation + DB init
├── diagnose.py                     # ML layer: preprocessing, feature extraction, RF prediction
├── database.py                     # SQLite layer — patients, screenings, chat_history, campaign_records
├── agents.py                       # 5-agent pipeline: Intake, Diagnostic, Triage, Explainer, CareCompanion
├── utils.py                        # Secrets/config/Claude client helper
├── train_model.py                  # Trains/retrains the two Random Forest models
├── test_pipeline.py                # Standalone backend smoke test (no UI)
│
├── pages/
│   ├── Campaign_Interface.py       # Nurse-facing: register patient, run screening
│   └── Patient_Interface.py        # Patient-facing: view report, chat with Care Companion
│
├── models/
│   ├── stenosis_regressor.joblib   # Trained Random Forest regressor
│   └── triage_classifier.joblib    # Trained Random Forest classifier
│
├── cardiac_data/                   # Demo + training reconstruction .npy files
│   └── demo_patient_{20,48,66,92}pct.npy   # Bundled demo files so you can test immediately
│
├── database/                       # SQLite file lives here at runtime (ara.db)
├── assets/                         # (empty — drop a logo/image here if you want one)
│
├── .streamlit/
│   ├── config.toml                 # App theme
│   └── secrets.toml.example        # Copy to secrets.toml and fill in your key (local only)
│
├── requirements.txt
└── README.md
```

This matches the **Streamlit Application Structure** section of the project
documentation, with `cardiac_data/` and `database/` added because the
backend needs them at runtime.

## 1. Local setup

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your real ANTHROPIC_API_KEY
streamlit run app.py
```

Four demo reconstruction files are already bundled in `cardiac_data/` so you
can run a full screening (Campaign Interface → pick a demo file → Run AI
Screening) without needing real k-Wave data. They cover a spread of
severities (~20%, 48%, 66%, 92%) and were verified to produce varied,
non-degenerate predictions across all three triage categories.

**Optional — train on real data:** if you have the real reconstructed
`.npy` files (e.g. from the `Reconstructed Data.zip` referenced in the
original Cardiac-PAI research repo), drop them into `cardiac_data/` and run
`python train_model.py` to retrain `models/*.joblib` before your demo.

**Verify the backend on its own** (no UI, useful for debugging):

```bash
python test_pipeline.py
```

## 2. Deploy to Streamlit Community Cloud

1. Push this whole folder to a GitHub repo (**make sure `models/*.joblib`
   and `cardiac_data/*.npy` are committed** — Streamlit Cloud does not run
   `train_model.py` for you, so without the committed models the app can't
   diagnose anything).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** →
   point it at your repo, branch `main`, main file path `app.py`.
3. Before (or right after) deploying, open the app's **Settings → Secrets**
   and add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-real-key-here"
   ```
4. Deploy. Streamlit auto-discovers `pages/Campaign_Interface.py` and
   `pages/Patient_Interface.py` and lists them in the sidebar — no extra
   routing config needed.

**Do not commit your real `secrets.toml`** — only `secrets.toml.example` is
checked in. `.gitignore` already excludes the real one.

## 3. How to use it once deployed

**As a nurse (Campaign Interface):**
1. Fill in camp details (date, village, nurse name).
2. Fill in patient details (name, age, gender, contact).
3. Upload a `.npy` reconstruction file, or pick one of the bundled demo
   files, and optionally a PPG/ECG `.csv` for supportive vitals.
4. Click **Run AI Screening** — the Intake → Diagnostic → Triage →
   Explainer agents run in sequence and you get a stenosis %, triage
   category, and a plain-language report.
5. Give the patient their **Patient ID** shown at the end.

**As a patient (Patient Interface):**
1. Enter your Patient ID.
2. View your simplified report under **My Report**.
3. Open **Care Companion Chat** to talk about how you're feeling, diet,
   or exercise — use the quick-prompt buttons or type your own message.

## Safety boundary (intentional, not a gap)

Every Claude-facing agent prompt in `agents.py` explicitly refuses to name,
dose, or compare medications, and redirects those questions to the
patient's doctor or clinic. This is a deliberate scope decision from the
project documentation — don't relax it when customizing prompts.

## Tech stack

Python · Streamlit · Scikit-learn (Random Forest) · OpenCV/NumPy/SciPy ·
Claude API (Anthropic) · SQLite — exactly as specified in the project
documentation's Technology Stack section, no separate backend server or
REST API.
