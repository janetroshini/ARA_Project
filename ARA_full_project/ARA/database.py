"""
database.py
-----------
SQLite persistence layer for ARA. Implements the 4-table schema from the
project documentation: Patient, Screening, Chat History, Campaign Records.

No ORM — plain sqlite3, matching the "lightweight, serverless, single-file
storage" decision in the tech stack. Every function opens and closes its
own connection so it's safe to call from Streamlit's rerun model (no
shared connection held across reruns/threads).
"""

import sqlite3
import datetime
from contextlib import contextmanager

from utils import DB_PATH, ensure_dirs


@contextmanager
def get_conn():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all tables if they don't already exist. Safe to call on every app boot."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL,
                age             INTEGER,
                gender          TEXT,
                contact_number  TEXT,
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS screenings (
                screening_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id      INTEGER NOT NULL,
                signal_file     TEXT,
                diagnosis       TEXT,
                risk_score      REAL,
                triage_result   TEXT,
                plaque_type     TEXT,
                created_at      TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                chat_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id      INTEGER NOT NULL,
                user_message    TEXT,
                claude_response TEXT,
                timestamp       TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            );

            CREATE TABLE IF NOT EXISTS campaign_records (
                campaign_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_date   TEXT,
                village         TEXT,
                nurse_name      TEXT,
                created_at      TEXT NOT NULL
            );
            """
        )


def _now():
    return datetime.datetime.utcnow().isoformat()


# --- Patients ------------------------------------------------------------

def add_patient(name, age, gender, contact_number):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO patients (name, age, gender, contact_number, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, age, gender, contact_number, _now()),
        )
        return cur.lastrowid


def get_patient(patient_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        return dict(row) if row else None


def list_patients(limit=200):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM patients ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# --- Screenings ------------------------------------------------------------

def add_screening(patient_id, signal_file, diagnosis, risk_score, triage_result, plaque_type=None):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO screenings "
            "(patient_id, signal_file, diagnosis, risk_score, triage_result, plaque_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (patient_id, signal_file, diagnosis, risk_score, triage_result, plaque_type, _now()),
        )
        return cur.lastrowid


def get_screenings_for_patient(patient_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM screenings WHERE patient_id = ? ORDER BY created_at DESC",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_latest_screening(patient_id):
    screenings = get_screenings_for_patient(patient_id)
    return screenings[0] if screenings else None


# --- Chat history ------------------------------------------------------------

def add_chat_message(patient_id, user_message, claude_response):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_history (patient_id, user_message, claude_response, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (patient_id, user_message, claude_response, _now()),
        )
        return cur.lastrowid


def get_chat_history(patient_id, limit=50):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_history WHERE patient_id = ? ORDER BY timestamp ASC LIMIT ?",
            (patient_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# --- Campaign records ------------------------------------------------------------

def add_campaign_record(campaign_date, village, nurse_name):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO campaign_records (campaign_date, village, nurse_name, created_at) "
            "VALUES (?, ?, ?, ?)",
            (campaign_date, village, nurse_name, _now()),
        )
        return cur.lastrowid


def list_campaign_records(limit=200):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM campaign_records ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
