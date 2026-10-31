"""Prototype admin login and demo encounter entry for the UCI schema."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import (
    AGE_LABELS,
    RECORD_SOURCE_DEMO,
    map_diag_group,
)
from database import DB_PATH, ENCOUNTER_COLUMNS, get_connection

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "hospital123"

GENDER_OPTIONS = ["Female", "Male", "Unknown/Invalid", "Unknown"]
RACE_OPTIONS = [
    "Caucasian",
    "AfricanAmerican",
    "Hispanic",
    "Asian",
    "Other",
    "Unknown",
]
INSULIN_OPTIONS = ["No", "Steady", "Up", "Down"]
CHANGE_OPTIONS = ["No", "Ch"]
DIABETES_MED_OPTIONS = ["Yes", "No"]
READMITTED_OPTIONS = ["NO", ">30", "<30"]


def admin_login():
    """Handle admin login for the hospital operations prototype."""
    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if st.session_state.admin_logged_in:
        return True

    _spacer, card, _right = st.columns([0.6, 1.4, 0.6])
    with card:
        st.markdown(
            """
            <div style="margin-top: 2.5rem; margin-bottom: 1.2rem; text-align: center;">
              <div style="display: inline-flex; align-items: center; justify-content: center; width: 50px; height: 50px;
                          border-radius: 14px; background: linear-gradient(135deg, #9385F5, #4880FF); color: white;
                          font-size: 1.5rem; font-weight: bold; margin-bottom: 0.6rem; box-shadow: 0 4px 14px rgba(147,133,245,0.35);">
                🏥
              </div>
              <div style="color: #9385F5; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase;">
                Hospital Operations & Readmission AI
              </div>
              <h2 style="color: #20243A; margin: 0.25rem 0 0.35rem 0; font-size: 1.65rem; font-weight: 800; letter-spacing: -0.02em;">
                Clinical Operations Portal
              </h2>
              <p style="color: #64748B; font-size: 0.92rem; margin: 0;">
                Modern SaaS Analytics Platform · Portfolio Demonstration
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                """
                <div style="background: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 8px; padding: 10px 14px; margin-bottom: 14px;">
                  <div style="font-weight: 700; font-size: 0.84rem; color: #5B21B6; display: flex; align-items: center; gap: 6px;">
                    <span>🔑</span> Demo Evaluation Credentials
                  </div>
                  <div style="font-size: 0.82rem; color: #6D28D9; margin-top: 3px;">
                    Username: <code style="background:#EDE9FE; color:#4C1D95; padding:2px 6px; border-radius:4px;">admin</code> &nbsp;|&nbsp; 
                    Password: <code style="background:#EDE9FE; color:#4C1D95; padding:2px 6px; border-radius:4px;">hospital123</code>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            username = st.text_input("Username", value=st.session_state.get("_demo_user", ""), placeholder="admin")
            password = st.text_input("Password", value=st.session_state.get("_demo_pass", ""), type="password", placeholder="••••••••")

            col_sub, col_fill = st.columns([1, 1])
            with col_sub:
                if st.button("Sign In", type="primary", use_container_width=True):
                    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                        st.session_state.admin_logged_in = True
                        st.success("Login successful. Initializing dashboard...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

            with col_fill:
                if st.button("⚡ Quick Fill Demo", use_container_width=True):
                    st.session_state._demo_user = ADMIN_USERNAME
                    st.session_state._demo_pass = ADMIN_PASSWORD
                    st.session_state.admin_logged_in = True
                    st.rerun()

            st.caption("Local portfolio prototype login. Does not transmit data outside local environment.")

    return False


def admin_logout_button() -> None:
    if st.button("Sign out", use_container_width=True):
        st.session_state.admin_logged_in = False
        st.rerun()


def _lookup_descriptions(table: str, id_col: str) -> list[tuple[int, str]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT {id_col}, description FROM {table} ORDER BY {id_col}"
        ).fetchall()
        return [(int(i), desc) for i, desc in rows]
    finally:
        conn.close()


def _distinct_or_default(column: str, defaults: list[str]) -> list[str]:
    conn = get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT DISTINCT {column}
            FROM encounters
            WHERE {column} IS NOT NULL
            ORDER BY {column}
            """
        ).fetchall()
        values = [r[0] for r in rows]
        return values or defaults
    except sqlite3.Error:
        return defaults
    finally:
        conn.close()


def add_patient_form():
    st.info(
        "Records added here are **demo / user-entered** rows. "
        "They are stored with `record_source = demo_user` and are **not** "
        "original UCI Diabetes 130-US Hospitals records."
    )

    admission_types = _lookup_descriptions("lookup_admission_type", "admission_type_id")
    admission_sources = _lookup_descriptions(
        "lookup_admission_source", "admission_source_id"
    )
    discharge_disps = _lookup_descriptions(
        "lookup_discharge_disposition", "discharge_disposition_id"
    )
    payers = _distinct_or_default("payer_code", ["Unknown", "MC", "MD", "HM", "SP", "BC"])
    specialties = _distinct_or_default(
        "medical_specialty", ["Unknown", "InternalMedicine", "Emergency/Trauma"]
    )

    type_labels = [f"{i} — {name}" for i, name in admission_types] or ["1 — Emergency"]
    source_labels = [f"{i} — {name}" for i, name in admission_sources] or [
        "7 — Emergency Room"
    ]
    discharge_labels = [f"{i} — {name}" for i, name in discharge_disps] or [
        "1 — Discharged to home"
    ]

    with st.form("add_patient_form"):
        st.markdown("**Patient identity**")
        id1, id2, id3, id4 = st.columns(4)
        with id1:
            patient_nbr = st.number_input("Patient ID (`patient_nbr`)", min_value=1, step=1)
        with id2:
            gender = st.selectbox("Gender", GENDER_OPTIONS)
        with id3:
            age = st.selectbox("Age group", AGE_LABELS, index=7)
        with id4:
            race = st.selectbox("Race", RACE_OPTIONS)

        st.markdown("**Encounter**")
        e1, e2, e3 = st.columns(3)
        with e1:
            admission_type_label = st.selectbox("Admission type", type_labels)
            payer_code = st.selectbox("Payer / insurance", payers)
            time_in_hospital = st.number_input(
                "Time in hospital (days)", min_value=1, max_value=14, value=3, step=1
            )
        with e2:
            admission_source_label = st.selectbox("Admission source", source_labels)
            medical_specialty = st.selectbox("Medical specialty", specialties)
            num_lab_procedures = st.number_input(
                "Number of lab procedures", min_value=0, max_value=150, value=40, step=1
            )
        with e3:
            discharge_label = st.selectbox("Discharge disposition", discharge_labels)
            num_procedures = st.number_input(
                "Number of procedures", min_value=0, max_value=10, value=0, step=1
            )
            num_medications = st.number_input(
                "Number of medications", min_value=0, max_value=80, value=15, step=1
            )

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            number_outpatient = st.number_input(
                "Prior outpatient visits (year)", min_value=0, max_value=50, value=0, step=1
            )
        with v2:
            number_emergency = st.number_input(
                "Prior emergency visits (year)", min_value=0, max_value=50, value=0, step=1
            )
        with v3:
            number_inpatient = st.number_input(
                "Prior inpatient visits (year)", min_value=0, max_value=50, value=0, step=1
            )
        with v4:
            number_diagnoses = st.number_input(
                "Number of diagnoses", min_value=1, max_value=16, value=9, step=1
            )

        d1, d2, d3 = st.columns(3)
        with d1:
            diag_1 = st.text_input("Primary diagnosis (`diag_1` ICD-9)", value="250.00")
        with d2:
            diag_2 = st.text_input("Secondary diagnosis (`diag_2`)", value="")
        with d3:
            diag_3 = st.text_input("Additional diagnosis (`diag_3`)", value="")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            insulin = st.selectbox("Insulin", INSULIN_OPTIONS)
        with m2:
            change = st.selectbox("Change of diabetes medications", CHANGE_OPTIONS)
        with m3:
            diabetes_med = st.selectbox("Diabetes medication prescribed", DIABETES_MED_OPTIONS)
        with m4:
            readmitted = st.selectbox(
                "Readmission outcome (demo label only)",
                READMITTED_OPTIONS,
            )
        submitted = st.form_submit_button("Save demo encounter", type="primary")

    if not submitted:
        return

    admission_type_id = int(admission_type_label.split(" — ", 1)[0])
    admission_source_id = int(admission_source_label.split(" — ", 1)[0])
    discharge_disposition_id = int(discharge_label.split(" — ", 1)[0])
    admission_type = admission_type_label.split(" — ", 1)[1]
    admission_source = admission_source_label.split(" — ", 1)[1]
    discharge_disposition = discharge_label.split(" — ", 1)[1]
    diag_1_clean = diag_1.strip() or "Unknown"
    diag_2_clean = diag_2.strip() or "Unknown"
    diag_3_clean = diag_3.strip() or "Unknown"
    readmission_30d = 1 if readmitted == "<30" else 0

    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT 1 FROM patients WHERE patient_nbr = ?",
            (int(patient_nbr),),
        ).fetchone()
        if not existing:
            conn.execute(
                """
                INSERT INTO patients (patient_nbr, gender, race, record_source)
                VALUES (?, ?, ?, ?)
                """,
                (int(patient_nbr), gender, race, RECORD_SOURCE_DEMO),
            )
            patient_note = "Created a new demo patient."
        else:
            patient_note = (
                "Existing patient found. A new encounter was added without duplicating "
                "the patient row."
            )

        max_enc = conn.execute("SELECT MAX(encounter_id) FROM encounters").fetchone()[0]
        new_encounter_id = int(max_enc or 0) + 1

        med_defaults = {col: "No" for col in (
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "acetohexamide",
            "glipizide",
            "glyburide",
            "tolbutamide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "troglitazone",
            "tolazamide",
            "examide",
            "citoglipton",
            "glyburide-metformin",
            "glipizide-metformin",
            "glimepiride-pioglitazone",
            "metformin-rosiglitazone",
            "metformin-pioglitazone",
        )}
        row = {
            "encounter_id": new_encounter_id,
            "patient_nbr": int(patient_nbr),
            "race": race,
            "gender": gender,
            "age": age,
            "weight": "Unknown",
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "admission_type": admission_type,
            "discharge_disposition": discharge_disposition,
            "admission_source": admission_source,
            "time_in_hospital": int(time_in_hospital),
            "payer_code": payer_code,
            "medical_specialty": medical_specialty,
            "num_lab_procedures": int(num_lab_procedures),
            "num_procedures": int(num_procedures),
            "num_medications": int(num_medications),
            "number_outpatient": int(number_outpatient),
            "number_emergency": int(number_emergency),
            "number_inpatient": int(number_inpatient),
            "diag_1": diag_1_clean,
            "diag_2": diag_2_clean,
            "diag_3": diag_3_clean,
            "number_diagnoses": int(number_diagnoses),
            "diag_1_group": map_diag_group(diag_1_clean),
            "max_glu_serum": "Unknown",
            "A1Cresult": "Unknown",
            **med_defaults,
            "insulin": insulin,
            "change": change,
            "diabetesMed": diabetes_med,
            "readmitted": readmitted,
            "readmission_30d": readmission_30d,
            "expired_or_hospice": 1 if discharge_disposition_id in {11, 13, 14, 19, 20, 21} else 0,
            "record_source": RECORD_SOURCE_DEMO,
        }
        placeholders = ",".join(["?"] * len(ENCOUNTER_COLUMNS))
        quoted = ",".join(
            f'"{c}"' if "-" in c else c for c in ENCOUNTER_COLUMNS
        )
        conn.execute(
            f"INSERT INTO encounters ({quoted}) VALUES ({placeholders})",
            [row[c] for c in ENCOUNTER_COLUMNS],
        )
        conn.commit()
        st.success(
            f"Saved demo encounter {new_encounter_id} for patient {int(patient_nbr)}."
        )
        st.caption(patient_note)
        st.cache_data.clear()
    except Exception as exc:
        conn.rollback()
        st.error(f"Error adding encounter: {exc}")
    finally:
        conn.close()

    # Show recent demo encounters if any exist
    conn = get_connection()
    try:
        import pandas as pd
        demo_rows = pd.read_sql_query(
            """
            SELECT
                encounter_id,
                patient_nbr,
                gender,
                age,
                race,
                admission_type,
                time_in_hospital,
                num_medications,
                diag_1,
                diag_1_group,
                readmitted,
                record_source
            FROM encounters
            WHERE record_source = 'demo_user'
            ORDER BY encounter_id DESC
            LIMIT 10
            """,
            conn,
        )
        if not demo_rows.empty:
            st.markdown("---")
            st.markdown("#### 📋 Recent Demo Encounters (`record_source = demo_user`)")
            st.caption("These user-entered encounters are stored in SQLite and can be filtered across the dashboard.")
            st.dataframe(demo_rows, width="stretch", hide_index=True)
    except Exception:
        pass
    finally:
        conn.close()
