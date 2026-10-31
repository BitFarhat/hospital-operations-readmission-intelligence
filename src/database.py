"""Build a UCI-based SQLite database without touching the live file until verified.

Safety:
- This module writes hospital_uci.db, verifies row counts, then copies it to
  hospital.db. It never uses pandas.to_sql(..., if_exists='replace') on the
  only existing hospital.db.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import (
    MEDICATION_COLUMNS,
    RAW_ID_MAPPING,
    RECORD_SOURCE_DEMO,
    RECORD_SOURCE_UCI,
    ROOT_DIR,
    parse_id_mapping,
    patient_table_from_encounters,
    prepare_encounter_frame,
)

DB_PATH = ROOT_DIR / "hospital.db"
STAGING_DB_PATH = ROOT_DIR / "hospital_uci.db"

LOOKUP_DDL = """
CREATE TABLE lookup_admission_type (
    admission_type_id INTEGER PRIMARY KEY,
    description TEXT NOT NULL
);

CREATE TABLE lookup_discharge_disposition (
    discharge_disposition_id INTEGER PRIMARY KEY,
    description TEXT NOT NULL
);

CREATE TABLE lookup_admission_source (
    admission_source_id INTEGER PRIMARY KEY,
    description TEXT NOT NULL
);
"""

PATIENT_DDL = """
CREATE TABLE patients (
    patient_nbr INTEGER PRIMARY KEY,
    gender TEXT,
    race TEXT,
    record_source TEXT NOT NULL DEFAULT 'uci'
);
"""

ENCOUNTER_DDL = """
CREATE TABLE encounters (
    encounter_id INTEGER PRIMARY KEY,
    patient_nbr INTEGER NOT NULL,
    race TEXT,
    gender TEXT,
    age TEXT,
    weight TEXT,
    admission_type_id INTEGER,
    discharge_disposition_id INTEGER,
    admission_source_id INTEGER,
    admission_type TEXT,
    discharge_disposition TEXT,
    admission_source TEXT,
    time_in_hospital INTEGER,
    payer_code TEXT,
    medical_specialty TEXT,
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    num_medications INTEGER,
    number_outpatient INTEGER,
    number_emergency INTEGER,
    number_inpatient INTEGER,
    diag_1 TEXT,
    diag_2 TEXT,
    diag_3 TEXT,
    number_diagnoses INTEGER,
    diag_1_group TEXT,
    max_glu_serum TEXT,
    A1Cresult TEXT,
    metformin TEXT,
    repaglinide TEXT,
    nateglinide TEXT,
    chlorpropamide TEXT,
    glimepiride TEXT,
    acetohexamide TEXT,
    glipizide TEXT,
    glyburide TEXT,
    tolbutamide TEXT,
    pioglitazone TEXT,
    rosiglitazone TEXT,
    acarbose TEXT,
    miglitol TEXT,
    troglitazone TEXT,
    tolazamide TEXT,
    examide TEXT,
    citoglipton TEXT,
    insulin TEXT,
    "glyburide-metformin" TEXT,
    "glipizide-metformin" TEXT,
    "glimepiride-pioglitazone" TEXT,
    "metformin-rosiglitazone" TEXT,
    "metformin-pioglitazone" TEXT,
    change TEXT,
    diabetesMed TEXT,
    readmitted TEXT,
    readmission_30d INTEGER,
    expired_or_hospice INTEGER,
    record_source TEXT NOT NULL DEFAULT 'uci',
    FOREIGN KEY (patient_nbr) REFERENCES patients(patient_nbr)
);
"""

INDEX_DDL = """
CREATE INDEX idx_encounters_patient ON encounters(patient_nbr);
CREATE INDEX idx_encounters_readmitted ON encounters(readmitted);
CREATE INDEX idx_encounters_gender ON encounters(gender);
CREATE INDEX idx_encounters_age ON encounters(age);
CREATE INDEX idx_encounters_race ON encounters(race);
CREATE INDEX idx_encounters_admission_type ON encounters(admission_type);
CREATE INDEX idx_encounters_source ON encounters(admission_source);
CREATE INDEX idx_encounters_payer ON encounters(payer_code);
CREATE INDEX idx_encounters_specialty ON encounters(medical_specialty);
CREATE INDEX idx_encounters_record_source ON encounters(record_source);
"""

ENCOUNTER_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "admission_type",
    "discharge_disposition",
    "admission_source",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "diag_1_group",
    "max_glu_serum",
    "A1Cresult",
    *MEDICATION_COLUMNS,
    "change",
    "diabetesMed",
    "readmitted",
    "readmission_30d",
    "expired_or_hospice",
    "record_source",
]


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _write_lookups(conn: sqlite3.Connection, mapping: dict[str, dict[int, str]]) -> None:
    conn.executemany(
        "INSERT INTO lookup_admission_type (admission_type_id, description) VALUES (?, ?)",
        sorted(mapping.get("admission_type_id", {}).items()),
    )
    conn.executemany(
        "INSERT INTO lookup_discharge_disposition (discharge_disposition_id, description) VALUES (?, ?)",
        sorted(mapping.get("discharge_disposition_id", {}).items()),
    )
    conn.executemany(
        "INSERT INTO lookup_admission_source (admission_source_id, description) VALUES (?, ?)",
        sorted(mapping.get("admission_source_id", {}).items()),
    )


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("PRAGMA foreign_keys = ON;")
    conn.executescript(LOOKUP_DDL)
    conn.executescript(PATIENT_DDL)
    conn.executescript(ENCOUNTER_DDL)
    conn.executescript(INDEX_DDL)


def verify_database(db_path: Path | None = None) -> dict:
    conn = get_connection(db_path)
    try:
        patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        encounters = conn.execute("SELECT COUNT(*) FROM encounters").fetchone()[0]
        distinct_patients = conn.execute(
            "SELECT COUNT(DISTINCT patient_nbr) FROM encounters"
        ).fetchone()[0]
        orphans = conn.execute(
            """
            SELECT COUNT(*) FROM encounters e
            LEFT JOIN patients p ON e.patient_nbr = p.patient_nbr
            WHERE p.patient_nbr IS NULL
            """
        ).fetchone()[0]
        readmit = conn.execute(
            "SELECT SUM(readmission_30d) FROM encounters"
        ).fetchone()[0]
        demo = conn.execute(
            "SELECT COUNT(*) FROM encounters WHERE record_source = ?",
            (RECORD_SOURCE_DEMO,),
        ).fetchone()[0]
        uci = conn.execute(
            "SELECT COUNT(*) FROM encounters WHERE record_source = ?",
            (RECORD_SOURCE_UCI,),
        ).fetchone()[0]
    finally:
        conn.close()

    summary = {
        "patients": patients,
        "encounters": encounters,
        "distinct_patients_in_encounters": distinct_patients,
        "orphan_encounters": orphans,
        "readmission_30d_positives": int(readmit or 0),
        "uci_encounters": uci,
        "demo_encounters": demo,
    }
    if patients != distinct_patients:
        raise RuntimeError(f"Patient/encounter mismatch: {summary}")
    if orphans:
        raise RuntimeError(f"Orphan encounters found: {summary}")
    if encounters != 101766:
        # Allow extra demo rows after the initial load.
        if encounters < 101766:
            raise RuntimeError(f"Unexpected encounter count: {summary}")
    return summary


def create_database(replace_live: bool = True) -> dict:
    mapping = parse_id_mapping(RAW_ID_MAPPING)
    encounters = prepare_encounter_frame(mapping=mapping)
    patients = patient_table_from_encounters(encounters)

    if STAGING_DB_PATH.exists():
        STAGING_DB_PATH.unlink()

    conn = sqlite3.connect(STAGING_DB_PATH)
    try:
        _create_schema(conn)
        _write_lookups(conn, mapping)
        patients.to_sql("patients", conn, if_exists="append", index=False)
        encounters[ENCOUNTER_COLUMNS].to_sql(
            "encounters",
            conn,
            if_exists="append",
            index=False,
            chunksize=10000,
        )
        conn.commit()
    finally:
        conn.close()

    summary = verify_database(STAGING_DB_PATH)
    if replace_live:
        # Copy verified staging DB over hospital.db.
        import shutil

        shutil.copy2(STAGING_DB_PATH, DB_PATH)
        summary["live_db"] = str(DB_PATH)
    summary["staging_db"] = str(STAGING_DB_PATH)
    print("UCI hospital database created successfully.")
    print(summary)
    return summary


if __name__ == "__main__":
    create_database(replace_live=True)
