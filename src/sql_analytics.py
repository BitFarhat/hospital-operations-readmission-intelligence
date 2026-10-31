"""Reusable SQL analytics for the UCI hospital SQLite database."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database import DB_PATH


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def build_filter_query(
    genders=None,
    age_groups=None,
    races=None,
    admission_types=None,
    admission_sources=None,
    payers=None,
    specialties=None,
    readmission_statuses=None,
    record_sources=None,
):
    conditions = []
    params = []

    def _in(column, values):
        if values:
            placeholders = ",".join(["?"] * len(values))
            conditions.append(f"{column} IN ({placeholders})")
            params.extend(values)

    _in("e.gender", genders)
    _in("e.age", age_groups)
    _in("e.race", races)
    _in("e.admission_type", admission_types)
    _in("e.admission_source", admission_sources)
    _in("e.payer_code", payers)
    _in("e.medical_specialty", specialties)
    _in("e.readmitted", readmission_statuses)
    _in("e.record_source", record_sources)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    from_clause = f"""
        FROM encounters e
        JOIN patients p
            ON e.patient_nbr = p.patient_nbr
        {where_clause}
    """
    return from_clause, params


def _read(query, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(query, conn, params=params or [])
    finally:
        conn.close()


def get_filter_options() -> dict:
    def _col(column: str) -> list:
        frame = _read(
            f"""
            SELECT DISTINCT {column} AS value
            FROM encounters
            WHERE {column} IS NOT NULL
            ORDER BY value
            """
        )
        return frame["value"].tolist()

    return {
        "genders": _col("gender"),
        "age_groups": _col("age"),
        "races": _col("race"),
        "admission_types": _col("admission_type"),
        "admission_sources": _col("admission_source"),
        "payers": _col("payer_code"),
        "specialties": _col("medical_specialty"),
        "readmission_statuses": _col("readmitted"),
        "record_sources": _col("record_source"),
    }


def get_basic_kpis(
    genders=None,
    age_groups=None,
    races=None,
    admission_types=None,
    admission_sources=None,
    payers=None,
    specialties=None,
    readmission_statuses=None,
    record_sources=None,
) -> pd.DataFrame:
    from_clause, params = build_filter_query(
        genders=genders,
        age_groups=age_groups,
        races=races,
        admission_types=admission_types,
        admission_sources=admission_sources,
        payers=payers,
        specialties=specialties,
        readmission_statuses=readmission_statuses,
        record_sources=record_sources,
    )
    query = f"""
    SELECT
        COUNT(*) AS total_encounters,
        COUNT(DISTINCT e.patient_nbr) AS total_patients,
        ROUND(AVG(e.time_in_hospital), 2) AS average_los_days,
        ROUND(100.0 * AVG(e.readmission_30d), 2) AS readmission_30d_rate_pct,
        ROUND(AVG(e.num_medications), 2) AS avg_medications,
        ROUND(AVG(e.num_lab_procedures), 2) AS avg_lab_procedures,
        ROUND(AVG(e.num_procedures), 2) AS avg_procedures
    {from_clause}
    """
    return _read(query, params)


def _grouped_counts(group_col: str, **filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    query = f"""
    SELECT
        {group_col} AS category,
        COUNT(*) AS encounter_count
    {from_clause}
    GROUP BY {group_col}
    ORDER BY encounter_count DESC
    """
    return _read(query, params)


def encounters_by_admission_type(**filters) -> pd.DataFrame:
    return _grouped_counts("e.admission_type", **filters).rename(
        columns={"category": "admission_type"}
    )


def encounters_by_admission_source(**filters) -> pd.DataFrame:
    return _grouped_counts("e.admission_source", **filters).rename(
        columns={"category": "admission_source"}
    )


def encounters_by_gender(**filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    query = f"""
    SELECT
        e.gender,
        COUNT(*) AS encounter_count,
        COUNT(DISTINCT e.patient_nbr) AS patient_count
    {from_clause}
    GROUP BY e.gender
    ORDER BY encounter_count DESC
    """
    return _read(query, params)


def encounters_by_age(**filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    query = f"""
    SELECT
        e.age AS age_group,
        COUNT(*) AS encounter_count
    {from_clause}
    GROUP BY e.age
    ORDER BY e.age
    """
    return _read(query, params)


def encounters_by_race(**filters) -> pd.DataFrame:
    return _grouped_counts("e.race", **filters).rename(columns={"category": "race"})


def readmission_outcome_distribution(**filters) -> pd.DataFrame:
    return _grouped_counts("e.readmitted", **filters).rename(
        columns={"category": "readmitted"}
    )


def length_of_stay_distribution(**filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    query = f"""
    SELECT
        e.time_in_hospital AS length_of_stay,
        COUNT(*) AS encounter_count
    {from_clause}
    GROUP BY e.time_in_hospital
    ORDER BY e.time_in_hospital
    """
    return _read(query, params)


def readmission_rate_by_age(**filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    query = f"""
    SELECT
        e.age AS age_group,
        COUNT(*) AS encounter_count,
        ROUND(100.0 * AVG(e.readmission_30d), 2) AS readmission_30d_rate_pct
    {from_clause}
    GROUP BY e.age
    ORDER BY e.age
    """
    return _read(query, params)


def readmission_rate_by_admission_type(**filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    query = f"""
    SELECT
        e.admission_type,
        COUNT(*) AS encounter_count,
        ROUND(100.0 * AVG(e.readmission_30d), 2) AS readmission_30d_rate_pct
    {from_clause}
    GROUP BY e.admission_type
    ORDER BY readmission_30d_rate_pct DESC
    """
    return _read(query, params)


def procedure_utilization_by_age(**filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    query = f"""
    SELECT
        e.age AS age_group,
        ROUND(AVG(e.num_procedures), 2) AS avg_procedures,
        ROUND(AVG(e.num_lab_procedures), 2) AS avg_lab_procedures,
        ROUND(AVG(e.num_medications), 2) AS avg_medications
    {from_clause}
    GROUP BY e.age
    ORDER BY e.age
    """
    return _read(query, params)


def medication_utilization_by_readmission(**filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    query = f"""
    SELECT
        e.readmitted,
        ROUND(AVG(e.num_medications), 2) AS avg_medications,
        ROUND(AVG(e.num_lab_procedures), 2) AS avg_lab_procedures,
        ROUND(AVG(e.num_procedures), 2) AS avg_procedures,
        COUNT(*) AS encounter_count
    {from_clause}
    GROUP BY e.readmitted
    ORDER BY e.readmitted
    """
    return _read(query, params)


def diagnosis_group_counts(**filters) -> pd.DataFrame:
    return _grouped_counts("e.diag_1_group", **filters).rename(
        columns={"category": "diag_1_group"}
    )


def insulin_distribution(**filters) -> pd.DataFrame:
    return _grouped_counts("e.insulin", **filters).rename(columns={"category": "insulin"})


def readmission_rate_by_diagnosis_group(min_encounters: int = 50, **filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    query = f"""
    SELECT
        e.diag_1_group,
        COUNT(*) AS encounter_count,
        ROUND(100.0 * AVG(e.readmission_30d), 2) AS readmission_30d_rate_pct
    {from_clause}
    GROUP BY e.diag_1_group
    HAVING COUNT(*) >= {int(min_encounters)}
    ORDER BY readmission_30d_rate_pct DESC
    """
    return _read(query, params)


def preview_encounters(limit: int = 250, search_query: str | None = None, **filters) -> pd.DataFrame:
    from_clause, params = build_filter_query(**filters)
    search_clause = ""
    if search_query and search_query.strip():
        term = f"%{search_query.strip()}%"
        if "WHERE" in from_clause:
            search_clause = " AND (CAST(e.encounter_id AS TEXT) LIKE ? OR CAST(e.patient_nbr AS TEXT) LIKE ?)"
        else:
            search_clause = " WHERE (CAST(e.encounter_id AS TEXT) LIKE ? OR CAST(e.patient_nbr AS TEXT) LIKE ?)"
        params.extend([term, term])
    query = f"""
    SELECT
        e.encounter_id,
        e.patient_nbr,
        e.gender,
        e.age,
        e.race,
        e.admission_type,
        e.admission_source,
        e.payer_code,
        e.medical_specialty,
        e.time_in_hospital,
        e.num_medications,
        e.num_lab_procedures,
        e.num_procedures,
        e.number_diagnoses,
        e.diag_1,
        e.diag_1_group,
        e.readmitted,
        e.record_source
    {from_clause}
    {search_clause}
    ORDER BY e.encounter_id DESC
    LIMIT {int(limit)}
    """
    return _read(query, params)


if __name__ == "__main__":
    print("\n--- BASIC KPIs ---")
    print(get_basic_kpis())
    print("\n--- BY ADMISSION TYPE ---")
    print(encounters_by_admission_type())
    print("\n--- READMISSION OUTCOMES ---")
    print(readmission_outcome_distribution())
    print("\n--- DIAGNOSIS GROUPS ---")
    print(diagnosis_group_counts().head(15))
