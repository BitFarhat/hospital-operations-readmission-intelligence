"""Load and prepare the UCI Diabetes 130-US Hospitals dataset.

This module is the source of truth for:
- locating the official CSV
- missing-value cleanup
- derived fields (30-day readmission flag, diagnosis group)
- pandas helpers used by the Streamlit preview layer

The dataset contains diabetes-related hospital encounters from 130 US
hospitals (1999-2008). It is not a general all-disease population.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_ENCOUNTERS = RAW_DIR / "diabetic_data.csv"
RAW_ID_MAPPING = RAW_DIR / "IDS_mapping.csv"

RECORD_SOURCE_UCI = "uci"
RECORD_SOURCE_DEMO = "demo_user"

AGE_LABELS = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]

MEDICATION_COLUMNS = [
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
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Discharge IDs that indicate death or hospice (cannot be readmitted).
EXPIRED_OR_HOSPICE_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}


class DataLoadError(FileNotFoundError):
    """Raised when the UCI source file is missing."""


def _require_raw_file() -> None:
    if not RAW_ENCOUNTERS.exists():
        raise DataLoadError(
            f"Missing UCI source file: {RAW_ENCOUNTERS}. "
            "Run `python scripts/download_uci.py` from the project root."
        )


def _clean_token(value) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if text in {"", "?", "NULL", "None", "nan", "NaN"}:
        return "Unknown"
    return text


def parse_id_mapping(path: Path | None = None) -> dict[str, dict[int, str]]:
    """Parse UCI IDS_mapping.csv (three lookup tables separated by blank rows)."""
    path = path or RAW_ID_MAPPING
    if not path.exists():
        raise DataLoadError(f"Missing ID mapping file: {path}")

    sections: dict[str, dict[int, str]] = {}
    current: str | None = None
    current_map: dict[int, str] = {}

    def _flush() -> None:
        nonlocal current, current_map
        if current:
            sections[current] = current_map
        current = None
        current_map = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line == ",":
            _flush()
            continue
        if line.endswith(",description"):
            _flush()
            current = line.split(",", 1)[0].strip()
            current_map = {}
            continue
        if current is None:
            continue
        if line.startswith('"'):
            continue
        parts = next(csv.reader([line]))
        if len(parts) < 2 or not str(parts[0]).strip().isdigit():
            continue
        key = int(parts[0].strip())
        desc = _clean_token(parts[1])
        if desc == "Unknown":
            desc = "Not Available"
        current_map[key] = desc.strip()
    _flush()
    return sections


def map_diag_group(code) -> str:
    """ICD-9 chapter-style groups used in Strack et al. (2014) on this dataset."""
    if pd.isna(code):
        return "Unknown"
    text = str(code).strip()
    if text in {"", "?"}:
        return "Unknown"
    if text.upper().startswith("E"):
        return "Injury / poisoning"
    if text.upper().startswith("V"):
        return "Supplementary classification"
    try:
        number = float(text)
    except ValueError:
        return "Unknown"

    if 390 <= number <= 459 or number == 785:
        return "Circulatory"
    if 460 <= number <= 519 or number == 786:
        return "Respiratory"
    if 520 <= number <= 579 or number == 787:
        return "Digestive"
    if 250 <= number < 251:
        return "Diabetes"
    if 800 <= number <= 999:
        return "Injury / poisoning"
    if 710 <= number <= 739:
        return "Musculoskeletal"
    if 580 <= number <= 629 or number == 788:
        return "Genitourinary"
    if 140 <= number <= 239:
        return "Neoplasms"
    if number in {780, 781, 784} or 790 <= number <= 799:
        return "Other symptoms"
    if 240 <= number <= 279:
        return "Endocrine / metabolic (excl. diabetes primary)"
    if 680 <= number <= 709 or number == 782:
        return "Skin"
    if 1 <= number <= 139:
        return "Infectious / parasitic"
    if 290 <= number <= 319:
        return "Mental"
    if 320 <= number <= 389:
        return "Nervous / sense organs"
    if 630 <= number <= 679:
        return "Pregnancy / childbirth"
    if 740 <= number <= 759:
        return "Congenital"
    if 760 <= number <= 779:
        return "Perinatal"
    return "Other"


def load_raw_encounters(path: Path | None = None) -> pd.DataFrame:
    _require_raw_file()
    return pd.read_csv(path or RAW_ENCOUNTERS, low_memory=True)


def prepare_encounter_frame(
    raw: pd.DataFrame | None = None,
    mapping: dict[str, dict[int, str]] | None = None,
) -> pd.DataFrame:
    """Clean the official CSV and add derived analytics/ML columns."""
    raw = load_raw_encounters() if raw is None else raw.copy()
    mapping = mapping or parse_id_mapping()

    admission_types = mapping.get("admission_type_id", {})
    discharge_disp = mapping.get("discharge_disposition_id", {})
    admission_sources = mapping.get("admission_source_id", {})

    frame = raw.copy()
    for col in (
        "race",
        "gender",
        "age",
        "weight",
        "payer_code",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
        "max_glu_serum",
        "A1Cresult",
        "change",
        "diabetesMed",
        "readmitted",
        *MEDICATION_COLUMNS,
    ):
        if col in frame.columns:
            frame[col] = frame[col].map(_clean_token)

    frame["admission_type"] = frame["admission_type_id"].map(
        lambda i: admission_types.get(int(i), "Not Available")
    )
    frame["discharge_disposition"] = frame["discharge_disposition_id"].map(
        lambda i: discharge_disp.get(int(i), "Not Available")
    )
    frame["admission_source"] = frame["admission_source_id"].map(
        lambda i: admission_sources.get(int(i), "Not Available")
    )
    frame["diag_1_group"] = frame["diag_1"].map(map_diag_group)
    frame["readmission_30d"] = (frame["readmitted"] == "<30").astype(int)
    frame["record_source"] = RECORD_SOURCE_UCI
    frame["expired_or_hospice"] = frame["discharge_disposition_id"].isin(
        EXPIRED_OR_HOSPICE_DISCHARGE_IDS
    ).astype(int)
    return frame


def patient_table_from_encounters(encounters: pd.DataFrame) -> pd.DataFrame:
    """One row per patient using the first encounter in file order."""
    first = encounters.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    patients = first[["patient_nbr", "gender", "race", "record_source"]].copy()
    return patients.reset_index(drop=True)


def filter_stays(
    stays: pd.DataFrame,
    genders: list[str] | None = None,
    age_groups: list[str] | None = None,
    races: list[str] | None = None,
    admission_types: list[str] | None = None,
    admission_sources: list[str] | None = None,
    payers: list[str] | None = None,
    specialties: list[str] | None = None,
    readmission_statuses: list[str] | None = None,
    record_sources: list[str] | None = None,
) -> pd.DataFrame:
    out = stays
    checks = (
        ("gender", genders),
        ("age", age_groups),
        ("race", races),
        ("admission_type", admission_types),
        ("admission_source", admission_sources),
        ("payer_code", payers),
        ("medical_specialty", specialties),
        ("readmitted", readmission_statuses),
        ("record_source", record_sources),
    )
    for column, values in checks:
        if values is not None:
            out = out[out[column].isin(values)]
    return out.copy()


def compute_kpis(stays: pd.DataFrame) -> dict:
    if stays.empty:
        return {
            "n_patients": 0,
            "n_encounters": 0,
            "avg_los_days": 0.0,
            "readmission_30d_rate_pct": 0.0,
            "avg_medications": 0.0,
            "avg_lab_procedures": 0.0,
            "avg_procedures": 0.0,
        }
    return {
        "n_patients": int(stays["patient_nbr"].nunique()),
        "n_encounters": int(len(stays)),
        "avg_los_days": float(stays["time_in_hospital"].mean()),
        "readmission_30d_rate_pct": float(stays["readmission_30d"].mean() * 100),
        "avg_medications": float(stays["num_medications"].mean()),
        "avg_lab_procedures": float(stays["num_lab_procedures"].mean()),
        "avg_procedures": float(stays["num_procedures"].mean()),
    }


def print_kpi_report(stays: pd.DataFrame) -> None:
    kpis = compute_kpis(stays)
    print("Hospital Operations - UCI KPI check")
    print(f"  Patients:                 {kpis['n_patients']}")
    print(f"  Encounters:               {kpis['n_encounters']}")
    print(f"  Average LOS (days):       {kpis['avg_los_days']:.2f}")
    print(f"  30-day readmission rate:  {kpis['readmission_30d_rate_pct']:.2f}%")
    print(f"  Avg medications:          {kpis['avg_medications']:.2f}")
    print(f"  Avg lab procedures:       {kpis['avg_lab_procedures']:.2f}")
    print(f"  Avg procedures:           {kpis['avg_procedures']:.2f}")


if __name__ == "__main__":
    table = prepare_encounter_frame()
    print(f"Encounter table: {table.shape[0]} rows x {table.shape[1]} columns")
    print_kpi_report(table)
