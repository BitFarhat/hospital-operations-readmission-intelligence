"""Command-line analytics report over the UCI SQLite database."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sql_analytics import (
    diagnosis_group_counts,
    encounters_by_admission_type,
    encounters_by_age,
    encounters_by_gender,
    get_basic_kpis,
    medication_utilization_by_readmission,
    readmission_outcome_distribution,
    readmission_rate_by_age,
)


def run_report() -> None:
    kpis = get_basic_kpis().iloc[0]
    print("Hospital Operations & Readmission Intelligence")
    print("Dataset: UCI Diabetes 130-US Hospitals (1999-2008)")
    print(f"  Patients:                {int(kpis['total_patients']):,}")
    print(f"  Encounters:              {int(kpis['total_encounters']):,}")
    print(f"  Average LOS (days):      {kpis['average_los_days']}")
    print(f"  30-day readmission rate: {kpis['readmission_30d_rate_pct']}%")
    print(f"  Avg medications:         {kpis['avg_medications']}")
    print(f"  Avg lab procedures:      {kpis['avg_lab_procedures']}")
    print(f"  Avg procedures:          {kpis['avg_procedures']}")
    print("\nEncounters by gender")
    print(encounters_by_gender().to_string(index=False))
    print("\nEncounters by age")
    print(encounters_by_age().to_string(index=False))
    print("\nEncounters by admission type")
    print(encounters_by_admission_type().to_string(index=False))
    print("\nReadmission outcome")
    print(readmission_outcome_distribution().to_string(index=False))
    print("\n30-day readmission rate by age")
    print(readmission_rate_by_age().to_string(index=False))
    print("\nUtilization by readmission outcome")
    print(medication_utilization_by_readmission().to_string(index=False))
    print("\nPrimary diagnosis groups")
    print(diagnosis_group_counts().to_string(index=False))


if __name__ == "__main__":
    run_report()
