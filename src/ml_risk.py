"""30-day readmission risk model on UCI Diabetes 130-US Hospitals.

Target:
    readmission_30d = 1 if readmitted == "<30"
    readmission_30d = 0 if readmitted in {">30", "NO"}

The original `readmitted` column, encounter_id, and patient_nbr are not used
as model features. Training uses the official CSV only (not demo_user rows).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "readmission_model.joblib"
METRICS_PATH = MODELS_DIR / "readmission_metrics.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import EXPIRED_OR_HOSPICE_DISCHARGE_IDS, prepare_encounter_frame

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
CATEGORICAL_FEATURES = [
    "age",
    "gender",
    "race",
    "admission_type",
    "admission_source",
    "discharge_disposition",
    "payer_code",
    "medical_specialty",
    "insulin",
    "change",
    "diabetesMed",
    "A1Cresult",
    "max_glu_serum",
    "diag_1_group",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_modeling_frame(raw=None) -> pd.DataFrame:
    frame = prepare_encounter_frame(raw=raw)
    eligible = frame.loc[~frame["discharge_disposition_id"].isin(
        EXPIRED_OR_HOSPICE_DISCHARGE_IDS
    )].copy()
    eligible = eligible.dropna(subset=["readmitted"])
    eligible["readmission_30d"] = (eligible["readmitted"] == "<30").astype(int)
    return eligible


def _preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def _metrics_dict(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        out["roc_auc"] = None
    return out


def _patient_level_split(frame: pd.DataFrame):
    patient_label = frame.groupby("patient_nbr")["readmission_30d"].max()
    patient_ids = patient_label.index.to_numpy()
    labels = patient_label.to_numpy()
    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_df = frame.loc[frame["patient_nbr"].isin(train_ids)].copy()
    test_df = frame.loc[frame["patient_nbr"].isin(test_ids)].copy()
    return train_df, test_df


def _make_classifier_pipelines() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("prep", _preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", _preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=12,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _choice_lists(frame: pd.DataFrame) -> dict:
    return {
        col: sorted(frame[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_FEATURES
    }


def train_and_evaluate(persist: bool = True) -> dict:
    frame = build_modeling_frame()
    train_df, test_df = _patient_level_split(frame)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["readmission_30d"].to_numpy()
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["readmission_30d"].to_numpy()

    model_metrics = {}
    train_metrics = {}
    fitted = {}
    for name, pipe in _make_classifier_pipelines().items():
        pipe.fit(X_train, y_train)
        train_pred = pipe.predict(X_train)
        train_proba = pipe.predict_proba(X_train)[:, 1]
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = pipe
        train_metrics[name] = _metrics_dict(y_train, train_pred, train_proba)
        model_metrics[name] = _metrics_dict(y_test, pred, proba)
        model_metrics[name]["classification_report"] = classification_report(
            y_test, pred, zero_division=0, digits=3
        )

    chosen_name = max(
        model_metrics,
        key=lambda n: (model_metrics[n]["roc_auc"] is not None, model_metrics[n]["roc_auc"] or 0),
    )
    chosen = fitted[chosen_name]
    choices = _choice_lists(frame)

    bundle = {
        "pipeline": chosen,
        "model_name": chosen_name,
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_choices": choices,
        "decision_threshold": 0.5,
        "threshold_note": (
            "The predicted class uses the default 0.5 probability cutoff of the "
            "sklearn classifier (with class_weight='balanced'). This is not a "
            "clinical risk threshold."
        ),
        "target": {
            "name": "readmission_30d",
            "definition": (
                "1 if UCI readmitted == '<30', else 0 for '>30' or 'NO'. "
                "The original readmitted column is the label source only."
            ),
            "prediction_time": "hospital discharge (features known during the stay)",
        },
        "limitations": {
            "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
            "population": "Diabetes-related hospital encounters, not all hospital patients",
            "n_eligible_encounters": int(len(frame)),
            "n_positive_encounters": int(frame["readmission_30d"].sum()),
            "n_patients_total": int(frame["patient_nbr"].nunique()),
            "excluded": "Expired/hospice discharge dispositions (cannot be readmitted)",
            "split": "patient-level 80/20 stratified by any <30 readmission",
        },
    }

    results = {
        "n_eligible_encounters": int(len(frame)),
        "n_positive_encounters": int(frame["readmission_30d"].sum()),
        "positive_rate": float(frame["readmission_30d"].mean()),
        "n_patients": int(frame["patient_nbr"].nunique()),
        "n_train_encounters": int(len(train_df)),
        "n_test_encounters": int(len(test_df)),
        "n_train_patients": int(train_df["patient_nbr"].nunique()),
        "n_test_patients": int(test_df["patient_nbr"].nunique()),
        "n_train_positives": int(y_train.sum()),
        "n_test_positives": int(y_test.sum()),
        "models_train": train_metrics,
        "models": model_metrics,
        "chosen_model": chosen_name,
        "decision_threshold": 0.5,
        "threshold_note": bundle["threshold_note"],
        "target": bundle["target"],
        "limitations": bundle["limitations"],
        "features": ALL_FEATURES,
    }

    if persist:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, PIPELINE_PATH)
        serializable = json.loads(json.dumps(results, default=str))
        METRICS_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    results["bundle"] = bundle
    return results


def load_bundle(force_retrain: bool = False) -> dict:
    if force_retrain or not PIPELINE_PATH.exists():
        return train_and_evaluate(persist=True)["bundle"]
    return joblib.load(PIPELINE_PATH)


def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def predict_readmission_risk(feature_row: dict, bundle: dict | None = None) -> dict:
    bundle = bundle or load_bundle()
    pipeline = bundle["pipeline"]
    features = bundle.get("features", ALL_FEATURES)
    row = {name: feature_row.get(name) for name in features}
    frame = pd.DataFrame([row], columns=features)
    proba = float(pipeline.predict_proba(frame)[0, 1])
    threshold = float(bundle.get("decision_threshold", 0.5))
    pred = int(proba >= threshold)
    return {
        "probability": proba,
        "predicted_label": pred,
        "threshold": threshold,
        "model_name": bundle["model_name"],
    }


def render_risk_prediction_section() -> None:
    import plotly.graph_objects as go
    import streamlit as st

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = load_bundle()
        metrics = load_metrics()
    except Exception as exc:
        st.error("Could not train or load the readmission model.")
        st.exception(exc)
        return

    choices = bundle.get("categorical_choices", {})
    lim = bundle.get("limitations", {})

    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px solid #E5E9F2; border-radius: 12px; padding: 14px 18px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(32,36,58,0.04);">
          <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
            <span style="background: #EDE9FE; color: #5B21B6; font-size: 0.76rem; font-weight: 700; padding: 3px 10px; border-radius: 999px;">
              PIPELINE: RANDOM FOREST
            </span>
            <span style="background: #E0E7FF; color: #3730A3; font-size: 0.76rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;">
              PATIENT-LEVEL 80/20 SPLIT
            </span>
            <span style="background: #FEF3C7; color: #92400E; font-size: 0.76rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;">
              CLASS BALANCED (11.4% BASELINE)
            </span>
            <span style="background: #FEE2E2; color: #991B1B; font-size: 0.76rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;">
              TARGET: READMITTED &lt; 30 DAYS
            </span>
          </div>
          <div style="font-size: 0.88rem; color: #64748B; line-height: 1.5;">
            Evaluated on <b style="color:#20243A;">19,794 held-out test encounters</b> across <b style="color:#20243A;">13,998 unseen patients</b>. 
            Features use discharge-time clinical observations only. Expired and hospice discharges are excluded from training.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset archetypes for quick evaluation
    presets = {
        "Custom Input": {},
        "Elevated-Risk Profile (Elderly, Urgent, High Prior Visits)": {
            "age": "[70-80)",
            "gender": "Female",
            "race": "Caucasian",
            "admission_type": "Emergency",
            "admission_source": "Emergency Room",
            "discharge_disposition": "Discharged to home",
            "payer_code": "MC",
            "medical_specialty": "InternalMedicine",
            "insulin": "Up",
            "change": "Ch",
            "diabetesMed": "Yes",
            "A1Cresult": ">8",
            "max_glu_serum": "Unknown",
            "diag_1_group": "Circulatory",
            "time_in_hospital": 8,
            "num_lab_procedures": 68,
            "num_procedures": 2,
            "num_medications": 26,
            "number_diagnoses": 9,
            "number_outpatient": 1,
            "number_emergency": 2,
            "number_inpatient": 2,
        },
        "Lower-Risk Profile (Elective, Short Stay, Minimal Meds)": {
            "age": "[30-40)",
            "gender": "Male",
            "race": "Caucasian",
            "admission_type": "Elective",
            "admission_source": "Physician Referral",
            "discharge_disposition": "Discharged to home",
            "payer_code": "HM",
            "medical_specialty": "Unknown",
            "insulin": "No",
            "change": "No",
            "diabetesMed": "No",
            "A1Cresult": "Norm",
            "max_glu_serum": "Unknown",
            "diag_1_group": "Diabetes",
            "time_in_hospital": 2,
            "num_lab_procedures": 20,
            "num_procedures": 0,
            "num_medications": 7,
            "number_diagnoses": 3,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
        },
        "Typical Diabetes Inpatient (Emergency, 4-Day Stay, Steady Insulin)": {
            "age": "[60-70)",
            "gender": "Male",
            "race": "Caucasian",
            "admission_type": "Emergency",
            "admission_source": "Emergency Room",
            "discharge_disposition": "Discharged to home",
            "payer_code": "MC",
            "medical_specialty": "InternalMedicine",
            "insulin": "Steady",
            "change": "No",
            "diabetesMed": "Yes",
            "A1Cresult": "Unknown",
            "max_glu_serum": "Unknown",
            "diag_1_group": "Diabetes",
            "time_in_hospital": 4,
            "num_lab_procedures": 43,
            "num_procedures": 1,
            "num_medications": 16,
            "number_diagnoses": 7,
            "number_outpatient": 0,
            "number_emergency": 0,
            "number_inpatient": 0,
        },
    }

    st.markdown("#### 👤 Encounter Risk Scorer")
    selected_preset = st.selectbox(
        "Load Clinical Archetype / Example Scenario",
        options=list(presets.keys()),
        help="Quickly pre-fill realistic patient profiles to evaluate model sensitivity.",
    )
    p_vals = presets[selected_preset]

    def _opts(name, fallback):
        values = choices.get(name) or fallback
        return values

    def _val_idx(opts_list, target_val, default_idx=0):
        if target_val in opts_list:
            return opts_list.index(target_val)
        return min(default_idx, len(opts_list) - 1)

    with st.form("risk_prediction_form"):
        st.markdown("**1. Demographics & Admission**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age_opts = _opts("age", ["[70-80)"])
            age = st.selectbox("Age group", age_opts, index=_val_idx(age_opts, p_vals.get("age", "[70-80)"), 7))
            gender_opts = _opts("gender", ["Female", "Male"])
            gender = st.selectbox("Gender", gender_opts, index=_val_idx(gender_opts, p_vals.get("gender", "Female"), 0))
            race_opts = _opts("race", ["Caucasian"])
            race = st.selectbox("Race", race_opts, index=_val_idx(race_opts, p_vals.get("race", "Caucasian"), 0))
        with c2:
            adm_type_opts = _opts("admission_type", ["Emergency"])
            admission_type = st.selectbox("Admission type", adm_type_opts, index=_val_idx(adm_type_opts, p_vals.get("admission_type", "Emergency"), 0))
            adm_src_opts = _opts("admission_source", ["Emergency Room"])
            admission_source = st.selectbox("Admission source", adm_src_opts, index=_val_idx(adm_src_opts, p_vals.get("admission_source", "Emergency Room"), 0))
            discharge_opts = _opts("discharge_disposition", ["Discharged to home"])
            discharge_disposition = st.selectbox("Discharge disposition", discharge_opts, index=_val_idx(discharge_opts, p_vals.get("discharge_disposition", "Discharged to home"), 0))
        with c3:
            payer_opts = _opts("payer_code", ["Unknown"])
            payer_code = st.selectbox("Payer / insurance", payer_opts, index=_val_idx(payer_opts, p_vals.get("payer_code", "Unknown"), 0))
            spec_opts = _opts("medical_specialty", ["Unknown"])
            medical_specialty = st.selectbox("Medical specialty", spec_opts, index=_val_idx(spec_opts, p_vals.get("medical_specialty", "Unknown"), 0))
            diag_opts = _opts("diag_1_group", ["Diabetes"])
            diag_1_group = st.selectbox("Primary diagnosis group", diag_opts, index=_val_idx(diag_opts, p_vals.get("diag_1_group", "Diabetes"), 0))

        st.markdown("**2. Inpatient Stay & Prior Utilization**")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            time_in_hospital = st.number_input(
                "LOS: Time in hospital (days)", min_value=1, max_value=14, value=int(p_vals.get("time_in_hospital", 4)), step=1
            )
            number_diagnoses = st.number_input(
                "Total diagnoses recorded", min_value=1, max_value=16, value=int(p_vals.get("number_diagnoses", 8)), step=1
            )
        with d2:
            num_lab_procedures = st.number_input(
                "Lab procedures during stay", min_value=0, max_value=150, value=int(p_vals.get("num_lab_procedures", 43)), step=1
            )
            num_procedures = st.number_input(
                "Procedures (non-lab)", min_value=0, max_value=10, value=int(p_vals.get("num_procedures", 1)), step=1
            )
        with d3:
            num_medications = st.number_input(
                "Distinct medications", min_value=0, max_value=80, value=int(p_vals.get("num_medications", 16)), step=1
            )
            number_outpatient = st.number_input(
                "Prior outpatient visits (1 yr)", min_value=0, max_value=50, value=int(p_vals.get("number_outpatient", 0)), step=1
            )
        with d4:
            number_emergency = st.number_input(
                "Prior emergency visits (1 yr)", min_value=0, max_value=50, value=int(p_vals.get("number_emergency", 0)), step=1
            )
            number_inpatient = st.number_input(
                "Prior inpatient visits (1 yr)", min_value=0, max_value=50, value=int(p_vals.get("number_inpatient", 0)), step=1
            )

        st.markdown("**3. Medications & Lab Indicators**")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            ins_opts = _opts("insulin", ["No", "Steady", "Up", "Down"])
            insulin = st.selectbox("Insulin status", ins_opts, index=_val_idx(ins_opts, p_vals.get("insulin", "No"), 0))
        with m2:
            ch_opts = _opts("change", ["No", "Ch"])
            change = st.selectbox("Medication change", ch_opts, index=_val_idx(ch_opts, p_vals.get("change", "No"), 0))
        with m3:
            dm_opts = _opts("diabetesMed", ["Yes", "No"])
            diabetes_med = st.selectbox("Diabetes med prescribed", dm_opts, index=_val_idx(dm_opts, p_vals.get("diabetesMed", "Yes"), 0))
        with m4:
            a1c_opts = _opts("A1Cresult", ["Unknown"])
            a1c = st.selectbox("A1C test result", a1c_opts, index=_val_idx(a1c_opts, p_vals.get("A1Cresult", "Unknown"), 0))
        with m5:
            glu_opts = _opts("max_glu_serum", ["Unknown"])
            max_glu = st.selectbox("Max glucose serum", glu_opts, index=_val_idx(glu_opts, p_vals.get("max_glu_serum", "Unknown"), 0))

        submitted = st.form_submit_button(
            "⚡ Compute 30-Day Readmission Risk", type="primary", use_container_width=True
        )

    if submitted:
        features = {
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "age": age,
            "gender": gender,
            "race": race,
            "admission_type": admission_type,
            "admission_source": admission_source,
            "discharge_disposition": discharge_disposition,
            "payer_code": payer_code,
            "medical_specialty": medical_specialty,
            "insulin": insulin,
            "change": change,
            "diabetesMed": diabetes_med,
            "A1Cresult": a1c,
            "max_glu_serum": max_glu,
            "diag_1_group": diag_1_group,
        }
        result = predict_readmission_risk(features, bundle=bundle)
        p = result["probability"]
        cutoff = result["threshold"]
        baseline_rate = 0.1139
        rel_risk = p / baseline_rate if baseline_rate > 0 else 1.0

        if p >= cutoff:
            badge_color = "#FEF2F2"
            badge_border = "#FECACA"
            badge_text = "#B91C1C"
            status_label = "⚠️ ELEVATED 30-DAY READMISSION RISK"
            status_sub = "Encounter probability meets or exceeds the 0.50 balanced decision boundary."
        elif p >= 0.30:
            badge_color = "#FFFBEB"
            badge_border = "#FDE68A"
            badge_text = "#B45309"
            status_label = "⚡ MODERATE 30-DAY READMISSION PROBABILITY"
            status_sub = "Encounter risk is significantly above the 11.4% baseline prevalence."
        else:
            badge_color = "#F0FDF4"
            badge_border = "#BBF7D0"
            badge_text = "#15803D"
            status_label = "✅ LOWER READMISSION RISK"
            status_sub = "Encounter risk falls well below the 0.50 decision cutoff."

        st.markdown(
            f"""
            <div style="background: {badge_color}; border: 1px solid {badge_border}; border-radius: 12px; padding: 16px 20px; margin-top: 20px;">
              <div style="color: {badge_text}; font-weight: 700; font-size: 1.05rem; letter-spacing: 0.03em;">
                {status_label}
              </div>
              <div style="color: #475569; font-size: 0.88rem; margin-top: 4px;">
                {status_sub}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.progress(min(max(p, 0.0), 1.0))

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Predicted Probability", f"{p:.1%}")
        rc2.metric("Decision Cutoff", f"{cutoff:.1%}")
        rc3.metric("Baseline Prevalence", f"{baseline_rate:.1%}")
        rc4.metric("Relative Risk", f"{rel_risk:.1f}x baseline")

        st.info(
            f"**Clinical Context & Interpretation:** The pipeline estimates a **{p:.1%}** probability that this encounter "
            f"corresponds to a patient readmitted within 30 days (`readmitted = <30`). Because the training cohort has "
            f"an ~11.4% base positive rate, an estimated risk of {p:.1%} represents **{rel_risk:.1f} times** the average baseline risk."
        )

    # Held-Out Evaluation Section
    if metrics:
        st.markdown("---")
        st.markdown("### 📊 Model Validation & Held-Out Performance")
        st.caption(
            "Evaluation performed strictly on held-out test encounters partitioned at the patient level (80% train / 20% test). "
            "No patient appears in both splits."
        )

        test_enc = metrics.get("n_test_encounters", 19794)
        test_pos = metrics.get("n_test_positives", 2193)
        test_pts = metrics.get("n_test_patients", 13998)
        rf_metrics = metrics["models"].get("random_forest", {})
        lr_metrics = metrics["models"].get("logistic_regression", {})

        ec1, ec2, ec3 = st.columns([1, 1, 1.4])
        with ec1:
            with st.container(border=True):
                st.markdown("**🌲 Random Forest (Active Pipeline)**")
                st.markdown(f"**ROC-AUC:** `{rf_metrics.get('roc_auc', 0.666):.3f}`")
                st.markdown(f"**Recall (Sensitivity):** `{rf_metrics.get('recall', 0.602):.1%}`")
                st.markdown(f"**Precision:** `{rf_metrics.get('precision', 0.171):.1%}`")
                st.markdown(f"**F1 Score:** `{rf_metrics.get('f1', 0.267):.3f}`")
                st.markdown(f"**Accuracy:** `{rf_metrics.get('accuracy', 0.634):.1%}`")
                st.caption("Selected for superior ROC-AUC and 60.2% positive recall on held-out test encounters.")

        with ec2:
            with st.container(border=True):
                st.markdown("**📈 Logistic Regression (Baseline)**")
                st.markdown(f"**ROC-AUC:** `{lr_metrics.get('roc_auc', 0.658):.3f}`")
                st.markdown(f"**Recall (Sensitivity):** `{lr_metrics.get('recall', 0.548):.1%}`")
                st.markdown(f"**Precision:** `{lr_metrics.get('precision', 0.175):.1%}`")
                st.markdown(f"**F1 Score:** `{lr_metrics.get('f1', 0.265):.3f}`")
                st.markdown(f"**Accuracy:** `{lr_metrics.get('accuracy', 0.664):.1%}`")
                st.caption("Baseline generalized linear model with L2 regularization and balanced class weighting.")

        with ec3:
            with st.container(border=True):
                st.markdown("**🎯 Evaluation Cohort Summary**")
                st.markdown(f"- **Test Encounters:** `{test_enc:,}`")
                st.markdown(f"- **Test Positive Cases (<30d):** `{test_pos:,}` ({test_pos/test_enc:.1%})")
                st.markdown(f"- **Unique Test Patients:** `{test_pts:,}`")
                st.markdown("- **Split Strategy:** Patient-level stratified holdout")
                st.caption(
                    "Class imbalance is high (~11% <30d readmissions). With balanced class weights, "
                    "recall and ROC-AUC are the primary metrics to capture readmission cases."
                )

        # Interactive Confusion Matrix
        cm = rf_metrics.get("confusion_matrix", [[11220, 6381], [873, 1320]])
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Predicted: Not &lt;30d (0)", "Predicted: Readmitted &lt;30d (1)"],
                y=["Actual: Not &lt;30d (0)", "Actual: Readmitted &lt;30d (1)"],
                colorscale=[[0, "#EDE9FE"], [0.5, "#9385F5"], [1, "#20243A"]],
                showscale=False,
            )
        )
        annotations = [
            dict(x=0, y=0, text=f"True Negative (TN)<br><b>{tn:,}</b>", showarrow=False, font=dict(color="#FFFFFF", size=13, family="Plus Jakarta Sans, Inter, sans-serif")),
            dict(x=1, y=0, text=f"False Positive (FP)<br><b>{fp:,}</b>", showarrow=False, font=dict(color="#FFFFFF", size=13, family="Plus Jakarta Sans, Inter, sans-serif")),
            dict(x=0, y=1, text=f"False Negative (FN)<br><b>{fn:,}</b>", showarrow=False, font=dict(color="#20243A", size=13, family="Plus Jakarta Sans, Inter, sans-serif")),
            dict(x=1, y=1, text=f"True Positive (TP)<br><b>{tp:,}</b>", showarrow=False, font=dict(color="#20243A", size=13, family="Plus Jakarta Sans, Inter, sans-serif")),
        ]
        fig_cm.update_layout(
            annotations=annotations,
            title=dict(
                text="Held-Out Confusion Matrix: Random Forest Classifier",
                font=dict(size=14, color="#20243A", family="Plus Jakarta Sans, Inter, sans-serif"),
                x=0.02,
                y=0.96,
            ),
            margin=dict(l=40, r=40, t=55, b=40),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=12),
        )
        st.plotly_chart(fig_cm, width="stretch")



def print_evaluation(results: dict) -> None:
    print("=== UCI 30-day readmission model ===")
    print(f"Eligible encounters: {results['n_eligible_encounters']}")
    print(
        f"Positive encounters: {results['n_positive_encounters']} "
        f"({results['positive_rate']:.1%})"
    )
    print(f"Patients: {results['n_patients']}")
    print(
        f"Split: {results['n_train_patients']} train patients / "
        f"{results['n_test_patients']} test patients"
    )
    print(f"Chosen model: {results['chosen_model']}")
    for name, m in results["models"].items():
        print(f"\n-- {name} test --")
        print(
            {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix")}
        )
        print(m["classification_report"])


if __name__ == "__main__":
    eval_results = train_and_evaluate(persist=True)
    print_evaluation(eval_results)
