"""Hospital Operations & Readmission Intelligence dashboard.

Run from the project root:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from admin import add_patient_form, admin_login, admin_logout_button
from data import DataLoadError
from ml_risk import render_risk_prediction_section
from sql_analytics import (
    diagnosis_group_counts,
    encounters_by_admission_source,
    encounters_by_admission_type,
    encounters_by_age,
    encounters_by_gender,
    encounters_by_race,
    get_basic_kpis,
    get_filter_options,
    insulin_distribution,
    length_of_stay_distribution,
    medication_utilization_by_readmission,
    preview_encounters,
    procedure_utilization_by_age,
    readmission_outcome_distribution,
    readmission_rate_by_admission_type,
    readmission_rate_by_age,
    readmission_rate_by_diagnosis_group,
)

PAGE_OVERVIEW = "Overview"
PAGE_OPERATIONS = "Operations"
PAGE_READMISSION = "Readmission"
PAGE_ENCOUNTERS = "Encounters"
PAGE_RISK = "Risk model"
PAGE_ADD = "Add encounter"

NAV_PAGES = [
    PAGE_OVERVIEW,
    PAGE_OPERATIONS,
    PAGE_READMISSION,
    PAGE_ENCOUNTERS,
    PAGE_RISK,
    PAGE_ADD,
]

PAGE_LABELS = {
    PAGE_OVERVIEW: "📊 Executive Overview",
    PAGE_OPERATIONS: "🏥 Operations & Census",
    PAGE_READMISSION: "🔄 Readmission Analytics",
    PAGE_ENCOUNTERS: "🔍 Encounter Explorer",
    PAGE_RISK: "🤖 30-Day Risk Model",
    PAGE_ADD: "➕ Add Demo Encounter",
}

# Modern SaaS Analytics Palette: Charcoal Navy, Lavender, Muted Blue, and Clean Lights
CHARCOAL = "#20243A"
LAVENDER = "#9385F5"
LAVENDER_DARK = "#7B6AF0"
LAVENDER_LIGHT = "#EDE9FE"
MUTED_BLUE = "#4880FF"
SLATE = "#64748B"
SLATE_LIGHT = "#94A3B8"
ROSE = "#F43F5E"
AMBER = "#F59E0B"
EMERALD = "#10B981"
BG_LIGHT = "#F8F9FD"

COLORWAY = [
    LAVENDER,
    MUTED_BLUE,
    "#38BDF8",
    EMERALD,
    AMBER,
    ROSE,
    "#818CF8",
    "#C084FC",
]

st.set_page_config(
    page_title="Hospital Intelligence | Modern SaaS Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap");

      html, body, [class*="css"] {
        font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .stApp {
        background-color: #F8F9FD;
      }
      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.8rem;
        max-width: 1480px;
      }

      /* Sidebar: Charcoal Navy Theme */
      [data-testid="stSidebar"] {
        background-color: #20243A !important;
        border-right: 1px solid #2C324D;
      }
      [data-testid="stSidebar"] * {
        color: #E2E8F0;
      }
      [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p {
        color: #A0AEC0;
      }
      [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08);
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #FFFFFF !important;
      }
      [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div,
      [data-testid="stSidebar"] input {
        background: #181B2C !important;
        color: #F8FAFC !important;
        border-color: #343A59 !important;
        border-radius: 8px !important;
      }
      /* Sidebar MultiSelect Placeholder Text Visibility */
      [data-testid="stSidebar"] [data-testid="stMultiSelect"] input::placeholder,
      [data-testid="stSidebar"] [data-testid="stMultiSelect"] input::-webkit-input-placeholder,
      [data-testid="stSidebar"] [data-testid="stMultiSelect"] input::-moz-placeholder,
      [data-testid="stSidebar"] [data-testid="stMultiSelect"] input:-ms-input-placeholder,
      [data-testid="stSidebar"] .stMultiSelect input::placeholder,
      [data-testid="stSidebar"] .stMultiSelect input::-webkit-input-placeholder,
      [data-testid="stSidebar"] .stMultiSelect input::-moz-placeholder,
      [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] input::placeholder,
      [data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] input::-webkit-input-placeholder,
      [data-testid="stSidebar"] div[data-baseweb="select"] input::placeholder,
      [data-testid="stSidebar"] div[data-baseweb="select"] input::-webkit-input-placeholder,
      [data-testid="stSidebar"] div[data-baseweb="select"] [class*="placeholder"],
      [data-testid="stSidebar"] div[data-baseweb="select"] span[class*="placeholder"],
      [data-testid="stSidebar"] .react-aria-Input::placeholder,
      [data-testid="stSidebar"] .react-aria-Input::-webkit-input-placeholder {
        color: #CBD5E1 !important;
        -webkit-text-fill-color: #CBD5E1 !important;
        opacity: 1 !important;
      }
      [data-testid="stSidebar"] button {
        background: #2D3352 !important;
        color: #EDE9FE !important;
        border: 1px solid #3F4770 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease;
      }
      [data-testid="stSidebar"] button:hover {
        background: #9385F5 !important;
        color: #FFFFFF !important;
        border-color: #9385F5 !important;
      }

      /* Header Elements */
      .page-kicker {
        color: #9385F5;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
      }
      .page-title {
        color: #20243A;
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.25rem 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
      }
      .page-sub {
        color: #64748B;
        font-size: 0.94rem;
        margin-bottom: 1.25rem;
      }
      .section-title {
        color: #20243A;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 1.2rem 0 0.65rem 0;
        letter-spacing: -0.01em;
      }

      /* Modern SaaS KPI Cards */
      .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E5E9F2;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(32, 36, 58, 0.04), 0 4px 12px rgba(32, 36, 58, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
      }
      .kpi-card:hover {
        box-shadow: 0 4px 16px rgba(147, 133, 245, 0.1);
        border-color: #D4DAE8;
      }
      .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
      }
      .kpi-title {
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .kpi-icon-pill {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
      }
      .kpi-value {
        color: #20243A;
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
      }
      .kpi-subtitle {
        color: #64748B;
        font-size: 0.76rem;
        margin: 0;
      }

      /* Modern Tabs */
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E5E9F2;
        margin-bottom: 1rem;
      }
      .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #64748B;
      }
      .stTabs [aria-selected="true"] {
        color: #9385F5 !important;
        border-bottom: 2px solid #9385F5 !important;
      }

      /* Forms and DataFrames */
      .stForm {
        background: #FFFFFF;
        border: 1px solid #E5E9F2;
        border-radius: 14px;
        padding: 1.1rem 1.3rem 1.4rem 1.3rem;
        box-shadow: 0 1px 3px rgba(32, 36, 58, 0.04);
      }
      .stDataFrame, [data-testid="stDataFrame"] {
        border: 1px solid #E5E9F2;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(32, 36, 58, 0.04);
      }
      .stAlert {
        border-radius: 12px;
      }
      header[data-testid="stHeader"] {
        background: rgba(248, 249, 253, 0.85);
      }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading filter options from SQLite...")
def cached_filter_options() -> dict:
    return get_filter_options()


def _base_layout(title: str = "", height: int = 360) -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=CHARCOAL, family="Plus Jakarta Sans, Inter, sans-serif"),
            x=0.02,
            y=0.96,
        ),
        margin=dict(l=24, r=20, t=55, b=65),
        height=height,
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#334155", size=11),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        colorway=COLORWAY,
        showlegend=False,
        bargap=0.25,
        hoverlabel=dict(
            bgcolor=CHARCOAL,
            font_size=12,
            font_family="Plus Jakarta Sans, Inter, sans-serif",
            font_color="#FFFFFF",
        ),
    )


def bar_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str = "Encounters",
    color: str | None = None,
    color_map: dict | None = None,
    color_col: str | None = None,
    height: int = 360,
) -> None:
    if color_map and color_col:
        fig = px.bar(
            frame,
            x=x,
            y=y,
            color=color_col,
            color_discrete_map=color_map,
            title=title,
        )
    else:
        fig = px.bar(
            frame,
            x=x,
            y=y,
            color_discrete_sequence=[color or LAVENDER],
            title=title,
        )

    layout = _base_layout(title, height=height)
    fig.update_layout(**layout)
    fig.update_traces(marker_line_width=0, hovertemplate="%{x}: <b>%{y:,}</b><extra></extra>")
    fig.update_xaxes(
        showgrid=False,
        tickangle=-25,
        linecolor="#E2E8F0",
        ticks="outside",
        tickfont=dict(size=11, color="#64748B"),
        title="",
    )
    fig.update_yaxes(
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        zeroline=False,
        tickfont=dict(size=11, color="#64748B"),
        title=ylabel,
    )
    st.plotly_chart(fig, width="stretch")


def horizontal_bar_chart(
    frame: pd.DataFrame,
    y: str,
    x: str,
    title: str,
    xlabel: str = "Encounters",
    color: str | None = None,
    height: int = 380,
    is_rate: bool = False,
) -> None:
    # Sort ascending so largest value displays at the top
    df_sorted = frame.sort_values(x, ascending=True)
    fig = px.bar(
        df_sorted,
        y=y,
        x=x,
        orientation="h",
        color_discrete_sequence=[color or MUTED_BLUE],
        title=title,
    )
    layout = _base_layout(title, height=height)
    layout["margin"] = dict(l=175, r=25, t=55, b=50)
    fig.update_layout(**layout)
    if is_rate:
        fig.update_traces(marker_line_width=0, hovertemplate="%{y}: <b>%{x:.2f}%</b><extra></extra>")
    else:
        fig.update_traces(marker_line_width=0, hovertemplate="%{y}: <b>%{x:,}</b><extra></extra>")
    fig.update_yaxes(
        showgrid=False,
        linecolor="#E2E8F0",
        tickfont=dict(size=11, color="#475569"),
        title="",
    )
    fig.update_xaxes(
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        zeroline=False,
        tickfont=dict(size=11, color="#64748B"),
        title=xlabel,
    )
    st.plotly_chart(fig, width="stretch")


def donut_chart(
    frame: pd.DataFrame,
    names: str,
    values: str,
    title: str,
    height: int = 360,
) -> None:
    fig = px.pie(
        frame,
        names=names,
        values=values,
        color_discrete_sequence=COLORWAY,
        hole=0.54,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        textfont=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans, Inter"),
        hovertemplate="<b>%{label}</b><br>Encounters: %{value:,}<br>Share: %{percent}<extra></extra>",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color=CHARCOAL, family="Plus Jakarta Sans, Inter, sans-serif"),
            x=0.02,
            y=0.96,
        ),
        margin=dict(l=20, r=20, t=55, b=45),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            x=0.5,
            xanchor="center",
            font=dict(size=11, color="#64748B", family="Plus Jakarta Sans, Inter, sans-serif"),
        ),
    )
    st.plotly_chart(fig, width="stretch")


def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-kicker">{kicker}</div>'
        f'<div class="page-title">{title}</div>'
        f'<p class="page-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def render_kpi_row(kpis: pd.Series) -> None:
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid {LAVENDER};">
              <div class="kpi-header">
                <span class="kpi-title">Total Patients</span>
                <span class="kpi-icon-pill" style="background: rgba(147, 133, 245, 0.12); color: {LAVENDER_DARK};">👥</span>
              </div>
              <div class="kpi-value">{int(kpis['total_patients']):,}</div>
              <div class="kpi-subtitle">Unique patient cohort</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r1c2:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid {MUTED_BLUE};">
              <div class="kpi-header">
                <span class="kpi-title">Total Encounters</span>
                <span class="kpi-icon-pill" style="background: rgba(72, 128, 255, 0.12); color: {MUTED_BLUE};">📋</span>
              </div>
              <div class="kpi-value">{int(kpis['total_encounters']):,}</div>
              <div class="kpi-subtitle">Inpatient admissions in scope</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r1c3:
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid {EMERALD};">
              <div class="kpi-header">
                <span class="kpi-title">Avg Length of Stay</span>
                <span class="kpi-icon-pill" style="background: rgba(16, 185, 129, 0.12); color: {EMERALD};">⏱️</span>
              </div>
              <div class="kpi-value">{float(kpis['average_los_days']):.1f} <span style="font-size:1rem; font-weight:500; color:{SLATE};">days</span></div>
              <div class="kpi-subtitle">time_in_hospital (1–14d)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r1c4:
        rate = float(kpis['readmission_30d_rate_pct'])
        rate_col = ROSE if rate >= 12.0 else LAVENDER
        st.markdown(
            f"""
            <div class="kpi-card" style="border-top: 3px solid {rate_col};">
              <div class="kpi-header">
                <span class="kpi-title">30-Day Readmission</span>
                <span class="kpi-icon-pill" style="background: rgba(244, 63, 94, 0.12); color: {ROSE};">🔄</span>
              </div>
              <div class="kpi-value" style="color: {rate_col};">{rate:.2f}%</div>
              <div class="kpi-subtitle">Early readmission rate (&lt;30d)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-header">
                <span class="kpi-title">Avg Medications</span>
                <span class="kpi-icon-pill" style="background: rgba(147, 133, 245, 0.1); color: {LAVENDER_DARK};">💊</span>
              </div>
              <div class="kpi-value">{float(kpis['avg_medications']):.1f}</div>
              <div class="kpi-subtitle">Prescribed meds per stay</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2c2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-header">
                <span class="kpi-title">Avg Lab Procedures</span>
                <span class="kpi-icon-pill" style="background: rgba(72, 128, 255, 0.1); color: {MUTED_BLUE};">🔬</span>
              </div>
              <div class="kpi-value">{float(kpis['avg_lab_procedures']):.1f}</div>
              <div class="kpi-subtitle">Diagnostic lab tests performed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with r2c3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-header">
                <span class="kpi-title">Avg Procedures</span>
                <span class="kpi-icon-pill" style="background: rgba(56, 189, 248, 0.1); color: #0284C7;">🩺</span>
              </div>
              <div class="kpi-value">{float(kpis['avg_procedures']):.1f}</div>
              <div class="kpi-subtitle">Clinical/operative procedures</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")


def render_overview_charts(filters: dict) -> None:
    st.markdown('<div class="section-title">Operational Census Snapshot</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        bar_chart(
            encounters_by_admission_type(**filters),
            "admission_type",
            "encounter_count",
            "Encounters by Admission Type",
            ylabel="Encounters",
            color=LAVENDER,
        )
    with c2:
        # Donut Chart with dedicated margins and non-overlapping bottom legend
        gender_df = encounters_by_gender(**filters)
        donut_chart(
            gender_df,
            "gender",
            "encounter_count",
            "Encounters by Patient Gender",
            height=360,
        )

    c3, c4 = st.columns(2)
    with c3:
        outcome = readmission_outcome_distribution(**filters)
        color_map = {"<30": ROSE, ">30": MUTED_BLUE, "NO": LAVENDER}
        bar_chart(
            outcome,
            "readmitted",
            "encounter_count",
            "Readmission Breakdown (<30 vs >30 vs None)",
            ylabel="Encounters",
            color_col="readmitted",
            color_map=color_map,
        )
        st.caption(
            "<b>Clinical interpretation:</b> <code>&lt;30</code> indicates early unplanned readmission (11.16% baseline rate). "
            "<code>&gt;30</code> indicates readmission after 30 days. <code>NO</code> indicates no recorded readmission.",
            unsafe_allow_html=True,
        )
    with c4:
        bar_chart(
            encounters_by_age(**filters),
            "age_group",
            "encounter_count",
            "Encounters by Age Cohort",
            ylabel="Encounters",
            color=MUTED_BLUE,
        )


def render_operations_charts(filters: dict) -> None:
    tab1, tab2, tab3 = st.tabs([
        "📋 Admissions & Demographics",
        "🔬 Clinical Utilization & LOS",
        "🩺 Diagnostic Chapters (ICD-9)",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            bar_chart(
                encounters_by_admission_type(**filters),
                "admission_type",
                "encounter_count",
                "Encounters by Admission Type",
                ylabel="Encounters",
                color=LAVENDER,
            )
        with c2:
            # Horizontal bar chart prevents clipped facility names
            source = encounters_by_admission_source(**filters).head(12)
            horizontal_bar_chart(
                source,
                y="admission_source",
                x="encounter_count",
                title="Top 12 Admission Sources",
                xlabel="Encounters",
                color=MUTED_BLUE,
                height=360,
            )

        c3, c4 = st.columns(2)
        with c3:
            gender_df = encounters_by_gender(**filters)
            donut_chart(
                gender_df,
                "gender",
                "encounter_count",
                "Gender Distribution",
                height=360,
            )
        with c4:
            bar_chart(
                encounters_by_race(**filters),
                "race",
                "encounter_count",
                "Encounters by Race / Ethnicity",
                ylabel="Encounters",
                color=LAVENDER_DARK,
            )

        st.markdown("---")
        bar_chart(
            encounters_by_age(**filters),
            "age_group",
            "encounter_count",
            "Age Distribution Across Inpatient Cohort",
            ylabel="Encounters",
            color=MUTED_BLUE,
        )

    with tab2:
        c5, c6 = st.columns(2)
        with c5:
            los = length_of_stay_distribution(**filters)
            bar_chart(
                los,
                "length_of_stay",
                "encounter_count",
                "Length of Stay Distribution (time_in_hospital: 1–14 days)",
                ylabel="Encounters",
                color=LAVENDER,
            )
        with c6:
            bar_chart(
                insulin_distribution(**filters),
                "insulin",
                "encounter_count",
                "Insulin Regimen During Encounter",
                ylabel="Encounters",
                color=AMBER,
            )

        st.markdown('<div class="section-title">Age-Stratified Clinical Utilization</div>', unsafe_allow_html=True)
        util = procedure_utilization_by_age(**filters)
        c7, c8 = st.columns(2)
        with c7:
            bar_chart(
                util,
                "age_group",
                "avg_procedures",
                "Average Clinical / Operative Procedures by Age",
                ylabel="Avg Procedures",
                color=LAVENDER,
            )
        with c8:
            bar_chart(
                util,
                "age_group",
                "avg_lab_procedures",
                "Average Diagnostic Laboratory Procedures by Age",
                ylabel="Avg Lab Tests",
                color=MUTED_BLUE,
            )

    with tab3:
        dx = diagnosis_group_counts(**filters).head(15)
        horizontal_bar_chart(
            dx,
            y="diag_1_group",
            x="encounter_count",
            title="Primary Diagnosis Groups (Top 15 ICD-9 Chapters)",
            xlabel="Encounters",
            color=LAVENDER,
            height=440,
        )
        st.caption(
            "ICD-9 diagnosis groupings follow the Strack et al. (2014) epidemiological taxonomy for the UCI Diabetes cohort. "
            "Circulatory and respiratory diseases represent the two largest primary admission reasons."
        )


def render_readmission_charts(filters: dict) -> None:
    c1, c2 = st.columns(2)
    with c1:
        outcome = readmission_outcome_distribution(**filters)
        color_map = {"<30": ROSE, ">30": MUTED_BLUE, "NO": LAVENDER}
        bar_chart(
            outcome,
            "readmitted",
            "encounter_count",
            "Readmission Breakdown (<30 vs >30 vs None)",
            ylabel="Encounters",
            color_col="readmitted",
            color_map=color_map,
        )
    with c2:
        bar_chart(
            readmission_rate_by_age(**filters),
            "age_group",
            "readmission_30d_rate_pct",
            "30-Day Readmission Rate by Age Cohort (%)",
            ylabel="Readmission Rate (%)",
            color=ROSE,
        )

    c3, c4 = st.columns(2)
    with c3:
        bar_chart(
            readmission_rate_by_admission_type(**filters),
            "admission_type",
            "readmission_30d_rate_pct",
            "30-Day Readmission Rate by Admission Type (%)",
            ylabel="Readmission Rate (%)",
            color=ROSE,
        )
    with c4:
        meds = medication_utilization_by_readmission(**filters)
        bar_chart(
            meds,
            "readmitted",
            "avg_medications",
            "Medication Intensity by Readmission Outcome",
            ylabel="Avg Medications",
            color=LAVENDER,
        )

    st.markdown('<div class="section-title">Readmission Risk by Primary Diagnosis Chapter</div>', unsafe_allow_html=True)
    dx_risk = readmission_rate_by_diagnosis_group(min_encounters=100, **filters).head(12)
    horizontal_bar_chart(
        dx_risk,
        y="diag_1_group",
        x="readmission_30d_rate_pct",
        title="30-Day Readmission Rate by Primary Diagnosis Group (Min. 100 Encounters)",
        xlabel="30-Day Rate (%)",
        color=ROSE,
        height=380,
        is_rate=True,
    )
    st.info(
        "**Hospital Readmissions Reduction Program (HRRP) Context:** Under CMS guidelines, 30-day unplanned readmissions "
        "serve as a core hospital quality metric. Patients readmitted within 30 days present with higher medication counts "
        "(16.9 vs 15.7 avg) and diagnostic procedure intensity."
    )


def render_encounters_table(filters: dict, kpis: pd.Series) -> None:
    st.markdown('<div class="section-title">Patient Encounter Data Explorer</div>', unsafe_allow_html=True)

    c_search, c_limit = st.columns([3, 1])
    with c_search:
        search_query = st.text_input(
            "🔍 Search Encounters",
            placeholder="Type encounter ID or patient ID (e.g., 100)...",
            help="Filter rows by partial or exact match on encounter_id or patient_nbr.",
        )
    with c_limit:
        row_limit = st.selectbox("Max rows to display", [100, 250, 500, 1000], index=1)

    preview = preview_encounters(limit=row_limit, search_query=search_query or None, **filters)

    if search_query and search_query.strip():
        clean_q = search_query.strip()
        if clean_q.isdigit() and not preview.empty:
            patient_matches = preview["patient_nbr"].astype(str) == clean_q
            if patient_matches.any():
                preview = preview[patient_matches]
            else:
                encounter_matches = preview["encounter_id"].astype(str) == clean_q
                preview = preview[encounter_matches]

    # Table KPI Header
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching Encounters", f"{len(preview):,}")
    m2.metric("Unique Patients in View", f"{preview['patient_nbr'].nunique():,}" if not preview.empty else "0")
    m3.metric("Average LOS (View)", f"{preview['time_in_hospital'].mean():.1f} days" if not preview.empty else "0")
    readmit_count = int((preview["readmitted"] == "<30").sum()) if not preview.empty else 0
    m4.metric("30-Day Readmissions in View", f"{readmit_count:,}")

    # CSV Download
    if not preview.empty:
        csv_data = preview.to_csv(index=False)
        st.download_button(
            label="📥 Export Current View to CSV",
            data=csv_data,
            file_name="hospital_encounters_export.csv",
            mime="text/csv",
        )

    st.dataframe(
        preview,
        width="stretch",
        hide_index=True,
        height=440,
        column_config={
            "encounter_id": st.column_config.NumberColumn("Encounter ID", format="%d"),
            "patient_nbr": st.column_config.NumberColumn("Patient ID", format="%d"),
            "gender": "Gender",
            "age": "Age Cohort",
            "race": "Race",
            "admission_type": "Admission Type",
            "admission_source": "Admission Source",
            "payer_code": "Payer",
            "medical_specialty": "Specialty",
            "time_in_hospital": st.column_config.NumberColumn("LOS (Days)", format="%d"),
            "num_medications": st.column_config.NumberColumn("Medications", format="%d"),
            "num_lab_procedures": st.column_config.NumberColumn("Lab Tests", format="%d"),
            "num_procedures": st.column_config.NumberColumn("Procedures", format="%d"),
            "number_diagnoses": st.column_config.NumberColumn("Diagnoses", format="%d"),
            "diag_1": "Primary ICD-9",
            "diag_1_group": "Dx Group",
            "readmitted": "Readmission",
            "record_source": "Data Origin",
        },
    )
    st.caption(
        f"Displaying {len(preview):,} encounters from the filtered set of {int(kpis['total_encounters']):,} encounters. "
        "<code>record_source = uci</code> represents verified historical records; <code>demo_user</code> indicates demo entries.",
        unsafe_allow_html=True,
    )


def main() -> None:
    if not admin_login():
        st.stop()

    if "nav_page" not in st.session_state or st.session_state.nav_page not in NAV_PAGES:
        st.session_state.nav_page = PAGE_OVERVIEW

    try:
        options = cached_filter_options()
    except DataLoadError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(
            "Could not read the hospital SQLite database. "
            "Build it first with `python src/database.py`."
        )
        st.exception(exc)
        st.stop()

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 6px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px;">
              <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 40px; height: 40px; border-radius: 10px; background: linear-gradient(135deg, #9385F5, #4880FF);
                            display: flex; align-items: center; justify-content: center; font-size: 1.3rem; box-shadow: 0 4px 12px rgba(147, 133, 245, 0.3);">
                  🏥
                </div>
                <div>
                  <div style="font-weight: 800; font-size: 0.98rem; color: #FFFFFF; line-height: 1.2; letter-spacing: -0.01em;">Hospital Intelligence</div>
                  <div style="font-size: 0.74rem; color: #A5B4FC;">Operations &amp; Predictive AI</div>
                </div>
              </div>
              <div style="margin-top: 10px; font-size: 0.72rem; color: #CBD5E1; background: rgba(255,255,255,0.06); padding: 5px 8px; border-radius: 6px;">
                Cohort: UCI Diabetes 130-US (1999–2008)
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            NAV_PAGES,
            key="nav_page",
            format_func=lambda x: PAGE_LABELS.get(x, x),
            label_visibility="collapsed",
        )
        st.markdown("---")

        needs_filters = page in {
            PAGE_OVERVIEW,
            PAGE_OPERATIONS,
            PAGE_READMISSION,
            PAGE_ENCOUNTERS,
        }

        selected_genders = []
        selected_ages = []
        selected_races = []
        selected_types = []
        selected_sources = []
        selected_payers = []
        selected_specialties = []
        selected_readmit = []
        selected_origin = []

        if needs_filters:
            st.markdown("### 🎛️ SQL Cohort Filters")

            with st.expander("👤 Demographics & Origin", expanded=False):
                selected_genders = st.multiselect("Gender", options["genders"], placeholder="All genders")
                selected_ages = st.multiselect("Age Cohort", options["age_groups"], placeholder="All age groups")
                selected_races = st.multiselect("Race / Ethnicity", options["races"], placeholder="All races")
                selected_origin = st.multiselect("Record Origin", options["record_sources"], placeholder="UCI + Demo records")

            with st.expander("🏥 Clinical & Admission", expanded=False):
                selected_types = st.multiselect("Admission Type", options["admission_types"], placeholder="All admission types")
                selected_sources = st.multiselect("Admission Source", options["admission_sources"], placeholder="All admission sources")
                selected_payers = st.multiselect("Payer / Insurance", options["payers"], placeholder="All payers")
                selected_specialties = st.multiselect("Medical Specialty", options["specialties"], placeholder="All specialties")
                selected_readmit = st.multiselect("Readmission Outcome", options["readmission_statuses"], placeholder="All outcomes")

            # Active filter indicator
            filter_count = sum(bool(x) for x in [
                selected_genders, selected_ages, selected_races, selected_types,
                selected_sources, selected_payers, selected_specialties, selected_readmit, selected_origin
            ])

            if filter_count > 0:
                st.caption(f"⚡ **{filter_count} active filter(s) applied**")
            else:
                st.caption("Active: All 101,766 encounters included.")

            st.markdown("---")
            st.markdown(
                '<p style="font-size:0.78rem; color:#A0AEC0; line-height:1.4;">'
                "<b>Notice:</b> The UCI extract contains no calendar admission timestamps. "
                'Length of stay is native <code style="color: #202B45 !important;">time_in_hospital</code> (1–14 days).'
                "</p>",
                unsafe_allow_html=True,
            )

        admin_logout_button()

    filters = dict(
        genders=selected_genders or None,
        age_groups=selected_ages or None,
        races=selected_races or None,
        admission_types=selected_types or None,
        admission_sources=selected_sources or None,
        payers=selected_payers or None,
        specialties=selected_specialties or None,
        readmission_statuses=selected_readmit or None,
        record_sources=selected_origin or None,
    )

    if page == PAGE_ADD:
        page_header(
            "Data Management",
            "Add Demo Patient / Encounter",
            "User-entered rows are saved with record_source = demo_user to preserve original UCI dataset integrity.",
        )
        add_patient_form()
        return

    if page == PAGE_RISK:
        page_header(
            "Machine Learning Intelligence",
            "30-Day Readmission Risk Scoring",
            "Supervised machine learning pipeline predicting 30-day readmissions (readmitted = <30) on UCI Diabetes 130-US Hospitals.",
        )
        render_risk_prediction_section()
        return

    kpis = get_basic_kpis(**filters).iloc[0]
    empty = int(kpis["total_encounters"] or 0) == 0

    if page == PAGE_OVERVIEW:
        page_header(
            "Operations & Clinical Intelligence",
            "Executive Inpatient Overview",
            "UCI Diabetes 130-US Hospitals (1999–2008) · 101,766 Encounters · 71,518 Unique Patients · Modern SaaS Analytics",
        )
        if empty:
            st.warning("No encounters match the current SQL filter criteria. Broaden your filters in the sidebar.")
            return
        render_kpi_row(kpis)
        render_overview_charts(filters)
        return

    if empty:
        st.warning("No encounters match the current SQL filter criteria. Broaden your filters in the sidebar.")
        return

    if page == PAGE_OPERATIONS:
        page_header(
            "Inpatient Operations",
            "Census, Length of Stay, & Clinical Utilization",
            "Real-time SQL aggregates calculated over encounters and patient records in SQLite.",
        )
        render_kpi_row(kpis)
        render_operations_charts(filters)
        return

    if page == PAGE_READMISSION:
        page_header(
            "Outcome Analytics",
            "30-Day Readmission Benchmarking",
            "Clinical and operational patterns associated with early unplanned hospital readmissions (<30 days).",
        )
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-top: 3px solid {MUTED_BLUE};">
                  <div class="kpi-header">
                    <span class="kpi-title">Total Encounters</span>
                    <span class="kpi-icon-pill" style="background: rgba(72, 128, 255, 0.12); color: {MUTED_BLUE};">📋</span>
                  </div>
                  <div class="kpi-value">{int(kpis['total_encounters']):,}</div>
                  <div class="kpi-subtitle">Inpatient admissions in scope</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            rate = float(kpis['readmission_30d_rate_pct'])
            st.markdown(
                f"""
                <div class="kpi-card" style="border-top: 3px solid {ROSE};">
                  <div class="kpi-header">
                    <span class="kpi-title">30-Day Readmission Rate</span>
                    <span class="kpi-icon-pill" style="background: rgba(244, 63, 94, 0.12); color: {ROSE};">🔄</span>
                  </div>
                  <div class="kpi-value" style="color: {ROSE};">{rate:.2f}%</div>
                  <div class="kpi-subtitle">Target CMS benchmark: &lt;30 days</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-top: 3px solid {LAVENDER};">
                  <div class="kpi-header">
                    <span class="kpi-title">Average Length of Stay</span>
                    <span class="kpi-icon-pill" style="background: rgba(147, 133, 245, 0.12); color: {LAVENDER_DARK};">⏱️</span>
                  </div>
                  <div class="kpi-value">{float(kpis['average_los_days']):.1f} <span style="font-size:0.95rem; font-weight:500; color:{SLATE};">days</span></div>
                  <div class="kpi-subtitle">Mean stay duration</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.write("")
        render_readmission_charts(filters)
        return

    if page == PAGE_ENCOUNTERS:
        page_header(
            "Encounter Explorer",
            "Filtered Inpatient Encounters Table",
            "Tabular view of historical encounters matching active SQL filter conditions, ordered by Encounter ID.",
        )
        render_encounters_table(filters, kpis)


if __name__ == "__main__":
    main()
