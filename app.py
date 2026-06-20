"""
app.py — HRV Clinical Insights Dashboard
Home page: data upload, preprocessing audit, session state management.
Visual design: HRV Design System (utils/styles.py)
"""

import streamlit as st
import pandas as pd

from utils.data_processing import load_data, preprocess_data, HRV_COLUMNS
from utils.styles import (
    inject_css, page_header, section_header, insight_panel,
    divider, metric_row, hrv_card, COLORS,
)

st.set_page_config(
    page_title="HRV Clinical Insights",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 0 0.6rem 0;">
        <div style="font-size:1.05rem;font-weight:700;color:#FFFFFF;
                    letter-spacing:-0.01em;">HRV Insights</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.5);
                    margin-top:0.15rem;">Clinical Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("""
    <div style="font-size:0.82rem;color:rgba(255,255,255,0.75);line-height:2.1;">
    &nbsp;📊&nbsp; Data Overview<br>
    &nbsp;🫀&nbsp; HRV Physiology<br>
    &nbsp;📈&nbsp; Relationships<br>
    &nbsp;🔍&nbsp; Segmentation<br>
    &nbsp;👤&nbsp; Patient Profiles<br>
    &nbsp;🏥&nbsp; Clinic Benchmarking<br>
    &nbsp;🌍&nbsp; Population Health<br>
    &nbsp;📋&nbsp; Reports
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Upload your CSV on this page to begin.")

# ── Page header ───────────────────────────────────────────────────────────────
page_header(
    "HRV Clinical Insights Dashboard",
    "Heart Rate Variability & Physiological Stress Analysis · "
    "For clinicians, researchers, and healthcare administrators",
)

# ── Upload area ───────────────────────────────────────────────────────────────
section_header("Data Source")

st.markdown("""
<div style="background:#FFFFFF;border:1.5px dashed #D0D7E2;border-radius:14px;
            padding:1.8rem 2rem;margin-bottom:1rem;">
    <div style="font-size:0.84rem;color:#6B7A8D;line-height:1.7;">
        Upload your clinic HRV export CSV. Column names are normalised automatically —
        no manual editing required. Records with <strong>Stress Score ≤ 20</strong>
        are excluded per protocol.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Choose CSV file",
    type=["csv"],
    label_visibility="collapsed",
)

# ── Processing ────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    if st.session_state.get("uploaded_filename") != uploaded_file.name:
        with st.spinner("Loading and preprocessing data…"):
            df_raw, load_error = load_data(uploaded_file)
            if load_error:
                st.error(f"Could not read file: {load_error}")
                st.stop()
            df_clean, log = preprocess_data(df_raw.copy())
            st.session_state["df_raw"]            = df_raw
            st.session_state["df"]                = df_clean
            st.session_state["preprocessing_log"] = log
            st.session_state["uploaded_filename"] = uploaded_file.name

    df_raw = st.session_state["df_raw"]
    df     = st.session_state["df"]
    log    = st.session_state["preprocessing_log"]

    # ── Preprocessing audit ───────────────────────────────────────────────────
    section_header("Preprocessing Audit")

    removed_stress = log.get("records_removed_stress_filter", 0)
    removed_dedup  = log.get("records_removed_duplicates", 0)
    completeness   = log.get("completeness_score", 100.0)

    metric_row([
        ("Records Loaded",   f"{log.get('initial_records', 0):,}",   "Total rows in CSV"),
        ("Stress ≤ 20 Removed", f"{removed_stress:,}",              "Excluded per protocol"),
        ("Duplicates Removed",  f"{removed_dedup:,}",               "Patient–date pairs"),
        ("Final Dataset",    f"{log.get('final_records', 0):,}",     f"Completeness {completeness:.1f}%"),
    ])

    missing_info = log.get("missing_values", {})
    if missing_info:
        with st.expander(f"Missing values found in {len(missing_info)} column(s)"):
            rows = [{"Column": c, "Missing": v["count"], "%": f"{v['pct']:.1f}%",
                     "Action": "Imputed with column median"}
                    for c, v in missing_info.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        insight_panel("No missing values detected after imputation.", kind="success")

    divider()

    # ── Quick stats ───────────────────────────────────────────────────────────
    section_header("Cohort Overview")

    n_patients = df["Patient_Name"].nunique() if "Patient_Name" in df.columns else len(df)
    n_clinics  = df["Clinic_Name"].nunique()  if "Clinic_Name"  in df.columns else 0
    date_min   = df["Date"].min().strftime("%d %b %Y") if "Date" in df.columns and pd.notna(df["Date"].min()) else "—"
    date_max   = df["Date"].max().strftime("%d %b %Y") if "Date" in df.columns and pd.notna(df["Date"].max()) else "—"
    avg_stress = df["Stress_Score"].mean() if "Stress_Score" in df.columns else 0
    avg_rmssd  = df["RMSSD_ms"].mean()     if "RMSSD_ms"     in df.columns else 0
    pct_high   = ((df.get("Stress_Level","") == "Severe Stress").mean() +
                  (df.get("Stress_Level","") == "Extremely Severe Stress").mean()) * 100 \
                 if "Stress_Level" in df.columns else 0

    metric_row([
        ("Unique Patients",  f"{n_patients:,}",     ""),
        ("Clinics",          f"{n_clinics:,}",       ""),
        ("Reports",          f"{len(df):,}",          ""),
        ("Date Range",       f"{date_min}",               f"to {date_max}"),
        ("Avg Stress Score", f"{avg_stress:.1f}",    ""),
        ("Avg RMSSD",        f"{avg_rmssd:.1f} ms",  "vagal tone marker"),
        ("High-Risk %",      f"{pct_high:.1f}%",     "Severe or Ext. Severe"),
    ])

    # ── Preview ───────────────────────────────────────────────────────────────
    section_header("Data Preview")
    preview_cols = [c for c in [
        "Patient_Name","Age","Sex","Clinic_Name","Date",
        "Stress_Score","Stress_Level","RMSSD_ms","SDNN_ms",
        "LF_HF_Ratio","HRV_Health_Index",
    ] if c in df.columns]
    st.dataframe(df[preview_cols].head(10), use_container_width=True, hide_index=True)

    insight_panel(
        "Data loaded successfully. Use the sidebar to navigate between "
        "analytical layers. Start with <strong>Data Overview</strong> for "
        "a full exploratory analysis.",
        kind="info",
    )

else:
    # ── Welcome state ─────────────────────────────────────────────────────────
    section_header("Expected Format")
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem;">
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        hrv_card("""
        <div style="font-size:0.7rem;font-weight:700;color:#9BA8B8;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">
            Patient & Clinic Columns
        </div>
        <div style="font-size:0.83rem;color:#364A69;line-height:2;">
            Patient Name &nbsp;·&nbsp; Patient Age &nbsp;·&nbsp; Patient Sex<br>
            Patient Weight (kg) &nbsp;·&nbsp; Patient Height (cm)<br>
            Clinic ID &nbsp;·&nbsp; Clinic Name &nbsp;·&nbsp; Report Date
        </div>
        """)
    with col2:
        hrv_card("""
        <div style="font-size:0.7rem;font-weight:700;color:#9BA8B8;
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">
            HRV & Stress Columns
        </div>
        <div style="font-size:0.83rem;color:#364A69;line-height:2;">
            Stress Score &nbsp;·&nbsp; Mean RR (ms) &nbsp;·&nbsp; SDNN (ms)<br>
            RMSSD (ms) &nbsp;·&nbsp; pNN50 (%) &nbsp;·&nbsp; pNN20 (%)<br>
            LF Power (ms²) &nbsp;·&nbsp; HF Power (ms²) &nbsp;·&nbsp; LF/HF Ratio
        </div>
        """)

    divider()
    section_header("Platform Layers")

    la, lb = st.columns(2)
    with la:
        hrv_card("""
        <div style="font-size:0.83rem;color:#364A69;line-height:2.1;">
            <strong style="color:#364A69;">Layer 1</strong> — Data Quality &amp; EDA<br>
            <strong style="color:#364A69;">Layer 2</strong> — HRV Physiology &amp; ANS<br>
            <strong style="color:#364A69;">Layer 3</strong> — Correlations &amp; Statistics<br>
            <strong style="color:#364A69;">Layer 4</strong> — PCA &amp; Clustering
        </div>
        """)
    with lb:
        hrv_card("""
        <div style="font-size:0.83rem;color:#364A69;line-height:2.1;">
            <strong style="color:#364A69;">Layer 5</strong> — Predictive Modelling &amp; SHAP<br>
            <strong style="color:#364A69;">Advanced</strong> — Patient Profiles<br>
            <strong style="color:#364A69;">Advanced</strong> — Clinic Benchmarking<br>
            <strong style="color:#364A69;">Advanced</strong> — Population Health
        </div>
        """)
