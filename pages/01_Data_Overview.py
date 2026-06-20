"""
pages/01_Data_Overview.py  —  Layer 1: Data Quality, Audit & EDA
Visual design: HRV Design System (utils/styles.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_processing import (
    HRV_COLUMNS, NUMERIC_COLUMNS, get_descriptive_stats,
    get_outliers_iqr, get_outliers_zscore, validate_physiological_ranges,
    BMI_CATEGORY_ORDER, BMI_CATEGORY_COLORS,
    AGE_GROUP_ORDER, STRESS_LEVEL_ORDER, STRESS_LEVEL_COLORS,
)
from utils.styles import (
    inject_css, page_header, section_header, insight_panel,
    apply_chart_style, divider, metric_row, COLORS, CHART_SEQ,
)

st.set_page_config(page_title="Data Overview · HRV", page_icon="📊", layout="wide")
inject_css()

if "df" not in st.session_state:
    st.warning("No data loaded — upload your CSV on the Home page first.")
    st.stop()

df  = st.session_state["df"]
log = st.session_state.get("preprocessing_log", {})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    clinics = ["All"] + sorted(df["Clinic_Name"].dropna().unique().tolist()) if "Clinic_Name" in df.columns else ["All"]
    sel_clinic = st.selectbox("Clinic", clinics)
    sexes = ["All"] + sorted(df["Sex"].dropna().unique().tolist()) if "Sex" in df.columns else ["All"]
    sel_sex = st.selectbox("Sex", sexes)
    age_min = int(df["Age"].min()) if "Age" in df.columns else 0
    age_max = int(df["Age"].max()) if "Age" in df.columns else 100
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))

df_f = df.copy()
if sel_clinic != "All" and "Clinic_Name" in df_f.columns:
    df_f = df_f[df_f["Clinic_Name"] == sel_clinic]
if sel_sex != "All" and "Sex" in df_f.columns:
    df_f = df_f[df_f["Sex"] == sel_sex]
if "Age" in df_f.columns:
    df_f = df_f[(df_f["Age"] >= age_range[0]) & (df_f["Age"] <= age_range[1])]

page_header("Data Quality & Exploratory Analysis", f"{len(df_f):,} records · after filters")

# ── A. Quality summary ────────────────────────────────────────────────────────
section_header("A · Data Quality Summary")

n_pts   = df_f["Patient_Name"].nunique() if "Patient_Name" in df_f.columns else len(df_f)
n_clin  = df_f["Clinic_Name"].nunique()  if "Clinic_Name"  in df_f.columns else 0
total   = df_f.shape[0] * df_f.shape[1]
compl   = round(df_f.notna().sum().sum() / total * 100, 1) if total else 100.0
d_min   = df_f["Date"].min().strftime("%d %b %Y") if "Date" in df_f.columns and pd.notna(df_f["Date"].min()) else "—"
d_max   = df_f["Date"].max().strftime("%d %b %Y") if "Date" in df_f.columns and pd.notna(df_f["Date"].max()) else "—"

metric_row([
    ("Patients",          f"{n_pts:,}",      ""),
    ("Clinics",           f"{n_clin}",        ""),
    ("Reports",           f"{len(df_f):,}",   ""),
    ("Completeness",      f"{compl:.1f}%",    ""),
    ("Date range",        d_min,              f"to {d_max}"),
    ("Stress ≤ 20 removed", f"{log.get('records_removed_stress_filter',0):,}", "pre-filter"),
    ("Duplicates removed",  f"{log.get('records_removed_duplicates',0):,}",    ""),
])

divider()

# Missing value matrix
mv     = df_f.isnull().sum()
mv_pct = (mv / len(df_f) * 100).round(2)
mv_df  = pd.DataFrame({"Column": mv.index, "Missing": mv.values, "%": mv_pct.values})
mv_df  = mv_df[mv_df["Missing"] > 0]

if len(mv_df):
    st.dataframe(mv_df, use_container_width=True, hide_index=True)
    fig = px.imshow(df_f.isnull().T.astype(int),
                    color_continuous_scale=["#F7F8FA", COLORS["blue"]],
                    aspect="auto", height=280)
    fig = apply_chart_style(fig, "Missing Value Matrix")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    insight_panel("No missing values detected in the filtered dataset.", kind="success")

# ── B. Demographics ───────────────────────────────────────────────────────────
section_header("B · Demographic Analysis")

if "Age" in df_f.columns:
    st.markdown("**Age Distribution**")
    t1, t2, t3 = st.tabs(["Histogram", "Boxplot", "Density"])
    with t1:
        fig = px.histogram(df_f, x="Age", nbins=20,
                           color_discrete_sequence=[COLORS["slate"]])
        st.plotly_chart(apply_chart_style(fig, "Age Histogram", 300), use_container_width=True)
    with t2:
        fig = px.box(df_f, y="Age", points="outliers",
                     color_discrete_sequence=[COLORS["blue"]])
        st.plotly_chart(apply_chart_style(fig, "Age Boxplot", 300), use_container_width=True)
    with t3:
        fig = px.violin(df_f, y="Age", box=True, points=False,
                        color_discrete_sequence=[COLORS["steel"]])
        st.plotly_chart(apply_chart_style(fig, "Age Density", 300), use_container_width=True)

    if "Age_Group" in df_f.columns:
        ag = df_f["Age_Group"].value_counts().reindex(AGE_GROUP_ORDER).dropna().reset_index()
        ag.columns = ["Age Group", "Count"]
        fig = px.bar(ag, x="Age Group", y="Count",
                     color_discrete_sequence=[COLORS["blue"]],
                     category_orders={"Age Group": AGE_GROUP_ORDER})
        st.plotly_chart(apply_chart_style(fig, "Count by Age Group", 300), use_container_width=True)

divider()

c1, c2 = st.columns(2)
with c1:
    if "Sex" in df_f.columns:
        sc = df_f["Sex"].value_counts().reset_index()
        sc.columns = ["Sex","Count"]
        fig = px.pie(sc, names="Sex", values="Count",
                     color_discrete_sequence=[COLORS["slate"], COLORS["steel"]])
        st.plotly_chart(apply_chart_style(fig, "Sex Distribution", 300, show_legend=True), use_container_width=True)

with c2:
    if "Heart_Rate_bpm" in df_f.columns:
        fig = px.histogram(df_f, x="Heart_Rate_bpm", nbins=30,
                           color_discrete_sequence=[COLORS["green"]])
        st.plotly_chart(apply_chart_style(fig, "Heart Rate Distribution (bpm)", 300), use_container_width=True)

# Anthropometrics
anthr = [c for c in ["Height_cm","Weight_kg","BMI"] if c in df_f.columns]
if anthr:
    cols = st.columns(len(anthr))
    lbl  = {"Height_cm":"Height (cm)","Weight_kg":"Weight (kg)","BMI":"BMI (kg/m²)"}
    for col, cn in zip(cols, anthr):
        with col:
            fig = px.histogram(df_f, x=cn, nbins=25,
                               color_discrete_sequence=[COLORS["blue"]])
            st.plotly_chart(apply_chart_style(fig, lbl.get(cn,cn), 260), use_container_width=True)

if "BMI_Category" in df_f.columns:
    bc = df_f["BMI_Category"].value_counts().reindex(BMI_CATEGORY_ORDER).dropna().reset_index()
    bc.columns = ["BMI Category","Count"]
    bc["Pct"]  = (bc["Count"]/bc["Count"].sum()*100).round(1)
    fig = px.bar(bc, x="BMI Category", y="Count",
                 color="BMI Category", color_discrete_map=BMI_CATEGORY_COLORS,
                 category_orders={"BMI Category": BMI_CATEGORY_ORDER}, text="Pct")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(apply_chart_style(fig, "BMI Category Distribution", 340), use_container_width=True)

# ── C. Quality checks ─────────────────────────────────────────────────────────
section_header("C · Data Quality Checks")

hrv_p = [c for c in HRV_COLUMNS if c in df_f.columns]
_, iqr_info = get_outliers_iqr(df_f, hrv_p)
_, z_info   = get_outliers_zscore(df_f, hrv_p)

out_rows = [{"Variable": c,
             "IQR Outliers": iqr_info.get(c,{}).get("count",0),
             "IQR %": f"{iqr_info.get(c,{}).get('pct',0):.1f}%",
             "Z-Score Outliers": z_info.get(c,{}).get("count",0),
             "Z-Score %": f"{z_info.get(c,{}).get('pct',0):.1f}%",
             "Lower Fence": iqr_info.get(c,{}).get("lower_fence","—"),
             "Upper Fence": iqr_info.get(c,{}).get("upper_fence","—")}
            for c in hrv_p]
st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

st.markdown("**Physiological Range Validation**")
val_df = validate_physiological_ranges(df_f)
st.dataframe(val_df, use_container_width=True, hide_index=True)

checks = []
if all(c in df_f.columns for c in ["LF_Power_ms2","HF_Power_ms2","LF_HF_Ratio"]):
    disc = ((df_f["LF_Power_ms2"]/df_f["HF_Power_ms2"].replace(0,np.nan) - df_f["LF_HF_Ratio"]).abs() > 0.5).sum()
    checks.append({"Check":"LF/HF computed = stored","Issues":int(disc),"Status":"✓ OK" if disc==0 else f"⚠ {disc}"})
if all(c in df_f.columns for c in ["pNN50_pct","pNN20_pct"]):
    inv = (df_f["pNN50_pct"] > df_f["pNN20_pct"]).sum()
    checks.append({"Check":"pNN50 ≤ pNN20","Issues":int(inv),"Status":"✓ OK" if inv==0 else f"⚠ {inv}"})
if "Stress_Score" in df_f.columns:
    oor = ((df_f["Stress_Score"]<=20)|(df_f["Stress_Score"]>100)).sum()
    checks.append({"Check":"Stress Score in (20,100]","Issues":int(oor),"Status":"✓ OK" if oor==0 else f"⚠ {oor}"})
if checks:
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

# ── D. Descriptive statistics ─────────────────────────────────────────────────
section_header("D · Descriptive Statistics")

desc_cols = [c for c in NUMERIC_COLUMNS + ["Stress_Score"] if c in df_f.columns]
desc_df   = get_descriptive_stats(df_f, desc_cols)
st.dataframe(desc_df.style.format("{:.3f}", na_rep="—"), use_container_width=True)

if "Stress_Score" in df_f.columns:
    sc1, sc2 = st.columns(2)
    with sc1:
        fig = px.histogram(df_f, x="Stress_Score", nbins=30,
                           color_discrete_sequence=[COLORS["slate"]])
        for t, lbl in [(40,"Norm/Mild"),(55,"Mild/Mod"),(70,"Mod/Sev"),(85,"Sev/Ext")]:
            fig.add_vline(x=t, line_dash="dot", line_color=COLORS["steel"],
                          annotation_text=lbl, annotation_font_size=9)
        st.plotly_chart(apply_chart_style(fig, "Stress Score Distribution", 320), use_container_width=True)
    with sc2:
        if "Stress_Level" in df_f.columns:
            sl = df_f["Stress_Level"].value_counts().reindex(STRESS_LEVEL_ORDER).dropna().reset_index()
            sl.columns = ["Stress Level","Count"]
            fig = px.bar(sl, x="Stress Level", y="Count",
                         color="Stress Level", color_discrete_map=STRESS_LEVEL_COLORS,
                         category_orders={"Stress Level": STRESS_LEVEL_ORDER})
            st.plotly_chart(apply_chart_style(fig, "5-Tier Stress Level Breakdown", 320), use_container_width=True)

insight_panel(
    "Skewness > 1 (or < −1) indicates substantial asymmetry — common in HRV "
    "frequency-domain metrics such as LF Power and HF Power. "
    "Kurtosis > 3 indicates heavier tails than a normal distribution.",
    kind="info",
)

# ── E. Downloads ──────────────────────────────────────────────────────────────
section_header("E · Download Reports")
dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button("Download Cleaned Dataset (CSV)",
                       df_f.to_csv(index=False).encode("utf-8"),
                       "hrv_cleaned_data.csv", "text/csv", use_container_width=True)
with dl2:
    st.download_button("Download Descriptive Stats (CSV)",
                       desc_df.to_csv().encode("utf-8"),
                       "hrv_descriptive_stats.csv", "text/csv", use_container_width=True)
with dl3:
    st.download_button("Download Outlier Summary (CSV)",
                       pd.DataFrame(out_rows).to_csv(index=False).encode("utf-8"),
                       "hrv_outlier_summary.csv", "text/csv", use_container_width=True)
