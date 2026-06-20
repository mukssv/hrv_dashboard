"""
pages/07_Clinic_Benchmarking.py  —  Advanced: Clinic Benchmarking
Visual design: HRV Design System (utils/styles.py)
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress

from utils.data_processing import (
    HRV_COLUMNS, HRV_REFERENCE_RANGES,
    STRESS_LEVEL_ORDER, STRESS_LEVEL_COLORS,
    BMI_CATEGORY_ORDER, BMI_CATEGORY_COLORS, AGE_GROUP_ORDER,
)
from utils.styles import (
    inject_css, page_header, section_header, insight_panel,
    apply_chart_style, divider, metric_row, hrv_card, COLORS, CHART_SEQ,
)

st.set_page_config(page_title="Clinic Benchmarking · HRV", page_icon="🏥", layout="wide")
inject_css()

if "df" not in st.session_state:
    st.warning("No data loaded — upload your CSV on the Home page first.")
    st.stop()

df = st.session_state["df"]

if "Clinic_Name" not in df.columns:
    st.error("Clinic_Name column not found.")
    st.stop()

clinics   = sorted(df["Clinic_Name"].dropna().unique().tolist())
n_clinics = len(clinics)
if n_clinics < 2:
    st.info("At least 2 clinics are required for benchmarking.")
    st.stop()

with st.sidebar:
    st.markdown("### Metrics")
    benchmark_cols = [c for c in [
        "Stress_Score","SDNN_ms","RMSSD_ms","pNN50_pct","pNN20_pct",
        "LF_HF_Ratio","HF_Power_ms2","LF_Power_ms2",
        "HRV_Health_Index","Parasympathetic_Activity_Index","Sympathetic_Dominance_Index",
    ] if c in df.columns]
    selected_metrics = st.multiselect("Compare metrics", benchmark_cols, default=benchmark_cols[:6])

if not selected_metrics:
    st.warning("Select at least one metric from the sidebar.")
    st.stop()

page_header("Clinic Benchmarking",
            "Rankings · multi-metric comparisons · age-adjusted HRV · high-risk profiling")

# ── A. Overview table ─────────────────────────────────────────────────────────
section_header("A · Clinic Overview")

base = df.groupby("Clinic_Name").agg(
    Patients   =("Patient_Name","nunique") if "Patient_Name" in df.columns else ("Stress_Score","count"),
    Reports    =("Stress_Score","count"),
    Avg_Age    =("Age","mean"),
    Avg_Stress =("Stress_Score","mean"),
).round(2).reset_index()

if "Stress_Level" in df.columns:
    high_risk = {"Severe Stress","Extremely Severe Stress"}
    hr = df.groupby("Clinic_Name").apply(
        lambda g: g["Stress_Level"].isin(high_risk).mean()*100
    ).round(1).reset_index(name="High_Risk_%")
    base = base.merge(hr, on="Clinic_Name", how="left")

for m in selected_metrics:
    if m in df.columns:
        avg = df.groupby("Clinic_Name")[m].mean().round(2).reset_index(name=f"Avg_{m}")
        base = base.merge(avg, on="Clinic_Name", how="left")

st.dataframe(base.sort_values("Avg_Stress", ascending=True),
             use_container_width=True, hide_index=True)
st.download_button("Download Clinic Summary (CSV)",
                   base.to_csv(index=False).encode("utf-8"),
                   "clinic_benchmarking.csv","text/csv")

divider()

# ── B. Rankings ───────────────────────────────────────────────────────────────
section_header("B · Clinic Rankings")

rank_metric     = st.selectbox("Rank clinics by", selected_metrics)
higher_better   = {"SDNN_ms","RMSSD_ms","pNN50_pct","pNN20_pct","HF_Power_ms2",
                   "HRV_Health_Index","Parasympathetic_Activity_Index","Mean_RR_ms"}
is_hb = rank_metric in higher_better
note  = "Higher is better" if is_hb else "Lower is better"

c_means = df.groupby("Clinic_Name")[rank_metric].mean()
c_means = c_means.sort_values(ascending=not is_hb)

fig = px.bar(x=c_means.values, y=c_means.index, orientation="h",
             color=c_means.values,
             color_continuous_scale=[[0,COLORS["steel"]],[1,COLORS["slate"]]] if is_hb
                                else [[0,COLORS["slate"]],[1,COLORS["steel"]]],
             labels={"x":rank_metric,"y":"Clinic"})
fig.update_traces(marker_line_width=0)
for i, (clinic, val) in enumerate(zip(c_means.index, c_means.values)):
    fig.add_annotation(x=val*0.97, y=clinic, text=f"#{i+1}", showarrow=False,
                       xanchor="right", font=dict(size=11,color="white",family="Inter"))
fig.update_layout(yaxis={"categoryorder":"array","categoryarray":c_means.index.tolist()},
                  height=max(300, n_clinics*65))
st.plotly_chart(apply_chart_style(fig, f"Clinic Ranking — {rank_metric} ({note})"),
                use_container_width=True)

divider()

# ── C. Multi-metric comparison ────────────────────────────────────────────────
section_header("C · Multi-Metric Comparison")

chosen = st.multiselect("Metrics to compare", selected_metrics, default=selected_metrics[:4])
if chosen:
    cdf = df.groupby("Clinic_Name")[chosen].mean().reset_index()
    mlt = cdf.melt(id_vars="Clinic_Name", var_name="Metric", value_name="Mean Value")
    fig = px.bar(mlt, x="Metric", y="Mean Value", color="Clinic_Name",
                 barmode="group", color_discrete_sequence=CHART_SEQ)
    st.plotly_chart(apply_chart_style(fig, "Clinic Comparison — Mean Values by Metric"),
                    use_container_width=True)

divider()

# ── D. Stress distribution ────────────────────────────────────────────────────
section_header("D · Stress Score Distribution by Clinic")

if "Stress_Score" in df.columns:
    fig = px.box(df.dropna(subset=["Stress_Score","Clinic_Name"]),
                 x="Clinic_Name", y="Stress_Score", color="Clinic_Name",
                 points="outliers", color_discrete_sequence=CHART_SEQ)
    for t, c in [(40,STRESS_LEVEL_COLORS["Mild Stress"]),(55,STRESS_LEVEL_COLORS["Moderate Stress"]),
                  (70,STRESS_LEVEL_COLORS["Severe Stress"]),(85,STRESS_LEVEL_COLORS["Extremely Severe Stress"])]:
        fig.add_hline(y=t, line_dash="dot", line_color=c, opacity=0.6,
                      annotation_text=str(t), annotation_font_size=9)
    st.plotly_chart(apply_chart_style(fig, "Stress Score Distribution per Clinic"),
                    use_container_width=True)

    if "Stress_Level" in df.columns:
        sl = df.groupby(["Clinic_Name","Stress_Level"]).size().reset_index(name="Count")
        sl["Pct"] = sl.groupby("Clinic_Name")["Count"].transform(lambda x: x/x.sum()*100).round(1)
        fig = px.bar(sl, x="Clinic_Name", y="Pct", color="Stress_Level",
                     color_discrete_map=STRESS_LEVEL_COLORS,
                     category_orders={"Stress_Level":STRESS_LEVEL_ORDER},
                     barmode="stack", text="Pct",
                     labels={"Pct":"% of clinic population"})
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="inside", textfont_size=9)
        st.plotly_chart(apply_chart_style(fig, "5-Tier Stress Level Breakdown (%) by Clinic"),
                        use_container_width=True)

divider()

# ── E. Age-adjusted comparison ────────────────────────────────────────────────
section_header("E · Age-Adjusted HRV Comparison")
insight_panel(
    "Age-adjusted values remove the expected age-related HRV decline so clinics with older "
    "populations are compared on equal footing. Method: OLS residuals from Age → HRV regression.",
    kind="info",
)

hrv_present = [c for c in HRV_COLUMNS if c in df.columns]
adj_metric  = st.selectbox("Metric to age-adjust", hrv_present)

if "Age" in df.columns and adj_metric in df.columns:
    adf = df[["Clinic_Name","Age",adj_metric]].dropna()
    slope, intercept, *_ = linregress(adf["Age"], adf[adj_metric])
    adf["Age_Adjusted"] = adf[adj_metric] - (slope*adf["Age"]+intercept)

    comp = pd.DataFrame({
        "Clinic":       adf.groupby("Clinic_Name")[adj_metric].mean().index,
        "Unadjusted":   adf.groupby("Clinic_Name")[adj_metric].mean().values.round(2),
        "Age-Adjusted": adf.groupby("Clinic_Name")["Age_Adjusted"].mean().values.round(2),
        "Avg Age":      adf.groupby("Clinic_Name")["Age"].mean().values.round(1),
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    mlt2 = comp.melt(id_vars="Clinic", value_vars=["Unadjusted","Age-Adjusted"],
                      var_name="Type", value_name="Value")
    fig = px.bar(mlt2, x="Clinic", y="Value", color="Type", barmode="group",
                 color_discrete_map={"Unadjusted":COLORS["border"],"Age-Adjusted":COLORS["blue"]})
    st.plotly_chart(apply_chart_style(fig, f"{adj_metric} — Unadjusted vs Age-Adjusted"),
                    use_container_width=True)

divider()

# ── F. BMI & Age profiles ─────────────────────────────────────────────────────
section_header("F · Demographic Profiles by Clinic")

fb1, fb2 = st.columns(2)
with fb1:
    if "BMI_Category" in df.columns:
        bmi_c = df.groupby(["Clinic_Name","BMI_Category"]).size().reset_index(name="Count")
        bmi_c["Pct"] = bmi_c.groupby("Clinic_Name")["Count"].transform(lambda x: x/x.sum()*100).round(1)
        bmi_c["BMI_Category"] = pd.Categorical(bmi_c["BMI_Category"],categories=BMI_CATEGORY_ORDER,ordered=True)
        bmi_c = bmi_c.sort_values("BMI_Category")
        fig = px.bar(bmi_c,x="Clinic_Name",y="Pct",color="BMI_Category",
                     color_discrete_map=BMI_CATEGORY_COLORS,
                     category_orders={"BMI_Category":BMI_CATEGORY_ORDER},
                     barmode="stack",labels={"Pct":"%"})
        st.plotly_chart(apply_chart_style(fig,"BMI Category by Clinic (%)",320),use_container_width=True)

with fb2:
    if "Age_Group" in df.columns:
        age_c = df.groupby(["Clinic_Name","Age_Group"]).size().reset_index(name="Count")
        age_c["Pct"] = age_c.groupby("Clinic_Name")["Count"].transform(lambda x: x/x.sum()*100).round(1)
        age_c["Age_Group"] = pd.Categorical(age_c["Age_Group"],categories=AGE_GROUP_ORDER,ordered=True)
        age_c = age_c.sort_values("Age_Group")
        fig = px.bar(age_c,x="Clinic_Name",y="Pct",color="Age_Group",
                     category_orders={"Age_Group":AGE_GROUP_ORDER},
                     color_discrete_sequence=CHART_SEQ,barmode="stack",labels={"Pct":"%"})
        st.plotly_chart(apply_chart_style(fig,"Age Group by Clinic (%)",320),use_container_width=True)

divider()

# ── G. Individual deep-dive ───────────────────────────────────────────────────
section_header("G · Individual Clinic Deep-Dive")

sel_c = st.selectbox("Select clinic", clinics)
cdf   = df[df["Clinic_Name"]==sel_c]

metric_row([
    ("Patients", f"{cdf['Patient_Name'].nunique():,}" if "Patient_Name" in cdf.columns else f"{len(cdf):,}", ""),
    ("Reports",  f"{len(cdf):,}",                       ""),
    ("Avg Age",  f"{cdf['Age'].mean():.1f}" if "Age" in cdf.columns else "—",            "years"),
    ("Avg Stress",f"{cdf['Stress_Score'].mean():.1f}" if "Stress_Score" in cdf.columns else "—","of 100"),
])

if "Age" in cdf.columns and "Stress_Score" in cdf.columns:
    fig = px.scatter(cdf, x="Age", y="Stress_Score",
                     color="Stress_Level" if "Stress_Level" in cdf.columns else None,
                     color_discrete_map=STRESS_LEVEL_COLORS,
                     category_orders={"Stress_Level":STRESS_LEVEL_ORDER},
                     trendline="ols", trendline_color_override=COLORS["blue"],
                     opacity=0.6,
                     hover_data=["Patient_Name"] if "Patient_Name" in cdf.columns else None)
    for t, c in [(40,STRESS_LEVEL_COLORS["Mild Stress"]),(55,STRESS_LEVEL_COLORS["Moderate Stress"]),
                  (70,STRESS_LEVEL_COLORS["Severe Stress"]),(85,STRESS_LEVEL_COLORS["Extremely Severe Stress"])]:
        fig.add_hline(y=t, line_dash="dot", line_color=c, opacity=0.4, annotation_text=str(t), annotation_font_size=9)
    st.plotly_chart(apply_chart_style(fig, f"Age vs Stress Score — {sel_c}", 380),
                    use_container_width=True)

if "RMSSD_ms" in cdf.columns:
    fig = px.histogram(cdf, x="RMSSD_ms", nbins=20, color_discrete_sequence=[COLORS["green"]])
    fig.add_vline(x=15.7, line_dash="dot", line_color=COLORS["slate"],
                  annotation_text="Low boundary (15.7 ms)", annotation_font_size=9)
    fig.add_vline(x=40.0, line_dash="dot", line_color=COLORS["green"],
                  annotation_text="Optimal boundary (40 ms)", annotation_font_size=9)
    st.plotly_chart(apply_chart_style(fig, f"RMSSD Distribution — {sel_c}", 300),
                    use_container_width=True)

if "Stress_Score" in cdf.columns:
    risk_cols = [c for c in ["Patient_Name","Age","Sex","Stress_Score","Stress_Level",
                              "RMSSD_ms","SDNN_ms","LF_HF_Ratio","HRV_Health_Index"] if c in cdf.columns]
    top_risk  = cdf.nlargest(20,"Stress_Score")[risk_cols].reset_index(drop=True)
    st.markdown("**Highest-Risk Patients**")
    st.dataframe(top_risk, use_container_width=True, hide_index=True)
    st.download_button(f"Download High-Risk Patients — {sel_c} (CSV)",
                       top_risk.to_csv(index=False).encode("utf-8"),
                       f"high_risk_{sel_c.replace(' ','_')}.csv","text/csv")
