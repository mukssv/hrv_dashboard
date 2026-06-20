"""
pages/08_Population_Health.py  —  Advanced: Population Health Overview
Visual design: HRV Design System (utils/styles.py)
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_processing import (
    HRV_COLUMNS, HRV_REFERENCE_RANGES, classify_hrv_value,
    STRESS_LEVEL_ORDER, STRESS_LEVEL_COLORS,
    AGE_GROUP_ORDER, BMI_CATEGORY_ORDER, BMI_CATEGORY_COLORS,
)
from utils.styles import (
    inject_css, page_header, section_header, insight_panel,
    apply_chart_style, divider, metric_row, hrv_card, COLORS, CHART_SEQ, CORR_SCALE,
)

st.set_page_config(page_title="Population Health · HRV", page_icon="🌍", layout="wide")
inject_css()

if "df" not in st.session_state:
    st.warning("No data loaded — upload your CSV on the Home page first.")
    st.stop()

df = st.session_state["df"]

with st.sidebar:
    st.markdown("### Filters")
    sel_clinic = "All Clinics"
    if "Clinic_Name" in df.columns:
        sel_clinic = st.selectbox("Clinic", ["All Clinics"]+sorted(df["Clinic_Name"].dropna().unique().tolist()))
    sel_sex = "All"
    if "Sex" in df.columns:
        sel_sex = st.selectbox("Sex", ["All"]+sorted(df["Sex"].dropna().unique().tolist()))

df_f = df.copy()
if sel_clinic != "All Clinics" and "Clinic_Name" in df_f.columns:
    df_f = df_f[df_f["Clinic_Name"] == sel_clinic]
if sel_sex != "All" and "Sex" in df_f.columns:
    df_f = df_f[df_f["Sex"] == sel_sex]

page_header("Population Health Overview",
            "Cohort trends · risk stratification · physiological distributions · demographic breakdowns")
st.caption(f"{len(df_f):,} records after filters")

# ── A. Headline metrics ───────────────────────────────────────────────────────
section_header("A · Cohort Headline Metrics")

metric_row([
    ("Unique Patients", f"{df_f['Patient_Name'].nunique():,}" if "Patient_Name" in df_f.columns else f"{len(df_f):,}", ""),
    ("Total Reports",   f"{len(df_f):,}",                      ""),
    ("Clinics",         f"{df_f['Clinic_Name'].nunique():,}"   if "Clinic_Name" in df_f.columns else "—",""),
    ("Avg Stress Score",f"{df_f['Stress_Score'].mean():.1f}"  if "Stress_Score" in df_f.columns else "—","of 100"),
    ("Avg RMSSD",       f"{df_f['RMSSD_ms'].mean():.1f} ms"   if "RMSSD_ms" in df_f.columns else "—","vagal marker"),
    ("Avg LF/HF",       f"{df_f['LF_HF_Ratio'].mean():.2f}"   if "LF_HF_Ratio" in df_f.columns else "—","balance"),
])

divider()

# ── B. Risk stratification ────────────────────────────────────────────────────
section_header("B · Risk Stratification (5-Tier)")

if "Stress_Level" in df_f.columns:
    total = len(df_f)
    cols5 = st.columns(5)
    for cw, lbl in zip(cols5, STRESS_LEVEL_ORDER):
        count = (df_f["Stress_Level"]==lbl).sum()
        color = STRESS_LEVEL_COLORS[lbl]
        with cw:
            st.markdown(
                f'<div style="background:{color};color:white;border-radius:12px;'
                f'padding:1rem 0.5rem;text-align:center;">'
                f'<div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.06em;line-height:1.3;margin-bottom:0.35rem;">{lbl}</div>'
                f'<div style="font-size:2rem;font-weight:700;">{count:,}</div>'
                f'<div style="font-size:0.8rem;opacity:0.9;">{count/total*100:.1f}%</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("")
    rb1, rb2 = st.columns(2)
    with rb1:
        if "Sex" in df_f.columns:
            sl = df_f.groupby(["Sex","Stress_Level"]).size().reset_index(name="Count")
            sl["Pct"] = sl.groupby("Sex")["Count"].transform(lambda x: x/x.sum()*100).round(1)
            fig = px.bar(sl,x="Sex",y="Pct",color="Stress_Level",
                         color_discrete_map=STRESS_LEVEL_COLORS,
                         category_orders={"Stress_Level":STRESS_LEVEL_ORDER},
                         barmode="stack",text="Pct",labels={"Pct":"%"})
            fig.update_traces(texttemplate="%{text:.0f}%",textposition="inside",textfont_size=9)
            st.plotly_chart(apply_chart_style(fig,"Stress Level by Sex (%)",320),use_container_width=True)
    with rb2:
        if "Age_Group" in df_f.columns:
            sl = df_f.groupby(["Age_Group","Stress_Level"]).size().reset_index(name="Count")
            sl["Pct"] = sl.groupby("Age_Group")["Count"].transform(lambda x: x/x.sum()*100).round(1)
            sl["Age_Group"] = pd.Categorical(sl["Age_Group"],categories=AGE_GROUP_ORDER,ordered=True)
            sl = sl.sort_values("Age_Group")
            fig = px.bar(sl,x="Age_Group",y="Pct",color="Stress_Level",
                         color_discrete_map=STRESS_LEVEL_COLORS,
                         category_orders={"Stress_Level":STRESS_LEVEL_ORDER,"Age_Group":AGE_GROUP_ORDER},
                         barmode="stack",text="Pct",labels={"Pct":"%"})
            fig.update_traces(texttemplate="%{text:.0f}%",textposition="inside",textfont_size=9)
            st.plotly_chart(apply_chart_style(fig,"Stress Level by Age Group (%)",320),use_container_width=True)

    if "BMI_Category" in df_f.columns:
        sl = df_f.groupby(["BMI_Category","Stress_Level"]).size().reset_index(name="Count")
        sl["Pct"] = sl.groupby("BMI_Category")["Count"].transform(lambda x: x/x.sum()*100).round(1)
        sl["BMI_Category"] = pd.Categorical(sl["BMI_Category"],categories=BMI_CATEGORY_ORDER,ordered=True)
        sl = sl.sort_values("BMI_Category")
        fig = px.bar(sl,x="BMI_Category",y="Pct",color="Stress_Level",
                     color_discrete_map=STRESS_LEVEL_COLORS,
                     category_orders={"Stress_Level":STRESS_LEVEL_ORDER,"BMI_Category":BMI_CATEGORY_ORDER},
                     barmode="stack",text="Pct",labels={"Pct":"%"})
        fig.update_traces(texttemplate="%{text:.0f}%",textposition="inside",textfont_size=9)
        st.plotly_chart(apply_chart_style(fig,"Stress Level by BMI Category (%)",320),use_container_width=True)

divider()

# ── C. Physiological distributions ───────────────────────────────────────────
section_header("C · Physiological Distributions")
insight_panel("Green shading = updated normal reference range. Proportions within each range are summarised below.",
              kind="info")

hrv_p  = [c for c in HRV_COLUMNS if c in df_f.columns]
n_cols = min(4, len(hrv_p))

for i in range(0, len(hrv_p), n_cols):
    row_cols = st.columns(n_cols)
    for cw, cn in zip(row_cols, hrv_p[i:i+n_cols]):
        ref = HRV_REFERENCE_RANGES.get(cn,{})
        lo  = ref.get("healthy_low"); hi = ref.get("healthy_high")
        with cw:
            fig = px.histogram(df_f, x=cn, nbins=25, color_discrete_sequence=[COLORS["blue"]])
            if lo and hi:
                fig.add_vrect(x0=lo,x1=hi,fillcolor=COLORS["green"],opacity=0.08,line_width=0,
                              annotation_text="Normal",annotation_font_size=8)
            fig.update_layout(margin=dict(t=35,b=20,l=5,r=5),height=220,
                              xaxis_title=ref.get("unit",""))
            st.plotly_chart(apply_chart_style(fig, ref.get("description",cn)), use_container_width=True)

# In-range summary
in_range_rows = []
for cn in hrv_p:
    ref = HRV_REFERENCE_RANGES.get(cn,{}); lo = ref.get("healthy_low",-np.inf); hi = ref.get("healthy_high",np.inf)
    pct = ((df_f[cn]>=lo)&(df_f[cn]<=hi)).mean()*100
    in_range_rows.append({"Metric":ref.get("description",cn),
                           "Normal Range":ref.get("clinical_note","").split("|")[0].strip(),
                           "% In Range":round(pct,1),"% Out of Range":round(100-pct,1),
                           "Status":"✓" if pct>=60 else ("~" if pct>=40 else "✗")})
ir_df = pd.DataFrame(in_range_rows)

fig = px.bar(ir_df, x="Metric", y="% In Range", color="% In Range",
             color_continuous_scale=[[0,COLORS["slate"]],[0.5,COLORS["steel"]],[1,COLORS["green"]]],
             text="% In Range")
fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
fig.add_hline(y=50,line_dash="dot",line_color=COLORS["border"],annotation_text="50%",annotation_font_size=9)
fig.update_layout(xaxis_tickangle=-15)
st.plotly_chart(apply_chart_style(fig,"% of Patients Within Normal Reference Range",340),use_container_width=True)
st.dataframe(ir_df, use_container_width=True, hide_index=True)

divider()

# ── D. Temporal trends ────────────────────────────────────────────────────────
section_header("D · Temporal Trends")

if "Date" in df_f.columns and df_f["Date"].notna().sum() > 10:
    df_t = df_f.dropna(subset=["Date"]).copy()
    df_t["YearMonth"] = df_t["Date"].dt.to_period("M").astype(str)
    tm = st.selectbox("Trend metric", ["Stress_Score"]+hrv_p, key="tm")

    if tm in df_t.columns:
        mo = df_t.groupby("YearMonth")[tm].agg(["mean","std","count"]).reset_index()
        mo.columns = ["YearMonth","Mean","Std","N"]
        mo["CI_U"] = mo["Mean"]+1.96*mo["Std"]/np.sqrt(mo["N"])
        mo["CI_L"] = mo["Mean"]-1.96*mo["Std"]/np.sqrt(mo["N"])

        fig = go.Figure()
        fig.add_scatter(
            x=mo["YearMonth"].tolist()+mo["YearMonth"].tolist()[::-1],
            y=mo["CI_U"].tolist()+mo["CI_L"].tolist()[::-1],
            fill="toself", fillcolor=f"rgba(80,129,191,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        )
        fig.add_scatter(x=mo["YearMonth"],y=mo["Mean"],mode="lines+markers",
                        name=f"Mean {tm}",
                        line=dict(color=COLORS["slate"],width=2),marker=dict(size=5))
        st.plotly_chart(apply_chart_style(fig,f"Monthly Trend — {tm}",
                                           x_title="Month",y_title=tm),use_container_width=True)

        df_t["Year"] = df_t["Date"].dt.year
        if df_t["Year"].nunique() > 1:
            yoy = df_t.groupby("Year")[tm].mean().reset_index()
            fig = px.bar(yoy,x="Year",y=tm,color=tm,
                         color_continuous_scale=[[0,"#F7F8FA"],[1,COLORS["slate"]]],text=tm)
            fig.update_traces(texttemplate="%{text:.2f}",textposition="outside")
            st.plotly_chart(apply_chart_style(fig,f"Year-over-Year Average — {tm}",300),use_container_width=True)
else:
    insight_panel("Temporal trend analysis requires a valid Date column with multiple time points.", kind="info")

divider()

# ── E. Demographic breakdowns ─────────────────────────────────────────────────
section_header("E · Demographic Breakdowns")

bm = st.selectbox("Compare metric",
                   [c for c in ["Stress_Score","RMSSD_ms","SDNN_ms","LF_HF_Ratio",
                                 "HRV_Health_Index","Parasympathetic_Activity_Index"] if c in df_f.columns],
                   key="bm")

if bm in df_f.columns:
    eb1,eb2 = st.columns(2)
    with eb1:
        if "Sex" in df_f.columns:
            fig = px.violin(df_f.dropna(subset=[bm,"Sex"]),x="Sex",y=bm,color="Sex",box=True,
                            color_discrete_sequence=[COLORS["slate"],COLORS["steel"]])
            st.plotly_chart(apply_chart_style(fig,f"{bm} by Sex",320),use_container_width=True)
    with eb2:
        if "Age_Group" in df_f.columns:
            fig = px.box(df_f.dropna(subset=[bm,"Age_Group"]),x="Age_Group",y=bm,color="Age_Group",
                         category_orders={"Age_Group":AGE_GROUP_ORDER},
                         color_discrete_sequence=CHART_SEQ)
            st.plotly_chart(apply_chart_style(fig,f"{bm} by Age Group",320),use_container_width=True)

    if "BMI_Category" in df_f.columns:
        fig = px.box(df_f.dropna(subset=[bm,"BMI_Category"]),x="BMI_Category",y=bm,color="BMI_Category",
                     color_discrete_map=BMI_CATEGORY_COLORS,
                     category_orders={"BMI_Category":BMI_CATEGORY_ORDER})
        st.plotly_chart(apply_chart_style(fig,f"{bm} by BMI Category",320),use_container_width=True)

    if "Stress_Level" in df_f.columns and bm != "Stress_Score":
        fig = px.violin(df_f.dropna(subset=[bm,"Stress_Level"]),x="Stress_Level",y=bm,
                        color="Stress_Level",box=True,points=False,
                        color_discrete_map=STRESS_LEVEL_COLORS,
                        category_orders={"Stress_Level":STRESS_LEVEL_ORDER})
        st.plotly_chart(apply_chart_style(fig,f"{bm} by 5-Tier Stress Level",320),use_container_width=True)

divider()

# ── F. Population heatmap ─────────────────────────────────────────────────────
section_header("F · Population Heatmap — Age Group × Clinic")

if all(c in df_f.columns for c in ["Age_Group","Clinic_Name"]):
    hm = st.selectbox("Heatmap metric",
                       [c for c in ["Stress_Score","RMSSD_ms","LF_HF_Ratio","HRV_Health_Index"] if c in df_f.columns])
    pivot = df_f.groupby(["Age_Group","Clinic_Name"])[hm].mean().round(2).unstack()
    pivot = pivot.reindex([g for g in AGE_GROUP_ORDER if g in pivot.index])

    stress_m = {"Stress_Score","Sympathetic_Dominance_Index","LF_HF_Ratio"}
    cscale   = [[0,COLORS["green"]],[0.5,"#F7F8FA"],[1,COLORS["slate"]]] if hm in stress_m \
               else [[0,"#F7F8FA"],[0.5,COLORS["steel"]],[1,COLORS["green"]]]

    fig = px.imshow(pivot, color_continuous_scale=cscale, text_auto=".1f", aspect="auto",
                    labels=dict(x="Clinic",y="Age Group",color=hm))
    fig.update_layout(xaxis_tickangle=-15,
                      yaxis=dict(categoryorder="array",
                                 categoryarray=[g for g in AGE_GROUP_ORDER if g in pivot.index]))
    st.plotly_chart(apply_chart_style(fig,f"Mean {hm} by Age Group × Clinic",380),use_container_width=True)

divider()

# ── G. HRV tier breakdown ─────────────────────────────────────────────────────
section_header("G · HRV Tier Classification Breakdown")
insight_panel("Each patient's HRV value is classified into its clinical tier using the updated reference ranges.",
              kind="info")

tier_m = st.selectbox("HRV metric to classify", hrv_p, key="tier_m")
if tier_m in df_f.columns:
    tier_s = df_f[tier_m].dropna().apply(lambda v: classify_hrv_value(tier_m, v))
    tc     = tier_s.value_counts().reset_index(); tc.columns=["Tier","Count"]
    tc["Pct"] = (tc["Count"]/tc["Count"].sum()*100).round(1)
    tc1,tc2 = st.columns(2)
    with tc1:
        fig=px.pie(tc,names="Tier",values="Count",color_discrete_sequence=CHART_SEQ)
        st.plotly_chart(apply_chart_style(fig,f"{tier_m} — Tier Distribution",300),use_container_width=True)
    with tc2:
        fig=px.bar(tc,x="Tier",y="Count",color="Count",text="Pct",
                   color_continuous_scale=[[0,"#F7F8FA"],[1,COLORS["slate"]]])
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        st.plotly_chart(apply_chart_style(fig,f"{tier_m} — Count per Clinical Tier",300),use_container_width=True)
    st.dataframe(tc, use_container_width=True, hide_index=True)

divider()

# ── H. Downloads ──────────────────────────────────────────────────────────────
section_header("H · Download Population Summary")
dl1,dl2 = st.columns(2)
with dl1:
    st.download_button("Download Filtered Dataset (CSV)",
                       df_f.to_csv(index=False).encode("utf-8"),
                       "population_health_data.csv","text/csv",use_container_width=True)
with dl2:
    if "Stress_Level" in df_f.columns:
        sl_s = (df_f.groupby("Stress_Level")
                    .agg(Count=("Stress_Score","count"),
                         Avg_Stress=("Stress_Score","mean"),
                         Avg_RMSSD=("RMSSD_ms","mean"),
                         Avg_SDNN=("SDNN_ms","mean"),
                         Avg_LF_HF=("LF_HF_Ratio","mean"))
                    .round(2).reindex(STRESS_LEVEL_ORDER).reset_index())
        st.download_button("Download Stress Level Summary (CSV)",
                           sl_s.to_csv(index=False).encode("utf-8"),
                           "stress_level_summary.csv","text/csv",use_container_width=True)
