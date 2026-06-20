"""
pages/02_HRV_Analysis.py  —  Layer 2: HRV Physiological Analysis
Visual design: HRV Design System (utils/styles.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

from utils.data_processing import (
    HRV_COLUMNS, HRV_REFERENCE_RANGES, compute_percentile_ranks,
    STRESS_LEVEL_ORDER, STRESS_LEVEL_COLORS,
)
from utils.styles import (
    inject_css, page_header, section_header, insight_panel,
    apply_chart_style, divider, metric_row, hrv_card, COLORS, CHART_SEQ,
)

st.set_page_config(page_title="HRV Analysis · HRV", page_icon="🫀", layout="wide")
inject_css()

if "df" not in st.session_state:
    st.warning("No data loaded — upload your CSV on the Home page first.")
    st.stop()

df = st.session_state["df"]

with st.sidebar:
    st.markdown("### Filters")
    clinics = ["All"] + sorted(df["Clinic_Name"].dropna().unique().tolist()) if "Clinic_Name" in df.columns else ["All"]
    sel_clinic = st.selectbox("Clinic", clinics)
    sexes = ["All"] + sorted(df["Sex"].dropna().unique().tolist()) if "Sex" in df.columns else ["All"]
    sel_sex = st.selectbox("Sex", sexes)

df_f = df.copy()
if sel_clinic != "All" and "Clinic_Name" in df_f.columns:
    df_f = df_f[df_f["Clinic_Name"] == sel_clinic]
if sel_sex != "All" and "Sex" in df_f.columns:
    df_f = df_f[df_f["Sex"] == sel_sex]

page_header("HRV Physiological Analysis",
            "Autonomic Nervous System metrics and composite indices")

# ── ANS overview ──────────────────────────────────────────────────────────────
section_header("Autonomic Nervous System Overview")

ov_items = [
    ("Avg SDNN",       f"{df_f['SDNN_ms'].mean():.1f} ms"   if "SDNN_ms"      in df_f.columns else "—", ">50 ms Normal"),
    ("Avg RMSSD",      f"{df_f['RMSSD_ms'].mean():.1f} ms"  if "RMSSD_ms"     in df_f.columns else "—", ">40 ms Optimal"),
    ("Avg LF/HF",      f"{df_f['LF_HF_Ratio'].mean():.2f}"  if "LF_HF_Ratio"  in df_f.columns else "—", "1.1–3.0 Normal"),
    ("Avg HF Power",   f"{df_f['HF_Power_ms2'].mean():.0f}" if "HF_Power_ms2" in df_f.columns else "—", "86–3630 ms²"),
    ("Avg Stress",     f"{df_f['Stress_Score'].mean():.1f}"  if "Stress_Score" in df_f.columns else "—", "of 100"),
]
metric_row(ov_items)
divider()

# ── Helper: 4-chart panel ─────────────────────────────────────────────────────
def four_panel(col_name, label, color=None):
    color = color or COLORS["slate"]
    if col_name not in df_f.columns:
        return
    s   = df_f[col_name].dropna()
    ref = HRV_REFERENCE_RANGES.get(col_name, {})
    lo  = ref.get("healthy_low"); hi = ref.get("healthy_high")
    t1, t2, t3, t4, t5 = st.tabs(["Histogram","Boxplot","Violin","Density","Statistics"])

    with t1:
        fig = px.histogram(df_f, x=col_name, nbins=30, color_discrete_sequence=[color])
        if lo and hi:
            fig.add_vrect(x0=lo, x1=hi, fillcolor=COLORS["green"], opacity=0.08,
                          line_width=0, annotation_text="Normal range", annotation_font_size=9)
        st.plotly_chart(apply_chart_style(fig, f"{label} — Histogram", 300), use_container_width=True)

    with t2:
        if "Sex" in df_f.columns:
            fig = px.box(df_f, x="Sex", y=col_name, color="Sex", points="outliers",
                         color_discrete_sequence=[COLORS["slate"], COLORS["steel"]])
        else:
            fig = px.box(df_f, y=col_name, points="outliers", color_discrete_sequence=[color])
        st.plotly_chart(apply_chart_style(fig, f"{label} — Boxplot", 300), use_container_width=True)

    with t3:
        if "Sex" in df_f.columns:
            fig = px.violin(df_f, x="Sex", y=col_name, color="Sex", box=True,
                            color_discrete_sequence=[COLORS["slate"], COLORS["steel"]])
        else:
            fig = px.violin(df_f, y=col_name, box=True, color_discrete_sequence=[color])
        st.plotly_chart(apply_chart_style(fig, f"{label} — Violin", 300), use_container_width=True)

    with t4:
        if len(s) > 1:
            counts, bins = np.histogram(s, bins=50, density=True)
            bc = (bins[:-1]+bins[1:])/2
            sm = gaussian_filter1d(counts.astype(float), sigma=2)
            fig = go.Figure()
            fig.add_bar(x=bc, y=counts, name="Density",
                        marker_color=color, opacity=0.35, width=bins[1]-bins[0])
            fig.add_scatter(x=bc, y=sm, mode="lines", name="KDE",
                            line=dict(color=COLORS["blue"], width=2))
            st.plotly_chart(apply_chart_style(fig, f"{label} — Density", 300), use_container_width=True)

    with t5:
        q_vals = s.quantile([.1,.25,.5,.75,.9]).values
        c1,c2  = st.columns(2)
        with c1:
            st.markdown("**Summary**")
            st.dataframe(pd.DataFrame({
                "Stat": ["N","Mean","Median","Std Dev","Min","Max","IQR"],
                "Value":[f"{len(s):,}",f"{s.mean():.3f}",f"{s.median():.3f}",
                         f"{s.std():.3f}",f"{s.min():.3f}",f"{s.max():.3f}",
                         f"{s.quantile(.75)-s.quantile(.25):.3f}"]
            }), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Percentiles**")
            st.dataframe(pd.DataFrame({
                "Pct":["P10","P25","P50","P75","P90"],
                "Value":[f"{v:.2f}" for v in q_vals],
                "Unit": ref.get("unit",""),
            }), hide_index=True, use_container_width=True)
            if ref.get("clinical_note"):
                insight_panel(ref["clinical_note"], kind="info")

# ── Individual metric analysis ────────────────────────────────────────────────
section_header("Individual HRV Metric Analysis")

metrics = [
    ("Mean_RR_ms",   "Mean RR Interval",   COLORS["slate"]),
    ("SDNN_ms",      "SDNN",               COLORS["blue"]),
    ("RMSSD_ms",     "RMSSD",              COLORS["green"]),
    ("pNN50_pct",    "pNN50 (%)",           COLORS["steel"]),
    ("pNN20_pct",    "pNN20 (%)",           COLORS["slate"]),
    ("LF_Power_ms2", "LF Power (ms²)",      COLORS["blue"]),
    ("HF_Power_ms2", "HF Power (ms²)",      COLORS["green"]),
    ("LF_HF_Ratio",  "LF/HF Ratio",        COLORS["steel"]),
]
for col_n, label, color in metrics:
    with st.expander(label, expanded=(col_n == "RMSSD_ms")):
        four_panel(col_n, label, color)

# ── Parasympathetic & sympathetic analysis ────────────────────────────────────
section_header("Parasympathetic Activity Analysis")
insight_panel("RMSSD (>40 ms optimal), pNN50 (>20% optimal), and HF Power (86–3630 ms²) "
              "together quantify vagal tone. Lower values indicate parasympathetic withdrawal.",
              kind="info")

para_cols = [c for c in ["RMSSD_ms","pNN50_pct","HF_Power_ms2"] if c in df_f.columns]
if para_cols:
    color_dim = "Stress_Level" if "Stress_Level" in df_f.columns else None
    fig = px.scatter_matrix(
        df_f[para_cols+([color_dim] if color_dim else [])].dropna(),
        dimensions=para_cols, color=color_dim,
        color_discrete_map=STRESS_LEVEL_COLORS,
        category_orders={"Stress_Level": STRESS_LEVEL_ORDER},
        opacity=0.55,
    )
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(apply_chart_style(fig, "Parasympathetic Marker Pairplot", 520), use_container_width=True)

section_header("Sympathetic Dominance Analysis")
if all(c in df_f.columns for c in ["LF_Power_ms2","LF_HF_Ratio"]):
    fig = px.scatter(
        df_f.dropna(subset=["LF_Power_ms2","LF_HF_Ratio"]),
        x="LF_Power_ms2", y="LF_HF_Ratio",
        color="Stress_Score" if "Stress_Score" in df_f.columns else None,
        color_continuous_scale=[[0,COLORS["green"]],[0.5,"#F7F8FA"],[1,COLORS["slate"]]],
        size="RMSSD_ms" if "RMSSD_ms" in df_f.columns else None,
        hover_data=[c for c in ["Patient_Name","Age","Sex"] if c in df_f.columns],
        opacity=0.65,
    )
    fig.add_hline(y=3.0, line_dash="dot", line_color=COLORS["slate"],
                  annotation_text="Sympathetic Dominance (>3.0)", annotation_font_size=9)
    fig.add_hline(y=1.1, line_dash="dot", line_color=COLORS["green"],
                  annotation_text="Parasympathetic Dominance (<1.1)", annotation_font_size=9)
    fig.add_vrect(x0=193, x1=1009, fillcolor=COLORS["green"], opacity=0.04,
                  line_width=0, annotation_text="Normal LF Power", annotation_font_size=9)
    st.plotly_chart(apply_chart_style(fig, "LF Power vs LF/HF Ratio", 440,
                                       x_title="LF Power (ms²)", y_title="LF/HF Ratio"),
                    use_container_width=True)

# ── Composite indices ─────────────────────────────────────────────────────────
section_header("Composite HRV Indices")
insight_panel(
    "Four composite scores (0–100) synthesise multiple HRV markers. "
    "<strong>HRV Health Index</strong> and <strong>Parasympathetic Activity Index</strong>: "
    "higher = better autonomic health. "
    "<strong>Sympathetic Dominance Index</strong> and <strong>Stress Physiology Index</strong>: "
    "higher = greater physiological stress burden.",
    kind="info",
)

index_cols = [c for c in [
    "HRV_Health_Index","Parasympathetic_Activity_Index",
    "Sympathetic_Dominance_Index","Stress_Physiology_Index",
] if c in df_f.columns]

if index_cols:
    fig = make_subplots(rows=1, cols=len(index_cols), subplot_titles=index_cols)
    palette = [COLORS["green"], COLORS["blue"], COLORS["slate"], COLORS["steel"]]
    for i, (cn, col) in enumerate(zip(index_cols, palette), 1):
        fig.add_trace(go.Histogram(x=df_f[cn], nbinsx=25, name=cn,
                                    marker_color=col, opacity=0.8), row=1, col=i)
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(apply_chart_style(fig, "HRV Composite Index Distributions"), use_container_width=True)

    st.markdown("**Patient Rankings**")
    df_rk = compute_percentile_ranks(df_f, index_cols)
    rk_cols = [c for c in ["Patient_Name","Age","Sex","Clinic_Name"]
                if c in df_rk.columns] + index_cols + \
              [f"{c}_Pct_Rank" for c in index_cols if f"{c}_Pct_Rank" in df_rk.columns]
    st.dataframe(df_rk[rk_cols].sort_values("HRV_Health_Index",ascending=False).head(50),
                 use_container_width=True, hide_index=True)
    st.download_button("Download HRV Index Rankings (CSV)",
                       df_rk[rk_cols].to_csv(index=False).encode("utf-8"),
                       "hrv_index_rankings.csv", "text/csv")
