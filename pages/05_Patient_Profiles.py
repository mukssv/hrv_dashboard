"""
pages/06_Patient_Profiles.py  —  Advanced: Individual Patient Profiles
Visual design: HRV Design System (utils/styles.py)
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_processing import (
    HRV_COLUMNS, HRV_REFERENCE_RANGES, compute_percentile_ranks,
    STRESS_LEVEL_ORDER, STRESS_LEVEL_COLORS, BMI_CATEGORY_ORDER, classify_hrv_value,
)
from utils.styles import (
    inject_css, page_header, section_header, insight_panel,
    apply_chart_style, divider, metric_row, hrv_card, COLORS, CHART_SEQ,
)

st.set_page_config(page_title="Patient Profiles · HRV", page_icon="👤", layout="wide")
inject_css()

if "df" not in st.session_state:
    st.warning("No data loaded — upload your CSV on the Home page first.")
    st.stop()

df = st.session_state["df"]

rank_cols = [c for c in HRV_COLUMNS + [
    "Stress_Score","HRV_Health_Index","Parasympathetic_Activity_Index",
    "Sympathetic_Dominance_Index","Stress_Physiology_Index",
] if c in df.columns]
df_ranked = compute_percentile_ranks(df, rank_cols)

page_header("Patient Profiles", "Search · HRV benchmarking · risk indicators · longitudinal trends")

# ── Search panel ──────────────────────────────────────────────────────────────
section_header("Patient Search")

if "Patient_Name" not in df.columns:
    st.error("Patient_Name column not found in dataset.")
    st.stop()

sc1, sc2 = st.columns([2, 1])
with sc1:
    all_names    = sorted(df["Patient_Name"].dropna().unique().tolist())
    selected_name = st.selectbox("Search by patient name", ["— Select a patient —"] + all_names)
with sc2:
    clinic_filter = "All Clinics"
    if "Clinic_Name" in df.columns:
        clinic_filter = st.selectbox("Filter by clinic",
                                     ["All Clinics"] + sorted(df["Clinic_Name"].dropna().unique().tolist()))

df_search = df_ranked.copy()
if clinic_filter != "All Clinics" and "Clinic_Name" in df.columns:
    df_search = df_search[df_search["Clinic_Name"] == clinic_filter]

if selected_name == "— Select a patient —":
    insight_panel("Select a patient from the dropdown above to view their full profile.", kind="info")
    section_header("Cohort Summary")
    metric_row([
        ("Total Patients",  f"{df['Patient_Name'].nunique():,}", ""),
        ("Avg Stress Score",f"{df['Stress_Score'].mean():.1f}"  if "Stress_Score" in df.columns else "—",""),
        ("Avg RMSSD",       f"{df['RMSSD_ms'].mean():.1f} ms"   if "RMSSD_ms"     in df.columns else "—",""),
        ("Avg LF/HF",       f"{df['LF_HF_Ratio'].mean():.2f}"   if "LF_HF_Ratio"  in df.columns else "—",""),
    ])
    if "Stress_Level" in df.columns:
        sl = df["Stress_Level"].value_counts().reindex(STRESS_LEVEL_ORDER).dropna().reset_index()
        sl.columns = ["Stress Level","Count"]
        fig = px.bar(sl, x="Stress Level", y="Count",
                     color="Stress Level", color_discrete_map=STRESS_LEVEL_COLORS,
                     category_orders={"Stress Level":STRESS_LEVEL_ORDER})
        st.plotly_chart(apply_chart_style(fig,"Cohort Stress Level Distribution",320),use_container_width=True)
    st.stop()

# ── Patient record ────────────────────────────────────────────────────────────
recs = df_search[df_search["Patient_Name"] == selected_name].copy()
if len(recs) == 0:
    st.error(f"No records found for '{selected_name}' with the selected clinic filter.")
    st.stop()

if "Date" in recs.columns:
    recs = recs.sort_values("Date", ascending=False)
    report_labels = recs["Date"].apply(lambda d: d.strftime("%d %b %Y") if pd.notna(d) else "Unknown").tolist()
else:
    report_labels = [f"Report {i+1}" for i in range(len(recs))]

sel_idx = st.selectbox(f"Report ({len(recs)} found)", range(len(recs)),
                        format_func=lambda i: report_labels[i])
pt = recs.iloc[sel_idx]

def sv(col, fmt="{}", default="—"):
    val = pt.get(col)
    if val is None or (isinstance(val, float) and np.isnan(val)): return default
    try: return fmt.format(val)
    except: return str(val)

def stress_color(score):
    if score <= 40:   return STRESS_LEVEL_COLORS["Normal Stress"]
    elif score <= 55: return STRESS_LEVEL_COLORS["Mild Stress"]
    elif score <= 70: return STRESS_LEVEL_COLORS["Moderate Stress"]
    elif score <= 85: return STRESS_LEVEL_COLORS["Severe Stress"]
    return STRESS_LEVEL_COLORS["Extremely Severe Stress"]

stress_val   = float(pt.get("Stress_Score", 0) or 0)
stress_lbl   = pt.get("Stress_Level", "—")
stress_col   = stress_color(stress_val)
report_date  = pt["Date"].strftime("%d %b %Y") if "Date" in pt.index and pd.notna(pt.get("Date")) else "—"

# ── Profile header ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:#364A69;border-radius:16px;padding:1.8rem 2rem;
            margin-bottom:1.5rem;display:flex;justify-content:space-between;
            align-items:center;flex-wrap:wrap;gap:1rem;">
    <div>
        <div style="font-size:1.7rem;font-weight:700;color:#FFFFFF;
                    letter-spacing:-0.02em;">{sv('Patient_Name')}</div>
        <div style="font-size:0.85rem;color:rgba(255,255,255,0.65);margin-top:0.4rem;">
            {sv('Age','Age: {} yrs')} &nbsp;·&nbsp; {sv('Sex')}
            &nbsp;·&nbsp; {sv('Clinic_Name')} &nbsp;·&nbsp; {report_date}
        </div>
    </div>
    <div style="background:{stress_col};border-radius:12px;padding:0.9rem 1.6rem;text-align:center;">
        <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.1em;color:rgba(255,255,255,0.85);">Stress Score</div>
        <div style="font-size:2.4rem;font-weight:700;color:#FFFFFF;line-height:1.1;">
            {stress_val:.0f}
        </div>
        <div style="font-size:0.78rem;color:rgba(255,255,255,0.9);font-weight:500;">{stress_lbl}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Demographics ──────────────────────────────────────────────────────────────
section_header("Demographics & Anthropometrics")
bmi_v = pt.get("BMI", np.nan); bmi_c = pt.get("BMI_Category","—")
metric_row([
    ("Age",    sv("Age","{:.0f} yrs"), ""),
    ("Sex",    sv("Sex"),              ""),
    ("Height", sv("Height_cm","{:.1f} cm"), ""),
    ("Weight", sv("Weight_kg","{:.1f} kg"), ""),
    ("BMI",    f"{bmi_v:.1f}" if not pd.isna(bmi_v) else "—", bmi_c),
])

divider()

# ── HRV benchmarking ──────────────────────────────────────────────────────────
section_header("HRV Metrics — Benchmarked Against Cohort")
insight_panel("Green = within updated normal range · Amber = moderate/reduced tier · Red = outside normal range",
              kind="info")

hrv_p = [c for c in HRV_COLUMNS if c in df.columns]
bench = []
for cn in hrv_p:
    ref  = HRV_REFERENCE_RANGES.get(cn, {})
    val  = pt.get(cn, np.nan)
    if pd.isna(val): continue
    lo, hi      = ref.get("healthy_low",-np.inf), ref.get("healthy_high",np.inf)
    in_range    = lo <= val <= hi
    tier_lbl    = classify_hrv_value(cn, float(val))
    status      = "✓" if in_range else ("~" if ("Moderate" in tier_lbl or "Reduced" in tier_lbl) else "✗")
    pct         = pt.get(f"{cn}_Pct_Rank", np.nan)
    bench.append({
        "Metric":          ref.get("description", cn),
        "Value":           round(val, 2),
        "Unit":            ref.get("unit",""),
        "Clinical Tier":   tier_lbl,
        "Status":          status,
        "Cohort Mean":     round(df[cn].mean(), 2),
        "Cohort SD":       round(df[cn].std(), 2),
        "Percentile":      f"{pct:.0f}th" if not pd.isna(pct) else "—",
    })

st.dataframe(pd.DataFrame(bench), use_container_width=True, hide_index=True)

# ── Radar chart ───────────────────────────────────────────────────────────────
section_header("Radar Profile — Patient vs Cohort")
radar_m = [c for c in ["SDNN_ms","RMSSD_ms","pNN50_pct","HF_Power_ms2",
                         "HRV_Health_Index","Parasympathetic_Activity_Index"]
           if c in df.columns and not pd.isna(pt.get(c))]

if len(radar_m) >= 3:
    cohort_m = df[radar_m].mean()
    pt_vals  = pd.Series({c: float(pt.get(c,0) or 0) for c in radar_m})
    maxv     = df[radar_m].max()
    np_r     = (pt_vals/maxv).clip(0,1)
    nc_r     = (cohort_m/maxv).clip(0,1)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=np_r.tolist()+[np_r.iloc[0]], theta=radar_m+[radar_m[0]],
        fill="toself", name=selected_name,
        line=dict(color=COLORS["blue"],width=2),
        fillcolor="rgba(80,129,191,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=nc_r.tolist()+[nc_r.iloc[0]], theta=radar_m+[radar_m[0]],
        fill="toself", name="Cohort Average",
        line=dict(color=COLORS["border"],width=2,dash="dot"),
        fillcolor="rgba(184,193,208,0.08)",
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1],
                                                   tickfont=dict(color=COLORS["text_muted"],size=9))),
                      height=420, legend=dict(x=0.8,y=1.1))
    st.plotly_chart(apply_chart_style(fig, f"{selected_name} vs Cohort Average (normalised)"),
                    use_container_width=True)

# ── Composite index percentiles ───────────────────────────────────────────────
section_header("Composite Index Percentile Rankings")
idx_cols = [c for c in ["HRV_Health_Index","Parasympathetic_Activity_Index",
                          "Sympathetic_Dominance_Index","Stress_Physiology_Index"]
            if c in df.columns]

if idx_cols:
    ic = st.columns(len(idx_cols))
    for cw, col_n in zip(ic, idx_cols):
        val = pt.get(col_n, np.nan); pct = pt.get(f"{col_n}_Pct_Rank", np.nan)
        if pd.isna(val): continue
        color = COLORS["green"] if ("Health" in col_n or "Parasympathetic" in col_n) else COLORS["slate"]
        with cw:
            st.markdown(f"""
            <div style="background:#FFFFFF;border-radius:12px;border:1px solid #E8EBF0;
                        padding:1rem;text-align:center;border-top:3px solid {color};">
                <div style="font-size:0.68rem;font-weight:700;color:#9BA8B8;
                            text-transform:uppercase;letter-spacing:0.08em;">
                    {col_n.replace('_',' ')}</div>
                <div style="font-size:1.8rem;font-weight:700;color:#1C2B3A;margin:0.3rem 0;">
                    {val:.1f}</div>
                <div style="font-size:0.78rem;color:{color};font-weight:600;">
                    {"—" if pd.isna(pct) else f"{pct:.0f}th percentile"}</div>
            </div>""", unsafe_allow_html=True)

divider()

# ── Risk indicators ───────────────────────────────────────────────────────────
section_header("Risk Indicators")

risk   = []
notice = []

if stress_val > 85:
    risk.append(("Extremely Severe Stress", f"Score {stress_val:.0f} > 85 — critical autonomic dysregulation"))
elif stress_val > 70:
    risk.append(("Severe Stress", f"Score {stress_val:.0f} in 71–85 range — marked sympathetic dominance"))
elif stress_val > 55:
    notice.append(("Moderate Stress", f"Score {stress_val:.0f} in 56–70 range"))
elif stress_val > 40:
    notice.append(("Mild Stress", f"Score {stress_val:.0f} in 41–55 range — early sympathetic activation"))

for col_n, thresh_lo, thresh_mod, lbl_lo, lbl_mod in [
    ("RMSSD_ms",   15.7, 40.0,  "Very Low RMSSD — vagal withdrawal",    "Moderate RMSSD"),
    ("SDNN_ms",    27.0, 50.0,  "Very Low SDNN — increased mortality risk","Moderate SDNN"),
    ("pNN50_pct",   1.0, 20.0,  "Abnormal pNN50 — severe vagal withdrawal","Reduced pNN50"),
]:
    v = pt.get(col_n, np.nan)
    if not pd.isna(v):
        if v < thresh_lo:   risk.append((lbl_lo,   f"{v:.2f} < {thresh_lo} ({col_n})"))
        elif v < thresh_mod: notice.append((lbl_mod, f"{v:.2f} in moderate range ({col_n})"))

lf = pt.get("LF_HF_Ratio", np.nan)
if not pd.isna(lf):
    if lf > 3.0: risk.append(("Sympathetic Dominance", f"LF/HF = {lf:.2f} > 3.0 threshold"))
    elif lf < 1.1: notice.append(("Parasympathetic Dominance", f"LF/HF = {lf:.2f} < 1.1 threshold"))

hf = pt.get("HF_Power_ms2", np.nan)
if not pd.isna(hf) and hf < 86:
    risk.append(("Reduced HF Power", f"HF Power = {hf:.1f} ms² < 86 ms²"))

bmi_cat = str(pt.get("BMI_Category",""))
if "Obesity Class 2" in bmi_cat or "Extreme" in bmi_cat:
    risk.append((bmi_cat, f"BMI {bmi_v:.1f} — associated with reduced HRV"))
elif "Obesity Class 1" in bmi_cat:
    notice.append((bmi_cat, f"BMI {bmi_v:.1f} — moderate obesity"))

hhi = pt.get("HRV_Health_Index", np.nan)
if not pd.isna(hhi):
    if hhi < 30:   risk.append(("Low HRV Health Index", f"{hhi:.1f}/100"))
    elif hhi < 50: notice.append(("Below-Average HRV Health Index", f"{hhi:.1f}/100"))

def _flag(lbl, detail, kind):
    col = {"risk":"#D4674A","notice":"#E8A84B"}[kind]
    bg  = {"risk":"rgba(212,103,74,0.05)","notice":"rgba(232,168,75,0.05)"}[kind]
    st.markdown(f"""
    <div style="border-left:3px solid {col};background:{bg};
                border-radius:0 10px 10px 0;padding:0.65rem 1rem;
                margin-bottom:0.4rem;font-size:0.83rem;color:#2D3E56;">
        <strong style="color:{col};">{lbl}</strong> &nbsp;—&nbsp; {detail}
    </div>""", unsafe_allow_html=True)

if risk:
    for l,d in risk: _flag(l,d,"risk")
if notice:
    for l,d in notice: _flag(l,d,"notice")
if not risk and not notice:
    insight_panel("No significant risk indicators detected based on updated HRV reference thresholds.",
                  kind="success")

# ── Cluster assignment ────────────────────────────────────────────────────────
if "km_labels" in st.session_state and "km_profile" in st.session_state:
    section_header("Cluster Assignment")
    pt_idx  = recs.iloc[sel_idx].name
    km_lbls = st.session_state["km_labels"]
    km_prof = st.session_state["km_profile"]
    if pt_idx < len(km_lbls):
        cid = km_lbls[pt_idx]
        clbl = km_prof.loc[cid,"Clinical Label"] if "Clinical Label" in km_prof.columns else f"Cluster {cid}"
        insight_panel(f"Cluster {cid} — {clbl}", kind="info")
        st.dataframe(km_prof.loc[[cid]], use_container_width=True)

# ── Longitudinal trend ────────────────────────────────────────────────────────
if len(recs) > 1:
    with st.expander(f"All {len(recs)} reports for {selected_name}"):
        show = [c for c in ["Date","Clinic_Name","Stress_Score","Stress_Level",
                             "SDNN_ms","RMSSD_ms","LF_HF_Ratio","HRV_Health_Index"] if c in recs.columns]
        st.dataframe(recs[show].reset_index(drop=True), use_container_width=True, hide_index=True)

        if "Date" in recs.columns and "Stress_Score" in recs.columns:
            td = recs[["Date","Stress_Score","RMSSD_ms"]].dropna(subset=["Date"]).sort_values("Date")
            if len(td) > 1:
                fig = go.Figure()
                fig.add_scatter(x=td["Date"], y=td["Stress_Score"], mode="lines+markers",
                                name="Stress Score",
                                line=dict(color=COLORS["slate"],width=2),
                                marker=dict(size=7))
                if "RMSSD_ms" in td.columns:
                    fig.add_scatter(x=td["Date"], y=td["RMSSD_ms"], mode="lines+markers",
                                    name="RMSSD (ms)", yaxis="y2",
                                    line=dict(color=COLORS["green"],width=2,dash="dot"),
                                    marker=dict(size=5))
                for t0,t1,c in [(0,40,STRESS_LEVEL_COLORS["Normal Stress"]),
                                  (40,55,STRESS_LEVEL_COLORS["Mild Stress"]),
                                  (55,70,STRESS_LEVEL_COLORS["Moderate Stress"]),
                                  (70,85,STRESS_LEVEL_COLORS["Severe Stress"]),
                                  (85,100,STRESS_LEVEL_COLORS["Extremely Severe Stress"])]:
                    fig.add_hrect(y0=t0,y1=t1,fillcolor=c,opacity=0.05,line_width=0)
                fig.update_layout(yaxis2=dict(overlaying="y",side="right",title="RMSSD (ms)"),
                                  legend=dict(x=0.01,y=0.99))
                st.plotly_chart(apply_chart_style(fig, f"Longitudinal Trend — {selected_name}",
                                                   x_title="Report Date", y_title="Stress Score"),
                                use_container_width=True)
