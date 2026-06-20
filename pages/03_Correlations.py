"""
pages/03_Correlations.py  —  Layer 3: Relationship Discovery
Visual design: HRV Design System (utils/styles.py)
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_processing import HRV_COLUMNS, STRESS_LEVEL_ORDER, STRESS_LEVEL_COLORS, AGE_GROUP_ORDER, BMI_CATEGORY_ORDER
from utils.statistical_utils import (correlation_analysis, get_top_correlations,
    perform_ttest, perform_mannwhitney, linear_regression_stats, run_subgroup_comparisons)
from utils.styles import (inject_css, page_header, section_header, insight_panel,
    apply_chart_style, divider, COLORS, CHART_SEQ, CORR_SCALE)

st.set_page_config(page_title="Correlations · HRV", page_icon="📈", layout="wide")
inject_css()

if "df" not in st.session_state:
    st.warning("No data loaded — upload your CSV on the Home page first.")
    st.stop()

df = st.session_state["df"]

with st.sidebar:
    st.markdown("### Options")
    corr_method = st.radio("Correlation method", ["pearson","spearman"])
    sel_clinic = "All"
    if "Clinic_Name" in df.columns:
        sel_clinic = st.selectbox("Clinic", ["All"]+sorted(df["Clinic_Name"].dropna().unique().tolist()))

df_f = df.copy()
if sel_clinic != "All" and "Clinic_Name" in df_f.columns:
    df_f = df_f[df_f["Clinic_Name"] == sel_clinic]

corr_cols = [c for c in ["Age","Weight_kg","BMI","Heart_Rate_bpm"]+HRV_COLUMNS+["Stress_Score"] if c in df_f.columns]

page_header("Relationship Discovery & Statistical Analysis",
            "Correlations, regression lines, and hypothesis tests across HRV metrics")

# ── A. Correlation matrices ───────────────────────────────────────────────────
section_header("A · Correlation Matrices")

tp, ts = st.tabs(["Pearson","Spearman"])
for tab, method in [(tp,"pearson"),(ts,"spearman")]:
    with tab:
        corr_mat, p_mat = correlation_analysis(df_f, corr_cols, method=method)
        fig = px.imshow(corr_mat, color_continuous_scale=CORR_SCALE,
                        zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
        fig = apply_chart_style(fig, f"{method.capitalize()} Correlation Matrix", 540)
        fig.update_coloraxes(colorbar_title="r")
        st.plotly_chart(fig, use_container_width=True)

        top = get_top_correlations(corr_mat, p_mat, n=20)
        st.dataframe(top, use_container_width=True, hide_index=True)
        st.download_button(f"Download {method.capitalize()} Table",
                           top.to_csv(index=False).encode("utf-8"),
                           f"corr_{method}.csv", "text/csv")

divider()

# ── B. Scatter plots ──────────────────────────────────────────────────────────
section_header("B · Scatter Plots with Regression Lines")

def scatter_reg(x_col, y_col, color_col=None, title=None):
    cols = list({x_col, y_col} | ({color_col} if color_col and color_col in df_f.columns else set()))
    if "Patient_Name" in df_f.columns: cols.append("Patient_Name")
    plot_df = df_f[cols].dropna(subset=[x_col, y_col])
    if len(plot_df) < 5: return
    reg = linear_regression_stats(plot_df[x_col], plot_df[y_col])
    x_r = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
    y_r = reg["slope"]*x_r + reg["intercept"]

    kw = dict(color_discrete_map=STRESS_LEVEL_COLORS,
              category_orders={"Stress_Level": STRESS_LEVEL_ORDER}) \
         if color_col == "Stress_Level" else dict(color_continuous_scale=[[0,COLORS["green"]],[1,COLORS["slate"]]])

    fig = px.scatter(plot_df, x=x_col, y=y_col,
                     color=color_col if color_col in plot_df.columns else None,
                     hover_data=["Patient_Name"] if "Patient_Name" in plot_df.columns else None,
                     opacity=0.55, **kw)
    lbl = f"R²={reg['r_squared']:.3f}  p={'<0.001' if reg['p_value']<0.001 else f'{reg['p_value']:.3f}'}"
    fig.add_scatter(x=x_r, y=y_r, mode="lines", name=f"OLS ({lbl})",
                    line=dict(color=COLORS["blue"], width=2,
                              dash="solid" if reg["significant"] else "dash"))
    st.plotly_chart(apply_chart_style(fig, title or f"{x_col} vs {y_col}", 340), use_container_width=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R²", f"{reg['r_squared']:.4f}")
    c2.metric("Slope", f"{reg['slope']:.4f}")
    c3.metric("p-value", f"{reg['p_value']:.4f}")
    c4.metric("Significant", "Yes" if reg["significant"] else "No")
    st.markdown('<div class="hrv-divider"></div>', unsafe_allow_html=True)

pairs = [
    ("Age","SDNN_ms","Stress_Score","Age vs SDNN"),
    ("Age","RMSSD_ms","Stress_Score","Age vs RMSSD"),
    ("Age","HF_Power_ms2","Stress_Score","Age vs HF Power"),
    ("Age","LF_HF_Ratio","Stress_Score","Age vs LF/HF Ratio"),
    ("Stress_Score","RMSSD_ms","Age","Stress Score vs RMSSD"),
    ("Stress_Score","HF_Power_ms2","Age","Stress Score vs HF Power"),
    ("Stress_Score","LF_HF_Ratio","Age","Stress Score vs LF/HF Ratio"),
    ("BMI","RMSSD_ms","Sex","BMI vs RMSSD"),
    ("BMI","SDNN_ms","Sex","BMI vs SDNN"),
    ("BMI","LF_HF_Ratio","Sex","BMI vs LF/HF Ratio"),
]
for i in range(0, len(pairs), 2):
    c1, c2 = st.columns(2)
    for cw, p in zip([c1,c2], pairs[i:i+2]):
        if p[0] in df_f.columns and p[1] in df_f.columns:
            with cw: scatter_reg(*p)

# ── C. Hypothesis tests ───────────────────────────────────────────────────────
section_header("C · Statistical Hypothesis Tests")

test_cols = [c for c in HRV_COLUMNS+["Stress_Score"] if c in df_f.columns]

if "Sex" in df_f.columns and df_f["Sex"].nunique() >= 2:
    st.markdown("**Sex-based Comparisons**")
    sexes = df_f["Sex"].dropna().unique().tolist()
    if len(sexes) >= 2:
        g1,g2 = sexes[0],sexes[1]
        rows = []
        for c in test_cols:
            t = perform_ttest(df_f,c,"Sex",g1,g2)
            m = perform_mannwhitney(df_f,c,"Sex",g1,g2)
            if "error" not in t:
                rows.append({"Variable":c, f"Mean({g1})":t.get("mean1"),
                             f"Mean({g2})":t.get("mean2"),
                             "t p":t.get("p_value"), "t Sig":"✓" if t.get("significant") else "—",
                             "MW p":m.get("p_value"), "MW Sig":"✓" if m.get("significant") else "—",
                             "Cohen d":t.get("effect_size"), "Effect":t.get("effect_label")})
        if rows:
            rdf = pd.DataFrame(rows)
            st.dataframe(rdf, use_container_width=True, hide_index=True)
            sig = rdf["t Sig"].str.contains("✓").sum()
            insight_panel(f"{sig} of {len(rows)} variables show statistically significant "
                          f"sex differences (p < 0.05).", kind="info")

group_options = {k:v for k,v in {
    "Age Group":"Age_Group","BMI Category":"BMI_Category",
    "Stress Level":"Stress_Level","Clinic":"Clinic_Name"
}.items() if v in df_f.columns}

if group_options:
    st.markdown("**Multi-Group Comparisons**")
    chosen_label = st.selectbox("Group by", list(group_options.keys()))
    chosen_col   = group_options[chosen_label]
    orders = {"Age_Group":AGE_GROUP_ORDER,"BMI_Category":BMI_CATEGORY_ORDER,"Stress_Level":STRESS_LEVEL_ORDER}
    multi_df = run_subgroup_comparisons(df_f, test_cols, chosen_col)
    if not multi_df.empty:
        st.dataframe(multi_df, use_container_width=True, hide_index=True)
        sig_vars = multi_df[multi_df["Significant"]=="✅ Yes"]["Variable"].tolist()
        if sig_vars:
            insight_panel(f"Significant differences by {chosen_label}: {', '.join(sig_vars)}", kind="success")
            for sv in sig_vars[:4]:
                ord_ = orders.get(chosen_col)
                fig = px.box(df_f.dropna(subset=[sv,chosen_col]),
                             x=chosen_col, y=sv, color=chosen_col,
                             color_discrete_map=STRESS_LEVEL_COLORS if chosen_col=="Stress_Level" else None,
                             category_orders={chosen_col:ord_} if ord_ else None,
                             points="outliers")
                st.plotly_chart(apply_chart_style(fig, f"{sv} by {chosen_label}", 340), use_container_width=True)

# ── D. Pairplot ───────────────────────────────────────────────────────────────
section_header("D · Multi-Dimensional Pairplot")
pp_cols = [c for c in ["Stress_Score","RMSSD_ms","SDNN_ms","LF_HF_Ratio","HF_Power_ms2","Age"] if c in df_f.columns]
if len(pp_cols) >= 3:
    color_dim = "Stress_Level" if "Stress_Level" in df_f.columns else None
    fig = px.scatter_matrix(
        df_f[pp_cols+([color_dim] if color_dim else [])].dropna(),
        dimensions=pp_cols, color=color_dim,
        color_discrete_map=STRESS_LEVEL_COLORS,
        category_orders={"Stress_Level":STRESS_LEVEL_ORDER},
        opacity=0.45,
    )
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(apply_chart_style(fig, "Pairplot — Key HRV & Stress Variables", 640), use_container_width=True)
