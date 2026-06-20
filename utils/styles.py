"""
utils/styles.py
===============
Master Design System for HRV Clinical Insights Dashboard
----------------------------------------------------------
Single source of truth for:
  • Color palette
  • Global CSS injection (Streamlit overrides + custom classes)
  • Plotly figure template
  • Reusable HTML component builders
  • Chart color sequences and scales

Usage in any page:
    from utils.styles import inject_css, page_header, metric_row, \
                              section_header, insight_panel, apply_chart_style, \
                              COLORS, CHART_SEQ, CLUSTER_COLORS, CORR_SCALE
    inject_css()   # ← call immediately after set_page_config
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "white":         "#FFFFFF",
    "slate":         "#364A69",   # Deep Slate Blue — primary, titles, sidebar
    "green":         "#7CB264",   # Soft Green — positive, recovery, high HRV
    "blue":          "#5081BF",   # Clinical Blue — interactive, links, accents
    "steel":         "#5B91B0",   # Muted Steel Blue — secondary information
    "bg":            "#F7F8FA",   # Page background
    "surface":       "#FFFFFF",   # Card / panel background
    "border":        "#E8EBF0",   # Thin dividers
    "text_primary":  "#1C2B3A",   # Main body text
    "text_muted":    "#6B7A8D",   # Captions, labels
    "text_light":    "#9BA8B8",   # Placeholder, disabled
    "shadow":        "rgba(54,74,105,0.07)",
}

# Chart color sequence (primary → secondary → green → steel → ...)
CHART_SEQ = [
    "#364A69", "#5081BF", "#7CB264", "#5B91B0",
    "#8BA4C2", "#A3C891", "#7BAAC4", "#4E6A8F",
]

# Cluster colours — editorial, not rainbow
CLUSTER_COLORS = {
    "0": "#364A69", "1": "#5081BF",
    "2": "#7CB264", "3": "#5B91B0",
    "4": "#8BA4C2", "5": "#4E6A8F",
}

# Correlation heatmap — diverging Blue → White → Green
CORR_SCALE = [
    [0.0,  "#5081BF"],
    [0.25, "#A8C4E0"],
    [0.5,  "#FFFFFF"],
    [0.75, "#B8D9A8"],
    [1.0,  "#7CB264"],
]

# Stress tier colours (kept from data_processing for risk context)
STRESS_COLORS = {
    "Normal Stress":           "#7CB264",
    "Mild Stress":             "#C8D96E",
    "Moderate Stress":         "#E8A84B",
    "Severe Stress":           "#D4674A",
    "Extremely Severe Stress": "#8B3A52",
}


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Fonts ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown,
.stTextInput, .stSelectbox, .stMultiSelect,
button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', sans-serif !important;
}

/* ── App background ──────────────────────────────────────────────────────── */
.stApp {
    background: #F7F8FA !important;
}

/* ── Main content area ───────────────────────────────────────────────────── */
.main .block-container {
    padding: 2rem 2.8rem 3rem 2.8rem !important;
    max-width: 1440px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #364A69 !important;
    border-right: none !important;
    box-shadow: 2px 0 12px rgba(54,74,105,0.12) !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.9) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.5rem !important;
    opacity: 0.65 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 0.8rem 0 !important;
}
[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: white !important;
}
[data-testid="stSidebar"] .stSlider > div {
    color: rgba(255,255,255,0.8) !important;
}
[data-testid="stSidebar"] .stCheckbox label {
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] [data-testid="stCaption"] {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.72rem !important;
}

/* ── Top header bar ──────────────────────────────────────────────────────── */
[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #364A69 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: background 0.18s ease, box-shadow 0.18s ease !important;
    box-shadow: 0 1px 4px rgba(54,74,105,0.18) !important;
}
.stButton > button:hover {
    background: #5081BF !important;
    box-shadow: 0 3px 10px rgba(80,129,191,0.28) !important;
}
.stButton > button:active {
    background: #2D3E56 !important;
}
.stDownloadButton > button {
    background: transparent !important;
    color: #364A69 !important;
    border: 1.5px solid #364A69 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
}
.stDownloadButton > button:hover {
    background: #364A69 !important;
    color: white !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: transparent !important;
    border-bottom: 2px solid #E8EBF0 !important;
    padding-bottom: 0 !important;
    margin-bottom: 1.2rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.5rem 1.1rem !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #6B7A8D !important;
    margin-bottom: -2px !important;
    transition: color 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #364A69 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #364A69 !important;
    border-bottom: 2px solid #5081BF !important;
    font-weight: 600 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 0 !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.3rem !important;
    border: 1px solid #E8EBF0 !important;
    box-shadow: 0 1px 6px rgba(54,74,105,0.06) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    color: #6B7A8D !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #1C2B3A !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E8EBF0 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #364A69 !important;
    padding: 0.75rem 1rem !important;
}

/* ── DataFrames / Tables ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #E8EBF0 !important;
}
.dataframe th {
    background: #F7F8FA !important;
    color: #364A69 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    border-bottom: 2px solid #E8EBF0 !important;
    padding: 0.6rem 0.8rem !important;
}
.dataframe td {
    font-size: 0.82rem !important;
    color: #1C2B3A !important;
    padding: 0.55rem 0.8rem !important;
    border-bottom: 1px solid #F0F2F5 !important;
}
.dataframe tr:hover td {
    background: #F7F9FC !important;
}

/* ── Selectbox / Multiselect ─────────────────────────────────────────────── */
.stSelectbox > label,
.stMultiSelect > label,
.stSlider > label,
.stRadio > label,
.stCheckbox > label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #364A69 !important;
    letter-spacing: 0.02em !important;
}
.stSelectbox > div > div,
.stMultiSelect > div > div {
    border: 1.5px solid #E8EBF0 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    font-size: 0.84rem !important;
    transition: border-color 0.15s ease !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: #5081BF !important;
    box-shadow: 0 0 0 3px rgba(80,129,191,0.12) !important;
}

/* ── Slider ──────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background: #364A69 !important;
    border: 2px solid #5081BF !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #5081BF !important;
}

/* ── Radio ───────────────────────────────────────────────────────────────── */
.stRadio > div {
    gap: 0.6rem !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.82rem !important;
    color: #364A69 !important;
}

/* ── Info / Success / Warning / Error boxes ──────────────────────────────── */
.stAlert {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    font-size: 0.84rem !important;
}
[data-testid="stInfo"] {
    background: rgba(80,129,191,0.07) !important;
    border-color: #5081BF !important;
    color: #2D3E56 !important;
}
[data-testid="stSuccess"] {
    background: rgba(124,178,100,0.08) !important;
    border-color: #7CB264 !important;
    color: #2D4A1E !important;
}
[data-testid="stWarning"] {
    background: rgba(232,168,75,0.08) !important;
    border-color: #E8A84B !important;
}
[data-testid="stError"] {
    background: rgba(212,103,74,0.08) !important;
    border-color: #D4674A !important;
}

/* ── Caption / small text ────────────────────────────────────────────────── */
[data-testid="stCaption"] {
    font-size: 0.76rem !important;
    color: #9BA8B8 !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #5081BF !important;
}

/* ── Plotly charts — remove default Streamlit container border ───────────── */
[data-testid="stPlotlyChart"] {
    background: #FFFFFF !important;
    border-radius: 14px !important;
    border: 1px solid #E8EBF0 !important;
    padding: 0.2rem !important;
    box-shadow: 0 1px 6px rgba(54,74,105,0.05) !important;
}

/* ── Custom component classes ────────────────────────────────────────────── */
.hrv-page-header {
    margin-bottom: 1.8rem;
}
.hrv-page-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #1C2B3A;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.hrv-page-header p {
    font-size: 0.9rem;
    color: #6B7A8D;
    margin: 0;
    font-weight: 400;
    line-height: 1.5;
}
.hrv-divider {
    height: 1px;
    background: #E8EBF0;
    border: none;
    margin: 1.5rem 0;
}
.hrv-section {
    font-size: 0.7rem;
    font-weight: 700;
    color: #9BA8B8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 2rem 0 0.8rem 0;
}
.hrv-card {
    background: #FFFFFF;
    border-radius: 14px;
    border: 1px solid #E8EBF0;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 1px 6px rgba(54,74,105,0.06);
    margin-bottom: 1rem;
}
.hrv-metric {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.hrv-metric-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #9BA8B8;
    text-transform: uppercase;
    letter-spacing: 0.09em;
}
.hrv-metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1C2B3A;
    line-height: 1.1;
}
.hrv-metric-sub {
    font-size: 0.74rem;
    color: #9BA8B8;
    margin-top: 0.1rem;
}
.hrv-insight {
    border-left: 3px solid #5081BF;
    background: rgba(80,129,191,0.05);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 1rem 0;
    font-size: 0.85rem;
    color: #2D3E56;
    line-height: 1.6;
}
.hrv-insight strong {
    color: #364A69;
}
.hrv-insight-green {
    border-left-color: #7CB264;
    background: rgba(124,178,100,0.05);
}
.hrv-insight-amber {
    border-left-color: #E8A84B;
    background: rgba(232,168,75,0.05);
}
.hrv-insight-red {
    border-left-color: #D4674A;
    background: rgba(212,103,74,0.05);
}
.hrv-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
}
.hrv-stat-item {
    background: #F7F8FA;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    border: 1px solid #E8EBF0;
}
.hrv-stat-item .lbl {
    font-size: 0.68rem;
    font-weight: 700;
    color: #9BA8B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.hrv-stat-item .val {
    font-size: 1.25rem;
    font-weight: 700;
    color: #364A69;
    margin-top: 0.15rem;
}
.hrv-badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.hrv-badge-blue   { background: rgba(80,129,191,0.12); color: #3A6099; }
.hrv-badge-green  { background: rgba(124,178,100,0.14); color: #4A7A30; }
.hrv-badge-slate  { background: rgba(54,74,105,0.10); color: #364A69; }
.hrv-badge-amber  { background: rgba(232,168,75,0.14); color: #8A5C1A; }
.hrv-tier-bar {
    display: flex;
    align-items: stretch;
    border-radius: 10px;
    overflow: hidden;
    height: 44px;
    margin: 0.5rem 0;
}
.hrv-tier-segment {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.68rem;
    font-weight: 700;
    color: rgba(255,255,255,0.9);
    text-align: center;
    padding: 0 0.3rem;
    letter-spacing: 0.02em;
}
</style>
"""


def inject_css() -> None:
    """Inject the master CSS into the current page. Call once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HTML COMPONENT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "") -> None:
    """Render the top-of-page title block with an optional subtitle and divider."""
    sub_html = (
        f'<p style="font-size:0.9rem;color:#6B7A8D;margin:0.3rem 0 0 0;">'
        f'{subtitle}</p>'
    ) if subtitle else ""
    st.markdown(f"""
    <div class="hrv-page-header">
        <h1 style="font-size:1.9rem;font-weight:700;color:#1C2B3A;
                   margin:0;letter-spacing:-0.02em;">{title}</h1>
        {sub_html}
    </div>
    <div class="hrv-divider"></div>
    """, unsafe_allow_html=True)


def section_header(label: str) -> None:
    """Render a small-caps section label with a bottom border."""
    st.markdown(f"""
    <div style="font-size:0.7rem;font-weight:700;color:#9BA8B8;
                text-transform:uppercase;letter-spacing:0.12em;
                margin:2rem 0 0.9rem 0;padding-bottom:0.55rem;
                border-bottom:1px solid #E8EBF0;">
        {label}
    </div>
    """, unsafe_allow_html=True)


def divider() -> None:
    """Thin horizontal rule."""
    st.markdown('<div class="hrv-divider"></div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "",
                accent: str = "#364A69") -> str:
    """Return HTML for a single metric card (use inside st.markdown)."""
    return f"""
    <div style="background:#FFFFFF;border-radius:14px;padding:1.1rem 1.3rem;
                border:1px solid #E8EBF0;box-shadow:0 1px 6px rgba(54,74,105,0.06);">
        <div style="font-size:0.7rem;font-weight:700;color:#9BA8B8;
                    text-transform:uppercase;letter-spacing:0.09em;
                    margin-bottom:0.35rem;">{label}</div>
        <div style="font-size:1.85rem;font-weight:700;color:{accent};
                    line-height:1.1;">{value}</div>
        {"<div style='font-size:0.74rem;color:#9BA8B8;margin-top:0.2rem;'>"
         + sub + "</div>" if sub else ""}
    </div>"""


def metric_row(items: list) -> None:
    """
    Render a responsive row of metric cards.
    items = list of (label, value, sub) tuples — sub is optional.
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label = item[0]
        value = item[1]
        sub   = item[2] if len(item) > 2 else ""
        with col:
            st.markdown(metric_card(label, value, sub), unsafe_allow_html=True)


def insight_panel(text: str, kind: str = "info",
                  prefix: str = "") -> None:
    """
    Render a left-accented insight / interpretation panel.
    kind: 'info' | 'success' | 'warning' | 'danger'
    """
    color_map = {
        "info":    "#5081BF",
        "success": "#7CB264",
        "warning": "#E8A84B",
        "danger":  "#D4674A",
    }
    bg_map = {
        "info":    "rgba(80,129,191,0.05)",
        "success": "rgba(124,178,100,0.05)",
        "warning": "rgba(232,168,75,0.05)",
        "danger":  "rgba(212,103,74,0.05)",
    }
    border = color_map.get(kind, "#5081BF")
    bg     = bg_map.get(kind, "rgba(80,129,191,0.05)")
    prefix_html = (
        f'<span style="font-weight:700;color:{border};">{prefix} </span>'
    ) if prefix else ""
    st.markdown(f"""
    <div style="border-left:3px solid {border};background:{bg};
                border-radius:0 10px 10px 0;padding:0.9rem 1.2rem;
                margin:0.8rem 0;font-size:0.85rem;color:#2D3E56;line-height:1.65;">
        {prefix_html}{text}
    </div>
    """, unsafe_allow_html=True)


def hrv_card(content_html: str, padding: str = "1.3rem 1.5rem") -> None:
    """Wrap arbitrary HTML in a white card."""
    st.markdown(f"""
    <div style="background:#FFFFFF;border-radius:14px;border:1px solid #E8EBF0;
                padding:{padding};box-shadow:0 1px 6px rgba(54,74,105,0.06);
                margin-bottom:1rem;">
        {content_html}
    </div>
    """, unsafe_allow_html=True)


def tier_bar(segments: list) -> None:
    """
    Horizontal colour-coded tier bar.
    segments = list of (label, pct, hex_color) e.g. ("Normal", 32, "#7CB264")
    """
    html = '<div style="display:flex;border-radius:10px;overflow:hidden;height:40px;margin:0.6rem 0;">'
    for label, pct, color in segments:
        if pct <= 0:
            continue
        html += f"""
        <div style="flex:{pct};background:{color};display:flex;
                    align-items:center;justify-content:center;
                    font-size:0.67rem;font-weight:700;
                    color:rgba(255,255,255,0.92);padding:0 4px;text-align:center;">
            {label}<br>{pct:.0f}%
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
def _build_plotly_template() -> go.layout.Template:
    """Build and register the HRV dashboard Plotly template."""
    template = go.layout.Template()

    template.layout = go.Layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color="#1C2B3A",
            size=12,
        ),
        title=dict(
            font=dict(
                color="#1C2B3A",
                size=15,
                family="Inter",
            ),
            x=0,
            xanchor="left",
            pad=dict(l=4, b=12),
        ),
        xaxis=dict(
            gridcolor="#F0F2F5",
            gridwidth=1,
            zerolinecolor="#E8EBF0",
            zerolinewidth=1,
            linecolor="#E8EBF0",
            tickfont=dict(color="#6B7A8D", size=11),
            title_font=dict(color="#6B7A8D", size=11),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="#F0F2F5",
            gridwidth=1,
            zerolinecolor="#E8EBF0",
            zerolinewidth=1,
            linecolor="#E8EBF0",
            tickfont=dict(color="#6B7A8D", size=11),
            title_font=dict(color="#6B7A8D", size=11),
            showgrid=True,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E8EBF0",
            borderwidth=1,
            font=dict(color="#364A69", size=11),
        ),
        colorway=CHART_SEQ,
        margin=dict(l=48, r=24, t=52, b=44),
        hoverlabel=dict(
            bgcolor="#364A69",
            bordercolor="#364A69",
            font=dict(color="white", size=12, family="Inter"),
        ),
        colorscale=dict(
            diverging=CORR_SCALE,
            sequential=[[0, "#F7F8FA"], [1, "#364A69"]],
            sequentialminus=[[0, "#F7F8FA"], [1, "#D4674A"]],
        ),
    )

    # Default trace styles
    template.data.scatter  = [go.Scatter(marker=dict(size=7, opacity=0.75))]
    template.data.bar      = [go.Bar(marker=dict(opacity=0.85,
                                                  line=dict(width=0)))]
    template.data.histogram= [go.Histogram(marker=dict(opacity=0.82,
                                                        line=dict(width=0)))]
    template.data.violin   = [go.Violin(fillcolor=CHART_SEQ[1],
                                         line_color=CHART_SEQ[0],
                                         opacity=0.75)]
    template.data.box      = [go.Box(marker_color=CHART_SEQ[0],
                                      line_color=CHART_SEQ[0],
                                      fillcolor="rgba(80,129,191,0.12)",
                                      marker=dict(outliercolor=CHART_SEQ[2],
                                                  size=4))]

    pio.templates["hrv"] = template
    return template


# Register template at import time
_build_plotly_template()
# Set as default so every px / go chart uses it automatically
pio.templates.default = "hrv"


def apply_chart_style(
    fig: go.Figure,
    title: str = None,
    height: int = None,
    show_legend: bool = True,
    x_title: str = None,
    y_title: str = None,
) -> go.Figure:
    """
    Apply consistent styling to any Plotly figure.
    Call after fig creation: fig = apply_chart_style(fig, title="...")
    """
    updates = dict(
        template="hrv",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=show_legend,
    )
    if title:
        updates["title"] = dict(
            text=title,
            font=dict(size=14, color="#1C2B3A", family="Inter"),
            x=0, xanchor="left", pad=dict(l=4, b=10),
        )
    if height:
        updates["height"] = height
    if x_title:
        updates["xaxis_title"] = x_title
    if y_title:
        updates["yaxis_title"] = y_title

    fig.update_layout(**updates)

    # Ensure axes use correct colours
    fig.update_xaxes(
        gridcolor="#F0F2F5", gridwidth=1,
        zerolinecolor="#E8EBF0", linecolor="#E8EBF0",
        tickfont=dict(color="#6B7A8D", size=11),
        title_font=dict(color="#6B7A8D", size=11),
    )
    fig.update_yaxes(
        gridcolor="#F0F2F5", gridwidth=1,
        zerolinecolor="#E8EBF0", linecolor="#E8EBF0",
        tickfont=dict(color="#6B7A8D", size=11),
        title_font=dict(color="#6B7A8D", size=11),
    )
    return fig
