"""
utils/statistical_utils.py
==========================
Statistical Analysis Utilities for HRV Clinical Dashboard
-----------------------------------------------------------
This module provides wrappers around standard statistical tests, organised
for clinical data analysis.  Every function returns a structured dict so
results can be displayed uniformly in the Streamlit UI regardless of which
test was used.

Statistical framework
---------------------
For each comparison the module runs *both* a parametric test (t-test, ANOVA)
and its non-parametric equivalent (Mann-Whitney U, Kruskal-Wallis).  This is
the recommended clinical practice because HRV variables are often non-normally
distributed.  The UI then highlights which test is more appropriate based on
a Shapiro-Wilk normality pre-check.

Effect sizes are always reported alongside p-values because statistical
significance alone is insufficient for clinical interpretation:
  Cohen's d  (t-test)     : < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large
  Eta-squared (ANOVA)     : < 0.06 small, 0.06–0.14 medium, > 0.14 large
  Cramér's V  (chi-square): < 0.1 small, 0.1–0.3 medium, > 0.3 large
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr,                    # Correlation
    ttest_ind, mannwhitneyu,                # Two-group comparison
    f_oneway, kruskal,                      # Multi-group comparison
    chi2_contingency,                       # Categorical independence
    shapiro, normaltest,                    # Normality tests
    linregress,                             # Simple linear regression
)


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame, columns: list, method: str = "pearson"):
    """
    Compute a correlation matrix plus a corresponding p-value matrix.

    Pearson correlation measures linear relationships and assumes normality.
    Spearman (rank) correlation is non-parametric and handles monotonic,
    non-linear associations — often preferable for HRV data.

    Parameters
    ----------
    df      : pd.DataFrame
    columns : list[str]  — columns to correlate
    method  : 'pearson' or 'spearman'

    Returns
    -------
    corr_matrix : pd.DataFrame — correlation coefficients (−1 to 1)
    p_matrix    : pd.DataFrame — two-tailed p-values
    """
    valid_cols = [c for c in columns if c in df.columns]
    corr_matrix = df[valid_cols].corr(method=method)

    # Compute pairwise p-values (scipy does not return them with .corr())
    p_matrix = pd.DataFrame(
        np.ones((len(valid_cols), len(valid_cols))),
        index=valid_cols,
        columns=valid_cols,
    )

    for i, c1 in enumerate(valid_cols):
        for j, c2 in enumerate(valid_cols):
            if i >= j:
                continue                        # avoid redundant computation
            pair = df[[c1, c2]].dropna()
            if len(pair) < 3:
                continue
            if method == "pearson":
                _, p = pearsonr(pair[c1], pair[c2])
            else:
                _, p = spearmanr(pair[c1], pair[c2])
            p_matrix.loc[c1, c2] = p
            p_matrix.loc[c2, c1] = p

    return corr_matrix, p_matrix


def get_top_correlations(corr_matrix: pd.DataFrame, p_matrix: pd.DataFrame,
                         n: int = 15) -> pd.DataFrame:
    """
    Return the top-N strongest correlations (excluding self-correlations),
    sorted by absolute correlation coefficient.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
    p_matrix    : pd.DataFrame
    n           : int — number of pairs to return

    Returns
    -------
    pd.DataFrame with columns: Variable 1, Variable 2, Correlation, P-Value, Significant.
    """
    rows = []
    seen = set()

    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if col == row:
                continue
            pair_key = tuple(sorted([col, row]))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            r = corr_matrix.loc[row, col]
            p = p_matrix.loc[row, col]
            rows.append(
                {
                    "Variable 1":   row,
                    "Variable 2":   col,
                    "Correlation":  round(r, 4),
                    "P-Value":      round(p, 4),
                    "Significant":  "✅ Yes" if p < 0.05 else "❌ No",
                    "Strength":     _interpret_r(abs(r)),
                }
            )

    df_corr = pd.DataFrame(rows)
    return df_corr.sort_values("Correlation", key=abs, ascending=False).head(n)


def _interpret_r(r_abs: float) -> str:
    """Verbal interpretation of correlation strength."""
    if r_abs >= 0.7:
        return "Strong"
    elif r_abs >= 0.4:
        return "Moderate"
    elif r_abs >= 0.2:
        return "Weak"
    return "Negligible"


# ─────────────────────────────────────────────────────────────────────────────
# NORMALITY TESTING
# ─────────────────────────────────────────────────────────────────────────────

def test_normality(series: pd.Series) -> dict:
    """
    Run Shapiro-Wilk (n ≤ 5000) or D'Agostino-Pearson (n > 5000) normality test.

    Returns
    -------
    dict with keys: test_name, statistic, p_value, is_normal.
    """
    s = series.dropna()
    if len(s) < 3:
        return {"test_name": "N/A", "statistic": None, "p_value": None, "is_normal": None}

    if len(s) <= 5000:
        stat, p = shapiro(s[:5000])          # shapiro limit is 5000
        test_name = "Shapiro-Wilk"
    else:
        stat, p = normaltest(s)
        test_name = "D'Agostino-Pearson"

    return {
        "test_name": test_name,
        "statistic": round(float(stat), 4),
        "p_value":   round(float(p), 4),
        "is_normal": bool(p >= 0.05),        # p ≥ 0.05 → fail to reject normality
    }


# ─────────────────────────────────────────────────────────────────────────────
# TWO-GROUP COMPARISONS
# ─────────────────────────────────────────────────────────────────────────────

def perform_ttest(df: pd.DataFrame, column: str, group_col: str,
                  group1, group2) -> dict:
    """
    Independent-samples t-test for comparing two groups on a continuous variable.

    Appropriate when the data are approximately normally distributed.
    Returns Cohen's d as the effect size measure.

    Parameters
    ----------
    df        : pd.DataFrame
    column    : str — the continuous outcome variable
    group_col : str — the column that defines groups
    group1, group2 : group labels (any comparable type)

    Returns
    -------
    dict with test results including statistic, p_value, effect_size.
    """
    d1 = df.loc[df[group_col] == group1, column].dropna()
    d2 = df.loc[df[group_col] == group2, column].dropna()

    if len(d1) < 2 or len(d2) < 2:
        return {"error": "Insufficient data for test"}

    t, p = ttest_ind(d1, d2, equal_var=False)  # Welch's t-test (no equal variance assumption)

    # Cohen's d: uses pooled SD of the two groups
    pooled_sd = np.sqrt((d1.std() ** 2 + d2.std() ** 2) / 2)
    d = float((d1.mean() - d2.mean()) / pooled_sd) if pooled_sd > 0 else 0.0

    return {
        "test":          "Welch t-test",
        "statistic":     round(float(t), 4),
        "p_value":       round(float(p), 4),
        "effect_size":   round(abs(d), 4),
        "effect_label":  _cohens_d_label(abs(d)),
        "significant":   bool(p < 0.05),
        "group1":        str(group1),
        "group2":        str(group2),
        "mean1":         round(float(d1.mean()), 3),
        "mean2":         round(float(d2.mean()), 3),
        "std1":          round(float(d1.std()), 3),
        "std2":          round(float(d2.std()), 3),
        "n1":            int(len(d1)),
        "n2":            int(len(d2)),
    }


def perform_mannwhitney(df: pd.DataFrame, column: str, group_col: str,
                        group1, group2) -> dict:
    """
    Mann-Whitney U test — non-parametric alternative to the t-test.

    Does not require normality.  Suitable for ordinal or non-normal
    continuous data, which is common in HRV research.
    Effect size: rank-biserial correlation r = 1 − 2U/(n1×n2).

    Returns
    -------
    dict with test results.
    """
    d1 = df.loc[df[group_col] == group1, column].dropna()
    d2 = df.loc[df[group_col] == group2, column].dropna()

    if len(d1) < 2 or len(d2) < 2:
        return {"error": "Insufficient data for test"}

    u, p = mannwhitneyu(d1, d2, alternative="two-sided")
    r    = float(1 - 2 * u / (len(d1) * len(d2)))  # rank-biserial correlation

    return {
        "test":        "Mann-Whitney U",
        "statistic":   round(float(u), 4),
        "p_value":     round(float(p), 4),
        "effect_size": round(abs(r), 4),
        "effect_label": _rb_corr_label(abs(r)),
        "significant": bool(p < 0.05),
        "group1":      str(group1),
        "group2":      str(group2),
        "median1":     round(float(d1.median()), 3),
        "median2":     round(float(d2.median()), 3),
        "n1":          int(len(d1)),
        "n2":          int(len(d2)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-GROUP COMPARISONS
# ─────────────────────────────────────────────────────────────────────────────

def perform_anova(df: pd.DataFrame, column: str, group_col: str) -> dict:
    """
    One-way ANOVA for comparing means across 3+ groups.

    Eta-squared (η²) is reported as the effect size.
    """
    groups      = df[group_col].dropna().unique()
    group_data  = [df.loc[df[group_col] == g, column].dropna().values
                   for g in groups if len(df.loc[df[group_col] == g, column].dropna()) > 1]

    if len(group_data) < 2:
        return {"error": "Need at least 2 groups with data"}

    f, p        = f_oneway(*group_data)
    grand_mean  = np.concatenate(group_data).mean()
    ss_between  = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_data)
    ss_total    = sum((x - grand_mean) ** 2 for g in group_data for x in g)
    eta_sq      = float(ss_between / ss_total) if ss_total > 0 else 0.0

    return {
        "test":          "One-way ANOVA",
        "statistic":     round(float(f), 4),
        "p_value":       round(float(p), 4),
        "effect_size":   round(eta_sq, 4),
        "effect_label":  _eta_sq_label(eta_sq),
        "significant":   bool(p < 0.05),
        "groups":        list(str(g) for g in groups),
        "group_means":   {str(g): round(float(gd.mean()), 3)
                          for g, gd in zip(groups, group_data)},
    }


def perform_kruskal(df: pd.DataFrame, column: str, group_col: str) -> dict:
    """
    Kruskal-Wallis H-test — non-parametric one-way ANOVA by ranks.

    Recommended when normality cannot be assumed or sample sizes are small.
    """
    groups     = df[group_col].dropna().unique()
    group_data = [df.loc[df[group_col] == g, column].dropna().values
                  for g in groups if len(df.loc[df[group_col] == g, column].dropna()) > 1]

    if len(group_data) < 2:
        return {"error": "Need at least 2 groups with data"}

    h, p = kruskal(*group_data)

    return {
        "test":        "Kruskal-Wallis",
        "statistic":   round(float(h), 4),
        "p_value":     round(float(p), 4),
        "significant": bool(p < 0.05),
        "groups":      list(str(g) for g in groups),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICAL INDEPENDENCE
# ─────────────────────────────────────────────────────────────────────────────

def perform_chisquare(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    Pearson chi-square test of independence between two categorical variables.

    Cramér's V is used as the effect size because it is symmetric and bounded
    in [0, 1] regardless of table size.

    Parameters
    ----------
    df   : pd.DataFrame
    col1 : str — first categorical column
    col2 : str — second categorical column

    Returns
    -------
    dict with test results.
    """
    ct        = pd.crosstab(df[col1].dropna(), df[col2].dropna())
    chi2, p, dof, _ = chi2_contingency(ct)

    n         = ct.sum().sum()
    k         = min(ct.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if n * k > 0 else 0.0

    return {
        "test":         "Chi-square",
        "statistic":    round(float(chi2), 4),
        "p_value":      round(float(p), 4),
        "dof":          int(dof),
        "effect_size":  round(cramers_v, 4),
        "effect_label": _cramers_v_label(cramers_v),
        "significant":  bool(p < 0.05),
        "n":            int(n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE LINEAR REGRESSION (for scatter-plot overlay)
# ─────────────────────────────────────────────────────────────────────────────

def linear_regression_stats(x_series: pd.Series, y_series: pd.Series) -> dict:
    """
    Fit a simple linear regression of y on x and return key statistics
    including 95% confidence intervals on the slope.

    Returns
    -------
    dict with slope, intercept, r², p-value, CI, and significance.
    """
    xy    = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
    if len(xy) < 3:
        return {}

    slope, intercept, r, p, se = linregress(xy["x"], xy["y"])
    ci_margin = 1.96 * se    # 95% CI for the slope

    return {
        "slope":      round(float(slope), 4),
        "intercept":  round(float(intercept), 4),
        "r":          round(float(r), 4),
        "r_squared":  round(float(r ** 2), 4),
        "p_value":    round(float(p), 4),
        "se":         round(float(se), 4),
        "ci_lower":   round(float(slope - ci_margin), 4),
        "ci_upper":   round(float(slope + ci_margin), 4),
        "significant": bool(p < 0.05),
        "n":           int(len(xy)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUBGROUP COMPARISON RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_subgroup_comparisons(df: pd.DataFrame, numeric_cols: list,
                              group_col: str) -> pd.DataFrame:
    """
    Run ANOVA + Kruskal-Wallis for every numeric variable across all groups
    defined by group_col.  Returns a tidy summary table.

    Parameters
    ----------
    df          : pd.DataFrame
    numeric_cols: list[str] — continuous outcome variables
    group_col   : str       — grouping variable (≥ 3 groups)

    Returns
    -------
    pd.DataFrame with one row per variable and columns: Variable, F, ANOVA_p,
    H, KW_p, Significant, Effect_Size (η²).
    """
    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        anova_r  = perform_anova(df, col, group_col)
        kruskal_r = perform_kruskal(df, col, group_col)

        if "error" in anova_r:
            continue

        rows.append(
            {
                "Variable":       col,
                "ANOVA F":        anova_r.get("statistic"),
                "ANOVA p":        anova_r.get("p_value"),
                "KW H":           kruskal_r.get("statistic"),
                "KW p":           kruskal_r.get("p_value"),
                "Eta-squared":    anova_r.get("effect_size"),
                "Effect Label":   anova_r.get("effect_label"),
                "Significant":    "✅ Yes" if anova_r.get("significant") else "❌ No",
            }
        )

    return pd.DataFrame(rows).sort_values("ANOVA p") if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# EFFECT SIZE INTERPRETERS (private helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _cohens_d_label(d: float) -> str:
    if d < 0.2:   return "Negligible"
    if d < 0.5:   return "Small"
    if d < 0.8:   return "Medium"
    return "Large"


def _rb_corr_label(r: float) -> str:
    """Rank-biserial correlation interpretation for Mann-Whitney."""
    if r < 0.1:   return "Negligible"
    if r < 0.3:   return "Small"
    if r < 0.5:   return "Medium"
    return "Large"


def _eta_sq_label(eta: float) -> str:
    if eta < 0.06:  return "Small"
    if eta < 0.14:  return "Medium"
    return "Large"


def _cramers_v_label(v: float) -> str:
    if v < 0.1:   return "Small"
    if v < 0.3:   return "Medium"
    return "Large"
