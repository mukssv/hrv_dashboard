"""
utils/data_processing.py
========================
Data Loading, Cleaning, Preprocessing & Feature Engineering
------------------------------------------------------------
All functions in this module transform raw clinical CSV data into a
clean, analysis-ready DataFrame.  The preprocessing pipeline is
deliberately documented step-by-step so every transformation is
auditable — a key requirement for clinical data systems.

Key responsibilities
--------------------
1. Load CSV data from a file path or an in-memory buffer (Streamlit upload).
2. Validate that required columns are present; derive optional ones if absent.
3. Apply the mandatory Stress Score > 20 filter and log removed records.
4. Remove duplicate patient–date records and log the count.
5. Impute residual missing values using column medians (conservative choice).
6. Engineer clinically meaningful derived features (BMI category, age group,
   HRV composite indices).
7. Detect physiological outliers using IQR and Z-score methods.
8. Validate that all physiological values fall within plausible ranges.
9. Compute comprehensive descriptive statistics for every numeric variable.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings                      # Suppress minor non-critical warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np                   # Numeric arrays, statistical operations
import pandas as pd                  # DataFrame operations and data manipulation
from scipy import stats              # Statistical functions (zscore, etc.)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA CONSTANTS
# These lists define the expected column names.  Downstream pages import them
# to avoid hard-coding column names in multiple places (single source of truth).
# ─────────────────────────────────────────────────────────────────────────────

# ── Internal (clean) column names used throughout the entire application ──────
# These are the names used AFTER the raw CSV columns have been remapped by
# normalize_columns().  All pages import these constants to avoid hard-coding
# column names in multiple places (single source of truth).

REQUIRED_COLUMNS = [
    # Identity
    "Patient_ID",           # ← Patient DB ID
    "Patient_Name",         # ← Patient Name       (used for patient search/profiles)
    "Age",                  # ← Patient Age
    "Sex",                  # ← Patient Sex
    # Clinic
    "Clinic_ID",            # ← Clinic ID
    "Clinic_Name",          # ← Clinic Name
    # Report
    "Date",                 # ← Report Date
    # HRV / physiological
    "Mean_RR_ms",           # ← Mean RR (ms)
    "SDNN_ms",              # ← SDNN (ms)
    "RMSSD_ms",             # ← RMSSD (ms)
    "pNN50_pct",            # ← pNN50 (%)
    "pNN20_pct",            # ← pNN20 (%)
    "LF_Power_ms2",         # ← LF Power (ms²)
    "HF_Power_ms2",         # ← HF Power (ms²)
    "LF_HF_Ratio",          # ← LF/HF Ratio
    "Stress_Score",         # ← Stress Score
]

# Optional columns that can be derived if absent in the upload
DERIVED_COLUMNS = ["Height_cm", "Weight_kg", "BMI", "Heart_Rate_bpm"]

# Core HRV measurement columns (used in most analytical layers)
HRV_COLUMNS = [
    "Mean_RR_ms", "SDNN_ms", "RMSSD_ms",
    "pNN50_pct", "pNN20_pct",
    "LF_Power_ms2", "HF_Power_ms2", "LF_HF_Ratio",
]

# Demographic / anthropometric columns
DEMO_COLUMNS = ["Age", "Height_cm", "Weight_kg", "BMI", "Heart_Rate_bpm"]

# All numeric columns used in statistical and ML analysis
NUMERIC_COLUMNS = DEMO_COLUMNS + HRV_COLUMNS + ["Stress_Score"]

# Derived composite HRV index columns (added by add_derived_features)
INDEX_COLUMNS = [
    "HRV_Health_Index",
    "Parasympathetic_Activity_Index",
    "Sympathetic_Dominance_Index",
    "Stress_Physiology_Index",
]

# ─────────────────────────────────────────────────────────────────────────────
# RAW → INTERNAL COLUMN MAPPING
# Maps every actual CSV column name produced by the clinic system to the clean
# internal name used throughout the rest of the codebase.
#
# Why a mapping instead of using raw names everywhere?
#   Raw names contain spaces, parentheses, units, and encoding artefacts
#   (e.g. "LF Power (msÂ²)" where Â² is a UTF-8/Latin-1 decode error for ²).
#   Renaming once at load time keeps every subsequent function clean and safe.
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_MAPPING = {
    # ── Identity ──────────────────────────────────────────────────────────────
    "Patient DB ID":                   "Patient_ID",
    "Patient System ID":               "Patient_System_ID",
    "Patient Name":                    "Patient_Name",
    "Patient Age":                     "Age",
    "Patient Sex":                     "Sex",
    "Patient Weight (kg)":             "Weight_kg",
    "Patient Height (cm)":             "Height_cm",
    "Patient Registered At (Clinic)":  "Patient_Registered_Clinic",
    "Patient Last Updated At (Clinic)":"Patient_Last_Updated",

    # ── Clinic ────────────────────────────────────────────────────────────────
    "Clinic ID":                       "Clinic_ID",
    "Clinic Name":                     "Clinic_Name",
    "Clinic Reports":                  "Clinic_Reports",
    "Generated Clinic Registered On":  "Clinic_Registered_On",

    # ── Report metadata ───────────────────────────────────────────────────────
    "Report ID":                       "Report_ID",
    "Report Date":                     "Date",
    "Report Created At":               "Report_Created_At",

    # ── Physiological / HRV ───────────────────────────────────────────────────
    "Stress Score":                    "Stress_Score",
    "Mean RR (ms)":                    "Mean_RR_ms",
    "SDNN (ms)":                       "SDNN_ms",
    "RMSSD (ms)":                      "RMSSD_ms",
    "pNN50 (%)":                       "pNN50_pct",
    "pNN20 (%)":                       "pNN20_pct",
    # The ² character is often mangled to Â² when a UTF-8 file is mis-read as
    # Latin-1.  normalize_columns() fixes the encoding BEFORE this mapping runs,
    # so both the clean and mangled forms are handled.
    "LF Power (ms\u00b2)":            "LF_Power_ms2",   # ² = U+00B2
    "HF Power (ms\u00b2)":            "HF_Power_ms2",
    "LF Power (msÂ²)":                "LF_Power_ms2",   # fallback for bad encoding
    "HF Power (msÂ²)":                "HF_Power_ms2",
    "LF/HF Ratio":                    "LF_HF_Ratio",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL REFERENCE RANGES
# These define the healthy adult reference window for each HRV metric based on
# published normative values (Task Force ESC/NASPE guidelines and meta-analyses).
# Used for range validation, colour-coding, and interpretation cards.
# ─────────────────────────────────────────────────────────────────────────────
HRV_REFERENCE_RANGES = {
    # ── Mean RR Interval ──────────────────────────────────────────────────────
    # Normal range: 785–1160 ms (corresponds to ~52–76 bpm resting heart rate).
    # Values outside this window suggest tachycardia or bradycardia.
    "Mean_RR_ms": {
        "min": 500, "max": 1500,
        "healthy_low": 785, "healthy_high": 1160,
        "unit": "ms",
        "description": "Mean RR Interval",
        "clinical_note": "Normal: 785–1160 ms (≈52–76 bpm). Outside range suggests arrhythmia risk.",
        "tiers": {
            "Normal":  {"low": 785,  "high": 1160, "label": "Normal"},
            "Low":     {"low": 0,    "high": 785,  "label": "Low (Tachycardia risk)"},
            "High":    {"low": 1160, "high": 9999, "label": "High (Bradycardia risk)"},
        },
    },

    # ── SDNN ─────────────────────────────────────────────────────────────────
    # Three clinical tiers: >50 ms Normal, 27–50 ms Moderate, <27 ms Low.
    # SDNN < 27 ms is independently associated with increased all-cause mortality.
    "SDNN_ms": {
        "min": 0, "max": 300,
        "healthy_low": 50, "healthy_high": 200,   # >50 = Normal
        "unit": "ms",
        "description": "Standard Deviation of NN Intervals",
        "clinical_note": ">50 ms Normal | 27–50 ms Moderate | <27 ms Low (elevated cardiac risk)",
        "tiers": {
            "Normal":   {"low": 50,   "high": 9999, "label": "Normal (>50 ms)"},
            "Moderate": {"low": 27,   "high": 50,   "label": "Moderate (27–50 ms)"},
            "Low":      {"low": 0,    "high": 27,   "label": "Low (<27 ms)"},
        },
    },

    # ── RMSSD ─────────────────────────────────────────────────────────────────
    # Gold-standard parasympathetic (vagal) marker.
    # >40 ms = Optimal, 15.7–40 ms = Moderate, <15.7 ms = Low vagal tone.
    "RMSSD_ms": {
        "min": 0, "max": 300,
        "healthy_low": 40, "healthy_high": 200,   # >40 = Optimal
        "unit": "ms",
        "description": "Root Mean Square of Successive Differences",
        "clinical_note": ">40 ms Optimal | 15.7–40 ms Moderate | <15.7 ms Low vagal tone",
        "tiers": {
            "Optimal":  {"low": 40,   "high": 9999, "label": "Optimal (>40 ms)"},
            "Moderate": {"low": 15.7, "high": 40,   "label": "Moderate (15.7–40 ms)"},
            "Low":      {"low": 0,    "high": 15.7, "label": "Low (<15.7 ms)"},
        },
    },

    # ── pNN50 ─────────────────────────────────────────────────────────────────
    # Strict inter-beat consistency metric; highly sensitive to vagal withdrawal.
    # >20% = Optimal, 1–20% = Reduced, <1% = Abnormal.
    "pNN50_pct": {
        "min": 0, "max": 100,
        "healthy_low": 20, "healthy_high": 100,   # >20% = Optimal
        "unit": "%",
        "description": "Proportion of NN intervals differing > 50 ms",
        "clinical_note": ">20% Optimal | 1–20% Reduced | <1% Abnormal (severe vagal withdrawal)",
        "tiers": {
            "Optimal":  {"low": 20,  "high": 100,  "label": "Optimal (>20%)"},
            "Reduced":  {"low": 1,   "high": 20,   "label": "Reduced (1–20%)"},
            "Abnormal": {"low": 0,   "high": 1,    "label": "Abnormal (<1%)"},
        },
    },

    # ── pNN20 ─────────────────────────────────────────────────────────────────
    # More inclusive parasympathetic metric; captures mild vagal changes that
    # pNN50 may miss, particularly in older or high-stress populations.
    # >50% = Optimal, 30–50% = Moderate, <30% = Low.
    "pNN20_pct": {
        "min": 0, "max": 100,
        "healthy_low": 50, "healthy_high": 100,   # >50% = Optimal
        "unit": "%",
        "description": "Proportion of NN intervals differing > 20 ms",
        "clinical_note": ">50% Optimal | 30–50% Moderate | <30% Low parasympathetic activity",
        "tiers": {
            "Optimal":  {"low": 50,  "high": 100,  "label": "Optimal (>50%)"},
            "Moderate": {"low": 30,  "high": 50,   "label": "Moderate (30–50%)"},
            "Low":      {"low": 0,   "high": 30,   "label": "Low (<30%)"},
        },
    },

    # ── LF Power ─────────────────────────────────────────────────────────────
    # Frequency-domain power in the 0.04–0.15 Hz band; reflects baroreceptor
    # activity (mixed sympathetic + parasympathetic modulation).
    # Normal: 193–1009 ms², <193 Reduced, >1009 Increased.
    "LF_Power_ms2": {
        "min": 0, "max": 10000,
        "healthy_low": 193, "healthy_high": 1009,
        "unit": "ms²",
        "description": "Low Frequency Power (0.04–0.15 Hz)",
        "clinical_note": "193–1009 ms² Normal | <193 ms² Reduced | >1009 ms² Increased",
        "tiers": {
            "Normal":    {"low": 193,  "high": 1009, "label": "Normal (193–1009 ms²)"},
            "Reduced":   {"low": 0,    "high": 193,  "label": "Reduced (<193 ms²)"},
            "Increased": {"low": 1009, "high": 9999, "label": "Increased (>1009 ms²)"},
        },
    },

    # ── HF Power ─────────────────────────────────────────────────────────────
    # Respiratory-coupled vagal power (0.15–0.40 Hz); purely parasympathetic.
    # Normal: 86–3630 ms², <86 Reduced, >3630 Increased.
    "HF_Power_ms2": {
        "min": 0, "max": 15000,
        "healthy_low": 86, "healthy_high": 3630,
        "unit": "ms²",
        "description": "High Frequency Power (0.15–0.40 Hz)",
        "clinical_note": "86–3630 ms² Normal | <86 ms² Reduced | >3630 ms² Increased",
        "tiers": {
            "Normal":    {"low": 86,   "high": 3630, "label": "Normal (86–3630 ms²)"},
            "Reduced":   {"low": 0,    "high": 86,   "label": "Reduced (<86 ms²)"},
            "Increased": {"low": 3630, "high": 9999, "label": "Increased (>3630 ms²)"},
        },
    },

    # ── LF/HF Ratio ──────────────────────────────────────────────────────────
    # Sympathovagal balance index.
    # 1.1–3.0 = Normal balance, >3.0 = Sympathetic Dominance, <1.1 = Parasympathetic Dominance.
    "LF_HF_Ratio": {
        "min": 0, "max": 15,
        "healthy_low": 1.1, "healthy_high": 3.0,
        "unit": "",
        "description": "LF/HF Ratio (Sympathovagal Balance)",
        "clinical_note": "1.1–3.0 Normal | >3.0 Sympathetic Dominance | <1.1 Parasympathetic Dominance",
        "tiers": {
            "Normal":                   {"low": 1.1,  "high": 3.0,  "label": "Normal (1.1–3.0)"},
            "Sympathetic Dominance":    {"low": 3.0,  "high": 9999, "label": "Sympathetic Dominance (>3.0)"},
            "Parasympathetic Dominance":{"low": 0,    "high": 1.1,  "label": "Parasympathetic Dominance (<1.1)"},
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — classify a single HRV value into its clinical tier label
# ─────────────────────────────────────────────────────────────────────────────
def classify_hrv_value(column: str, value: float) -> str:
    """
    Return the clinical tier label for a single HRV measurement.

    Uses the 'tiers' dict inside HRV_REFERENCE_RANGES.  Falls back to
    "N/A" if the column or value is not recognised.

    Parameters
    ----------
    column : str   — internal column name (e.g. "RMSSD_ms")
    value  : float — raw measurement value

    Returns
    -------
    str — tier label (e.g. "Optimal (>40 ms)", "Moderate (15.7–40 ms)")
    """
    ref = HRV_REFERENCE_RANGES.get(column, {})
    tiers = ref.get("tiers", {})
    for tier_name, bounds in tiers.items():
        if bounds["low"] <= value < bounds["high"]:
            return bounds["label"]
    # Handle edge case of exact maximum
    if tiers:
        last_tier = list(tiers.values())[-1]
        if value >= last_tier["low"]:
            return last_tier["label"]
    return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix column name encoding issues and apply the COLUMN_MAPPING to rename
    all raw CSV columns to clean internal names.

    This is the FIRST operation run on any loaded DataFrame.  All subsequent
    functions assume the internal naming convention is already in place.

    Encoding fix
    ------------
    Some clinic export systems save the superscript ² (U+00B2) in UTF-8
    (bytes: 0xC2 0xB2), then the CSV is opened as Latin-1 which renders those
    two bytes as "Â²".  We normalise the column names before applying the
    mapping so both the clean and mangled forms are handled transparently.

    Parameters
    ----------
    df : pd.DataFrame — raw, just-loaded DataFrame

    Returns
    -------
    pd.DataFrame — same data, column names replaced with internal names.
    """
    # Step 1: strip leading/trailing whitespace from all column names
    df.columns = df.columns.str.strip()

    # Step 2: fix the Â² → ² encoding artefact in column names
    df.columns = [
        col.replace("Â²", "\u00b2").replace("â²", "\u00b2")
        for col in df.columns
    ]

    # Step 3: apply the COLUMN_MAPPING (only renames columns that are present;
    #         unknown extra columns are left unchanged and cause no errors)
    df = df.rename(columns=COLUMN_MAPPING)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(file_source):
    """
    Load a CSV file from a file path string or a Streamlit UploadedFile buffer,
    then immediately normalise column names via normalize_columns().

    Parameters
    ----------
    file_source : str or file-like object
        Path string like 'data/hrv.csv' OR the object returned by
        st.file_uploader() when a user uploads a CSV.

    Returns
    -------
    tuple (pd.DataFrame | None, str | None)
        (dataframe, None) on success.
        (None, error_message) on failure.

    Notes
    -----
    Using a try/except here prevents the entire Streamlit app from crashing
    if the uploaded file is malformed or has an unsupported encoding.
    The function tries UTF-8 first, then falls back to Latin-1 (ISO-8859-1)
    which handles the "Â²" encoding scenario at the file level too.
    """
    try:
        # Attempt standard UTF-8 read first
        try:
            df = pd.read_csv(file_source, encoding="utf-8")
        except UnicodeDecodeError:
            # If the file was saved as Latin-1 (common with older clinic systems),
            # re-read with that encoding so raw bytes are decoded correctly.
            if hasattr(file_source, "seek"):
                file_source.seek(0)          # reset buffer position for re-read
            df = pd.read_csv(file_source, encoding="latin-1")

        # Rename columns to clean internal names immediately after loading
        df = normalize_columns(df)
        return df, None

    except Exception as exc:
        return None, str(exc)              # surface the error message to the UI


# ─────────────────────────────────────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame):
    """
    Execute the full data preprocessing pipeline and return a clean DataFrame
    together with an audit log documenting every transformation.

    Pipeline steps (in order):
    1. Parse the Date column to datetime objects.
    2. Derive BMI, Heart Rate, and LF/HF Ratio if not already present.
    3. Filter: keep only Stress_Score > 20 (per dashboard specification).
    4. Remove duplicate patient–date records.
    5. Record missing value counts before imputation.
    6. Impute remaining missing numerics with per-column medians.
    7. Enrich data with derived clinical features (categories, indices).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as loaded from the CSV (may contain NaN, duplicates, etc.)

    Returns
    -------
    tuple (pd.DataFrame, dict)
        - Cleaned, enriched DataFrame
        - preprocessing_log: a dict with counts and metadata for each step,
          displayed on the Home and Data Overview pages.
    """
    log = {}                                    # Audit trail dictionary
    log["initial_records"] = len(df)            # Count before any cleaning

    # ── Step 0: Column name normalisation ─────────────────────────────────────
    # Strip whitespace (belt-and-suspenders in case normalize_columns was not
    # called before preprocess_data — e.g. when testing functions directly).
    df.columns = df.columns.str.strip()

    # ── Step 1: Parse Date column ─────────────────────────────────────────────
    # errors='coerce' turns unparseable strings into NaT instead of crashing.
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ── Step 2: Derive optional columns if absent ─────────────────────────────
    # BMI: standard weight-height formula; critical for demographic stratification.
    if "BMI" not in df.columns:
        if {"Height_cm", "Weight_kg"}.issubset(df.columns):
            df["BMI"] = (df["Weight_kg"] / (df["Height_cm"] / 100) ** 2).round(1)

    # Heart Rate (bpm): derived from Mean RR; used in correlation analysis.
    if "Heart_Rate_bpm" not in df.columns and "Mean_RR_ms" in df.columns:
        df["Heart_Rate_bpm"] = (60_000 / df["Mean_RR_ms"]).round(1)

    # LF/HF Ratio: key sympathovagal balance metric.
    if "LF_HF_Ratio" not in df.columns:
        if {"LF_Power_ms2", "HF_Power_ms2"}.issubset(df.columns):
            df["LF_HF_Ratio"] = (
                df["LF_Power_ms2"] / df["HF_Power_ms2"].replace(0, np.nan)
            ).round(2)

    # ── Step 3: Stress Score filter (Stress_Score > 20) ───────────────────────
    # Dashboard specification mandates excluding low-stress records so that
    # analyses focus on the clinically relevant stress-affected population.
    if "Stress_Score" in df.columns:
        pre_filter = len(df)
        df = df[df["Stress_Score"] > 20].copy()
        post_filter = len(df)

        log["records_before_stress_filter"] = pre_filter
        log["records_removed_stress_filter"] = pre_filter - post_filter  # documented
        log["records_after_stress_filter"]   = post_filter
    else:
        log["stress_filter_skipped"] = "Stress_Score column not found"

    # ── Step 4: Duplicate removal ─────────────────────────────────────────────
    # A duplicate is defined as an entire row being identical to another row.
    # keep='first' retains the first occurrence (arbitrary but consistent).
    pre_dedup = len(df)

    df = df.drop_duplicates(keep="first")   # no subset → compares ALL columns

    log["records_removed_duplicates"] = pre_dedup - len(df)
    log["records_after_dedup"]        = len(df)

    # ── Step 5: Missing value audit (before imputation) ───────────────────────
    missing_counts = df.isnull().sum()
    missing_pct    = (missing_counts / len(df) * 100).round(2)

    log["missing_values"] = {
        col: {"count": int(missing_counts[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing_counts[col] > 0
    }

    # ── Step 6: Median imputation ─────────────────────────────────────────────
    # Median is preferred over mean for clinical data because HRV distributions
    # are often right-skewed; the median is resistant to extreme outliers.
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # ── Step 7: Derived features ──────────────────────────────────────────────
    df = add_derived_features(df)

    # ── Final audit counts ────────────────────────────────────────────────────
    log["final_records"] = len(df)
    log["total_columns"] = len(df.columns)

    # Data completeness score: percentage of cells that are non-null.
    total_cells    = df.shape[0] * df.shape[1]
    non_null_cells = df.notna().sum().sum()
    log["completeness_score"] = round(non_null_cells / total_cells * 100, 2)

    log["preprocessing_complete"] = True

    return df.reset_index(drop=True), log


# ─────────────────────────────────────────────────────────────────────────────
# DERIVED FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the cleaned DataFrame with clinically interpretable derived columns.

    Adds
    ----
    BMI_Category    : WHO obesity classification (string)
    Age_Group       : Decade-based age band (string)
    Stress_Level    : Three-tier stress classification (string)
    HRV_Health_Index            : Composite HRV health score 0–100
    Parasympathetic_Activity_Index : Vagal tone index 0–100
    Sympathetic_Dominance_Index : Sympathetic load index 0–100
    Stress_Physiology_Index     : Combined stress physiology score 0–100

    Parameters
    ----------
    df : pd.DataFrame  (must have already passed through preprocess_data steps 1–6)

    Returns
    -------
    pd.DataFrame with additional derived columns.
    """

    # ── Categorical encodings ─────────────────────────────────────────────────

    # BMI Category — Updated clinical classification with full descriptive labels.
    # Bins use right=False so each boundary belongs to the upper category,
    # e.g. BMI=25.0 → "Overweight" not "Healthy / Normal Weight".
    # BMI values are rounded to 1 decimal place before categorisation.
    if "BMI" in df.columns:
        df["BMI"] = df["BMI"].round(1)          # enforce 1 d.p. per specification
        df["BMI_Category"] = pd.cut(
            df["BMI"],
            bins=[0, 18.5, 25.0, 30.0, 35.0, 40.0, float("inf")],
            labels=[
                "Underweight",           # BMI < 18.5
                "Healthy / Normal Weight",   # 18.5 ≤ BMI < 25.0
                "Overweight",            # 25.0 ≤ BMI < 30.0
                "Obesity Class 1",       # 30.0 ≤ BMI < 35.0
                "Obesity Class 2",       # 35.0 ≤ BMI < 40.0
                "Extreme Obesity Class 3",  # BMI ≥ 40.0
            ],
            right=False,
        ).astype(str)

    # Age Group — clinical age bands aligned with cardiovascular risk stratification.
    # Using right=False so e.g. age=25 → "25–40" not "18–25".
    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 25, 40, 55, 65, float("inf")],
            labels=["18–25", "25–40", "40–55", "55–65", "65+"],
            right=False,
        ).astype(str)

    # Stress Level — five clinical tiers reflecting progressive autonomic burden.
    # Thresholds:  Normal <40 | Mild 41–55 | Moderate 56–70 | Severe 71–85 | Extremely Severe >85
    # Note: because the dataset is pre-filtered for Stress_Score > 20, the "Normal"
    # category will contain scores in the 21–40 range.
    if "Stress_Score" in df.columns:
        df["Stress_Level"] = pd.cut(
            df["Stress_Score"],
            bins=[0, 40, 55, 70, 85, float("inf")],
            labels=[
                "Normal Stress",          # ≤ 40
                "Mild Stress",            # 41–55
                "Moderate Stress",        # 56–70
                "Severe Stress",          # 71–85
                "Extremely Severe Stress", # > 85
            ],
            right=True,                   # right=True: 40 → "Normal Stress", 41 → "Mild Stress"
        ).astype(str)

    # ─────────────────────────────────────────────────────────────────────────────
    # SCORING FUNCTION 1: Ramp-and-Cap (for monotonic metrics)
    # Use when: higher value is always better, with a defined Optimal ceiling.
    # ─────────────────────────────────────────────────────────────────────────────
    def _ramp_and_cap(series: pd.Series, low: float, optimal: float) -> pd.Series:
        """
        Scores a metric on 0–1 where:
        ≤ low     → 0.0  (Low / Abnormal tier)
        ≥ optimal → 1.0  (Optimal tier — capped, no extra credit above this)
        between   → linear ramp

        Why 'cap' at optimal?
        '>40 ms is Optimal' means 40 ms and 80 ms are medically equivalent.
        The old norm_clip(15.7, 80) gave 40 ms a score of 0.38, unfairly penalising
        a clinically optimal patient. This function gives both a score of 1.0.

        Parameters
        ----------
        low     : value at which score = 0 (the Low/Abnormal boundary)
        optimal : value at which score = 1 (the Optimal/Normal boundary, ceiling)
        """
        return ((series - low) / (optimal - low)).clip(0, 1)


    # ─────────────────────────────────────────────────────────────────────────────
    # SCORING FUNCTION 2: Trapezoid (for target-zone metrics)
    # Use when: a normal range exists and being outside it in EITHER direction is bad.
    # ─────────────────────────────────────────────────────────────────────────────
    def _trapezoid(series: pd.Series,
                bad_low: float, normal_low: float,
                normal_high: float, bad_high: float) -> pd.Series:
        """
        Scores a metric on 0–1 using a trapezoidal membership function:

        Score
        1.0 |        ┌─────────────┐
            |       /               \\
        0.5 |      /                 \\
            |     /                   \\
        0.0 |____/                     \\____
                    ↑         ↑     ↑         ↑
                bad_low  norm_lo norm_hi  bad_high

        Every value inside [normal_low, normal_high] scores 1.0.
        This fixes the LF Power problem: both 193 ms² and 1009 ms² score 1.0
        (both are Normal), instead of 193 scoring 0 and 1009 scoring 1.

        Parameters
        ----------
        bad_low    : score is 0 at this value and below
        normal_low : score reaches 1.0 here (bottom of Normal range)
        normal_high: score leaves 1.0 here (top of Normal range)
        bad_high   : score falls back to 0 at this value
        """
        score = pd.Series(0.0, index=series.index, dtype=float)

        # Segment 1: Rising ramp (bad_low → normal_low)
        rising = (series > bad_low) & (series < normal_low)
        score[rising] = (series[rising] - bad_low) / (normal_low - bad_low)

        # Segment 2: Flat top (normal_low → normal_high) — entire normal range = 1.0
        flat = (series >= normal_low) & (series <= normal_high)
        score[flat] = 1.0

        # Segment 3: Falling ramp (normal_high → bad_high)
        falling = (series > normal_high) & (series < bad_high)
        score[falling] = 1.0 - (series[falling] - normal_high) / (bad_high - normal_high)

        # Values at or below bad_low and at or above bad_high remain 0.0
        return score.clip(0, 1)


    # ─────────────────────────────────────────────────────────────────────────────
    # WEIGHT FUNCTION: PCA-Derived Weights
    # Replaces hardcoded weights (0.40, 0.30, 0.30) with data-driven loadings.
    # ─────────────────────────────────────────────────────────────────────────────
    def _pca_weights(df: pd.DataFrame, metrics: list) -> dict:
        """
        Compute weights for combining sub-metrics using PCA first-component loadings.

        What this does in plain English:
        1. Takes the sub-metrics (e.g. RMSSD, pNN50, HF Power for PAI).
        2. Standardises them so scale differences don't bias the result.
        3. Runs PCA to find the direction of maximum shared variation.
        4. Metrics that align strongly with that direction get higher weights.
        5. Normalises so all weights sum to 1.0.

        Why this is better than hardcoded weights:
        - Derived entirely from YOUR patient data.
        - A metric that is nearly constant across your patients (low variance)
            automatically gets a lower weight — it is not helping differentiate anyone.
        - Updates each time you apply it to a new dataset.

        Fallback:
        If fewer than 10 rows or fewer than 2 valid metrics are present,
        returns equal weights so the index calculation never crashes.

        Returns: dict {metric_name: weight}  where weights sum to 1.0
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        valid = [m for m in metrics if m in df.columns]

        # Safety check: PCA needs at least 2 metrics and enough rows to be stable
        if len(valid) < 2 or len(df[valid].dropna()) < max(10, len(valid)):
            n = max(len(valid), 1)
            return {m: round(1 / n, 4) for m in valid}

        X = df[valid].dropna()

        # Standardise: PCA is scale-sensitive. Without this, a metric measured in
        # thousands (e.g. HF Power in ms²) would dominate simply due to magnitude.
        X_scaled = StandardScaler().fit_transform(X)

        # Fit PCA with just 1 component — we only want the first principal component
        pca = PCA(n_components=1, random_state=42)
        pca.fit(X_scaled)

        # components_[0]: one loading per metric.
        # High absolute loading = metric strongly aligned with the shared latent concept.
        # We take absolute values because direction (positive/negative) doesn't affect weight.
        loadings = np.abs(pca.components_[0])

        # Normalise so all weights sum exactly to 1.0
        weights = loadings / loadings.sum()

        return {m: round(float(w), 4) for m, w in zip(valid, weights)}


    # ─────────────────────────────────────────────────────────────────────────────
    # COMPOSITE INDEX CALCULATIONS
    # ─────────────────────────────────────────────────────────────────────────────

    required_for_index = [
        "SDNN_ms", "RMSSD_ms", "pNN50_pct", "pNN20_pct",
        "LF_Power_ms2", "HF_Power_ms2", "LF_HF_Ratio",
    ]
    if all(c in df.columns for c in required_for_index):

        # ── Pre-score every metric once, using the medically correct scorer ───────

        # MONOTONIC metrics (higher = healthier, cap at Optimal threshold):
        # Threshold values = the boundary between Moderate and Optimal/Normal
        s_sdnn  = _ramp_and_cap(df["SDNN_ms"],   low=27.0, optimal=50.0)  # >50 = Normal
        s_rmssd = _ramp_and_cap(df["RMSSD_ms"],  low=15.7, optimal=40.0)  # >40 = Optimal
        s_pnn50 = _ramp_and_cap(df["pNN50_pct"], low=1.0,  optimal=20.0)  # >20 = Optimal
        s_pnn20 = _ramp_and_cap(df["pNN20_pct"], low=30.0, optimal=50.0)  # >50 = Optimal

        # TARGET-ZONE metrics (score 1 across entire normal range):
        s_hf_health = _trapezoid(
            df["HF_Power_ms2"],
            bad_low=0, normal_low=86, normal_high=3630, bad_high=7200
        )
        s_lf_health = _trapezoid(
            df["LF_Power_ms2"],
            bad_low=0, normal_low=193, normal_high=1009, bad_high=3000
        )
        # LF/HF Ratio for health context: 1.0 across normal range (1.1–3.0), 0 at extremes
        s_ratio_health = _trapezoid(
            df["LF_HF_Ratio"],
            bad_low=0.0, normal_low=1.1, normal_high=3.0, bad_high=6.0
        )

        # SYMPATHETIC DIRECTION scorers: score INCREASES as metric moves toward dominance.
        # LF/HF Ratio for SDI: 0 at bottom of normal (1.1), 1 at extreme dominance (6.0)
        # A ratio of 3.0 (the dominance threshold) scores (3.0−1.1)/(6.0−1.1) = 0.39
        s_ratio_sdi = _ramp_and_cap(df["LF_HF_Ratio"], low=1.1, optimal=6.0)

        # LF Power for SDI: 0 at bottom of normal (193), 1 at clearly elevated (3000)
        # Values within normal range (193–1009) score proportionally — rising sympathetic signal
        s_lf_sdi = _ramp_and_cap(df["LF_Power_ms2"], low=193, optimal=3000)

        # ── Index 1: Parasympathetic Activity Index (PAI) ─────────────────────────
        # Concept: how active is the rest-and-recovery system?
        # Components: the three gold-standard parasympathetic markers
        pai_metrics = ["RMSSD_ms", "pNN50_pct", "HF_Power_ms2"]
        pai_scores  = {"RMSSD_ms": s_rmssd, "pNN50_pct": s_pnn50, "HF_Power_ms2": s_hf_health}
        pai_w       = _pca_weights(df, pai_metrics)

        df["Parasympathetic_Activity_Index"] = (
            sum(pai_scores[m] * pai_w.get(m, 1/3) for m in pai_metrics) * 100
        ).clip(0, 100).round(1)

        # ── Index 2: Sympathetic Dominance Index (SDI) ────────────────────────────
        # Concept: how much has the stress/fight-or-flight system taken over?
        # Both inputs are scored in the SYMPATHETIC DIRECTION (higher = more dominant)
        sdi_metrics = ["LF_Power_ms2", "LF_HF_Ratio"]
        sdi_scores  = {"LF_Power_ms2": s_lf_sdi, "LF_HF_Ratio": s_ratio_sdi}
        sdi_w       = _pca_weights(df, sdi_metrics)

        df["Sympathetic_Dominance_Index"] = (
            sum(sdi_scores[m] * sdi_w.get(m, 0.5) for m in sdi_metrics) * 100
        ).clip(0, 100).round(1)

        # ── Index 3: HRV Health Index (HHI) ──────────────────────────────────────
        # Concept: overall autonomic health — combines total HRV, vagal tone, and balance.
        # LF/HF Ratio is scored using the HEALTH scorer (1 = balanced, 0 = either extreme).
        # This means BOTH sympathetic dominance AND parasympathetic dominance are penalised.
        hhi_metrics = ["SDNN_ms", "RMSSD_ms", "pNN50_pct", "LF_HF_Ratio"]
        hhi_scores  = {
            "SDNN_ms":     s_sdnn,
            "RMSSD_ms":    s_rmssd,
            "pNN50_pct":   s_pnn50,
            "LF_HF_Ratio": s_ratio_health,   # 1 = balanced, 0 = either extreme
        }
        hhi_w = _pca_weights(df, hhi_metrics)

        df["HRV_Health_Index"] = (
            sum(hhi_scores[m] * hhi_w.get(m, 0.25) for m in hhi_metrics) * 100
        ).clip(0, 100).round(1)

        # ── Index 4: Stress Physiology Index (SPI) ───────────────────────────────
        # Concept: total physiological stress burden across all dimensions.
        # Builds on the already-computed HHI and SDI rather than raw metrics.
        # Inverted HHI: low health → high stress burden (1 − HHI/100).
        if "Stress_Score" in df.columns:
            spi_stress    = _ramp_and_cap(df["Stress_Score"], low=20, optimal=100)
            spi_sdi_norm  = df["Sympathetic_Dominance_Index"] / 100
            spi_hrv_inv   = 1 - df["HRV_Health_Index"] / 100  # low health = high SPI

            # Run PCA on these three component series to find data-driven weights
            spi_input_df = pd.DataFrame({
                "Stress_Score_scaled": spi_stress,
                "SDI_scaled":          spi_sdi_norm,
                "HHI_inverted":        spi_hrv_inv,
            })
            spi_w = _pca_weights(spi_input_df, list(spi_input_df.columns))

            df["Stress_Physiology_Index"] = (
                (spi_stress   * spi_w.get("Stress_Score_scaled", 1/3) +
                spi_sdi_norm * spi_w.get("SDI_scaled",          1/3) +
                spi_hrv_inv  * spi_w.get("HHI_inverted",        1/3)) * 100
            ).clip(0, 100).round(1)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICAL ORDERING CONSTANTS
# Single source of truth for the ordered labels of every derived category.
# All dashboard pages import these to ensure consistent ordering in charts.
# ─────────────────────────────────────────────────────────────────────────────

# BMI categories in ascending severity order
BMI_CATEGORY_ORDER = [
    "Underweight",
    "Healthy / Normal Weight",
    "Overweight",
    "Obesity Class 1",
    "Obesity Class 2",
    "Extreme Obesity Class 3",
]

# Age groups in ascending order
AGE_GROUP_ORDER = ["18–25", "25–40", "40–55", "55–65", "65+"]

# Stress levels in ascending severity order
STRESS_LEVEL_ORDER = [
    "Normal Stress",
    "Mild Stress",
    "Moderate Stress",
    "Severe Stress",
    "Extremely Severe Stress",
]

# Colour map for stress levels (green → red gradient by severity)
STRESS_LEVEL_COLORS = {
    "Normal Stress":           "#2ecc71",   # green
    "Mild Stress":             "#f1c40f",   # yellow
    "Moderate Stress":         "#e67e22",   # orange
    "Severe Stress":           "#e74c3c",   # red
    "Extremely Severe Stress": "#8e44ad",   # purple
}

# Colour map for BMI categories
BMI_CATEGORY_COLORS = {
    "Underweight":              "#3498db",   # blue
    "Healthy / Normal Weight":  "#2ecc71",   # green
    "Overweight":               "#f39c12",   # amber
    "Obesity Class 1":          "#e67e22",   # orange
    "Obesity Class 2":          "#e74c3c",   # red
    "Extreme Obesity Class 3":  "#8e44ad",   # purple
}


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_outliers_iqr(df: pd.DataFrame, columns: list):
    """
    Identify outlier records using the IQR (Tukey fence) method.

    A value is flagged as an outlier if it falls below Q1 − 1.5×IQR
    or above Q3 + 1.5×IQR.  This is the clinical standard for
    screening physiological outliers because it is robust to skewed
    distributions (common in HRV data).

    Parameters
    ----------
    df      : pd.DataFrame — the cleaned dataset
    columns : list[str]    — columns to check for outliers

    Returns
    -------
    outlier_mask : pd.Series[bool]  — True for any row that is outlying in ≥1 column
    outlier_info : dict             — per-column outlier statistics
    """
    outlier_mask = pd.Series(False, index=df.index)
    outlier_info = {}

    for col in columns:
        if col not in df.columns:
            continue
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr   # lower Tukey fence
        hi  = q3 + 1.5 * iqr   # upper Tukey fence

        mask = (df[col] < lo) | (df[col] > hi)
        outlier_mask = outlier_mask | mask

        outlier_info[col] = {
            "count":       int(mask.sum()),
            "pct":         round(mask.sum() / len(df) * 100, 1),
            "lower_fence": round(lo, 2),
            "upper_fence": round(hi, 2),
            "below_lower": int((df[col] < lo).sum()),
            "above_upper": int((df[col] > hi).sum()),
        }

    return outlier_mask, outlier_info


def get_outliers_zscore(df: pd.DataFrame, columns: list, threshold: float = 3.0):
    """
    Identify outliers using the Z-score method.

    A value is flagged if |z-score| > threshold (default 3.0 standard
    deviations from the mean, per convention).  This method assumes
    approximate normality, so it complements the IQR method which
    does not require normality.

    Parameters
    ----------
    df        : pd.DataFrame
    columns   : list[str]
    threshold : float — z-score threshold (default 3.0)

    Returns
    -------
    outlier_mask : pd.Series[bool]
    outlier_info : dict
    """
    outlier_mask = pd.Series(False, index=df.index)
    outlier_info = {}

    for col in columns:
        if col not in df.columns:
            continue

        valid       = df[col].dropna()
        z_scores    = np.abs(stats.zscore(valid))
        col_outlier = pd.Series(False, index=df.index)
        col_outlier.loc[valid.index[z_scores > threshold]] = True
        outlier_mask = outlier_mask | col_outlier

        outlier_info[col] = {
            "count": int(col_outlier.sum()),
            "pct":   round(col_outlier.sum() / len(df) * 100, 1),
            "max_z": round(float(z_scores.max()), 2) if len(z_scores) > 0 else 0.0,
        }

    return outlier_mask, outlier_info


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def get_descriptive_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Compute a comprehensive descriptive statistics table for numeric columns.

    Includes: count, mean, median, std, variance, min, max, Q1, Q3, IQR,
    skewness, and kurtosis.

    Skewness interpretation:
        |skewness| < 0.5 → approximately symmetric
        0.5–1.0          → moderate skew
        > 1.0            → high skew (common in HRV frequency-domain data)

    Kurtosis interpretation (excess kurtosis):
        ≈ 0  → mesokurtic (normal-like)
        > 0  → leptokurtic (heavy tails)
        < 0  → platykurtic (light tails)

    Parameters
    ----------
    df      : pd.DataFrame
    columns : list[str] — numeric columns to include

    Returns
    -------
    pd.DataFrame with one row per variable and one column per statistic.
    """
    records = {}

    for col in columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)

        records[col] = {
            "Count":       int(len(s)),
            "Mean":        round(s.mean(), 3),
            "Median":      round(s.median(), 3),
            "Std Dev":     round(s.std(), 3),
            "Variance":    round(s.var(), 3),
            "Min":         round(s.min(), 3),
            "Max":         round(s.max(), 3),
            "Q1 (25%)":    round(q1, 3),
            "Q3 (75%)":    round(q3, 3),
            "IQR":         round(q3 - q1, 3),
            "Skewness":    round(s.skew(), 3),
            "Kurtosis":    round(s.kurtosis(), 3),
        }

    return pd.DataFrame(records).T


# ─────────────────────────────────────────────────────────────────────────────
# PHYSIOLOGICAL RANGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_physiological_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check that each physiological variable falls within its plausible range.

    Out-of-range values may indicate:
    - Data entry errors (e.g., weight entered in lbs instead of kg)
    - Instrument calibration issues
    - Extreme outliers that should be reviewed clinically

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with columns: Variable, Expected Range, Out-of-Range Count, Status.
    """
    # (min, max) plausibility bounds — deliberately wide to catch only gross errors
    range_checks = {
        "Age":            (18,   105),
        "Height_cm":      (100,  220),
        "Weight_kg":      (30,   200),
        "BMI":            (10,    60),
        "Heart_Rate_bpm": (25,   220),
        "Mean_RR_ms":     (280, 2400),
        "SDNN_ms":        (0,    300),
        "RMSSD_ms":       (0,    300),
        "pNN20_pct":      (0,    100),
        "pNN50_pct":      (0,    100),
        "LF_Power_ms2":   (0,  15000),
        "HF_Power_ms2":   (0,  15000),
        "LF_HF_Ratio":    (0,     25),
        "Stress_Score":   (0,    100),
    }

    rows = []
    for col, (lo, hi) in range_checks.items():
        if col not in df.columns:
            continue
        oor = int(((df[col] < lo) | (df[col] > hi)).sum())
        rows.append(
            {
                "Variable":           col,
                "Expected Range":     f"{lo} – {hi}",
                "Out-of-Range Count": oor,
                "Status": "✅ OK" if oor == 0 else f"⚠️ {oor} value(s)",
            }
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PERCENTILE RANKING
# ─────────────────────────────────────────────────────────────────────────────

def compute_percentile_ranks(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Add percentile rank columns for each specified metric.

    A patient in the 80th percentile for RMSSD has higher vagal tone than
    80% of the cohort — clinically useful for patient profiling.

    Parameters
    ----------
    df      : pd.DataFrame
    columns : list[str]

    Returns
    -------
    pd.DataFrame with additional '{col}_Pct_Rank' columns.
    """
    df_out = df.copy()
    for col in columns:
        if col in df_out.columns and pd.api.types.is_numeric_dtype(df_out[col]):
            df_out[f"{col}_Pct_Rank"] = (
                df_out[col].rank(pct=True) * 100
            ).round(1)
    return df_out
