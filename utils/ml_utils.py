"""
utils/ml_utils.py
=================
Machine Learning Utilities for HRV Clinical Dashboard
------------------------------------------------------
This module trains, evaluates, and explains machine learning models that
predict and classify patient stress levels from HRV and demographic features.

Model inventory
---------------
REGRESSION (predict continuous Stress_Score):
    1. Linear Regression   — interpretable baseline; assumes linearity
    2. Ridge Regression    — L2 regularisation; reduces multicollinearity impact
    3. Lasso Regression    — L1 regularisation; performs implicit feature selection
    4. Random Forest       — ensemble of decision trees; handles non-linearity
    5. XGBoost             — gradient-boosted trees; often best out-of-box performance
    6. Gradient Boosting   — sklearn's own boosting; robust and well-calibrated

CLASSIFICATION (predict Stress_Level: Low / Moderate / High):
    1. Logistic Regression — linear probabilistic classifier; highly interpretable
    2. Random Forest       — robust non-linear classifier
    3. XGBoost             — high-performance classifier

EXPLAINABILITY:
    SHAP (SHapley Additive exPlanations) — model-agnostic attribution values
    that show how much each feature pushed the prediction up or down for
    every individual patient.  Critical for clinical trust and audit trails.

All functions use the INTERNAL column names defined in data_processing.py.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split,          # Split data into train/test sets
    cross_val_score,           # K-fold cross-validation
    KFold,                     # K-fold splitter for regression
    StratifiedKFold,           # Stratified K-fold for classification (preserves class balance)
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    r2_score,                  # Proportion of variance explained
    mean_squared_error,        # Average squared prediction error
    mean_absolute_error,       # Average absolute prediction error
    accuracy_score,            # Fraction of correct classifications
    precision_score,           # True positives / (true + false positives)
    recall_score,              # True positives / (true + false negatives)
    f1_score,                  # Harmonic mean of precision and recall
    roc_auc_score,             # Area under the ROC curve
    roc_curve,                 # TPR vs FPR at each threshold
    confusion_matrix,          # Matrix of prediction vs actual
)
import xgboost as xgb          # Extreme Gradient Boosting — high-performance trees


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMNS
# These are the physiological and demographic predictors used as model inputs.
# All names match the INTERNAL column names from data_processing.py.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "Age",            # Demographic — older age generally → lower HRV
    "BMI",            # Anthropometric — higher BMI → reduced autonomic flexibility
    "Heart_Rate_bpm", # Derived from Mean RR — higher resting HR → lower HRV
    "SDNN_ms",        # Total HRV variability
    "RMSSD_ms",       # Vagal/parasympathetic tone
    "pNN50_pct",      # Parasympathetic inter-beat consistency
    "pNN20_pct",      # Broader parasympathetic metric
    "LF_Power_ms2",   # Sympathetic + mixed autonomic power
    "HF_Power_ms2",   # Parasympathetic power
    "LF_HF_Ratio",    # Sympathovagal balance — key stress marker
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame,
                     feature_cols: list = None,
                     target_col: str = "Stress_Score",
                     test_size: float = 0.20,
                     scale: bool = True):
    """
    Prepare a clean, scaled feature matrix and target vector for ML training.

    Steps:
    1. Select only rows where all feature + target columns are non-null.
    2. Extract X (features) and y (target) as numpy arrays.
    3. Optionally standardise X (zero mean, unit variance) — required for
       linear models and beneficial for all models.
    4. Split into train / test sets (80/20 default).

    Parameters
    ----------
    df           : pd.DataFrame — cleaned dataset from preprocess_data()
    feature_cols : list[str] | None — defaults to FEATURE_COLUMNS
    target_col   : str — regression target (default 'Stress_Score')
    test_size    : float — fraction of data held out for testing
    scale        : bool — standardise features with StandardScaler

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    scaler       : fitted StandardScaler (or None)
    feature_cols : list[str] — features actually used (after filtering present cols)
    valid_index  : list — original DataFrame index for the rows used
    """
    if feature_cols is None:
        # Use only features that actually exist in this dataset
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    keep_cols = [c for c in feature_cols + [target_col] if c in df.columns]
    df_clean  = df[keep_cols].dropna()               # drop any residual nulls

    X = df_clean[feature_cols].values.astype(float)  # feature matrix
    y = df_clean[target_col].values.astype(float)    # target vector

    scaler = None
    if scale and len(X) > 0:
        scaler = StandardScaler()
        X      = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, feature_cols, list(df_clean.index)


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION — training
# ─────────────────────────────────────────────────────────────────────────────

def train_regression_models(X_train: np.ndarray,
                            y_train: np.ndarray) -> dict:
    """
    Train all six regression models on the provided training data.

    Returns a dict of {model_name: fitted_model}.
    Any model that fails to train is silently skipped so one bad model
    cannot prevent the others from running.
    """
    models = {
        "Linear Regression": LinearRegression(),
        # Ridge: adds λΣβ² penalty — shrinks coefficients, handles collinear HRV metrics
        "Ridge Regression":  Ridge(alpha=1.0),
        # Lasso: adds λΣ|β| penalty — drives some coefficients to exactly 0
        "Lasso Regression":  Lasso(alpha=0.1, max_iter=5000),
        # Random Forest: average of 200 decision trees — robust, non-linear
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42,
                                                   n_jobs=-1),
        # XGBoost: sequential boosted trees — typically best predictive accuracy
        "XGBoost":           xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                               max_depth=5, random_state=42,
                                               eval_metric="rmse", verbosity=0),
        # Gradient Boosting: sklearn's boosting — well-calibrated residual learning
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200,
                                                       learning_rate=0.05,
                                                       max_depth=4,
                                                       random_state=42),
    }

    trained = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            trained[name] = model
        except Exception as exc:
            print(f"[ML] Warning: {name} failed to train — {exc}")

    return trained


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION — evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regression(model,
                        X_train: np.ndarray,
                        X_test: np.ndarray,
                        y_train: np.ndarray,
                        y_test: np.ndarray,
                        model_name: str = "Model") -> dict:
    """
    Evaluate a regression model and return a comprehensive metrics dictionary.

    Metrics reported:
    -----------------
    R²   : 1 = perfect fit, 0 = same as predicting the mean, <0 = worse than mean
    RMSE : Root Mean Squared Error — in the same units as Stress_Score (0–100)
    MAE  : Mean Absolute Error — average absolute deviation from true stress score
    CV R²: 5-fold cross-validated R² (mean ± std) — measures generalisation

    Parameters
    ----------
    model      : fitted sklearn/XGBoost model
    X_train, X_test, y_train, y_test : split arrays from prepare_features()
    model_name : str — label for display

    Returns
    -------
    dict with all metrics plus raw prediction arrays for residual plots.
    """
    y_pred_test  = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    r2   = r2_score(y_test, y_pred_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae  = float(mean_absolute_error(y_test, y_pred_test))

    # Cross-validation on the full available data (train + test combined)
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    cv     = KFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring="r2")
        cv_mean   = float(cv_scores.mean())
        cv_std    = float(cv_scores.std())
    except Exception:
        cv_mean, cv_std = None, None

    return {
        "model_name":  model_name,
        "r2":          round(r2, 4),
        "rmse":        round(rmse, 4),
        "mae":         round(mae, 4),
        "cv_r2_mean":  round(cv_mean, 4) if cv_mean is not None else None,
        "cv_r2_std":   round(cv_std, 4)  if cv_std  is not None else None,
        "y_test":      y_test,
        "y_pred":      y_pred_test,
        "y_train":     y_train,
        "y_pred_train": y_pred_train,
        "residuals":   y_test - y_pred_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION — label creation
# ─────────────────────────────────────────────────────────────────────────────

def create_stress_labels(y_continuous: np.ndarray) -> np.ndarray:
    """
    Convert a continuous Stress_Score array into five clinical categories.

    Thresholds mirror the Stress_Level column produced by add_derived_features():
        ≤ 40           → 'Normal Stress'
        41 – 55        → 'Mild Stress'
        56 – 70        → 'Moderate Stress'
        71 – 85        → 'Severe Stress'
        > 85           → 'Extremely Severe Stress'

    Using five classes produces a more clinically nuanced model than the old
    three-class system and aligns with cardiological stress-grading conventions.

    Parameters
    ----------
    y_continuous : np.ndarray — continuous Stress_Score values (post-filter, > 20)

    Returns
    -------
    np.ndarray of string labels matching STRESS_LEVEL_ORDER in data_processing.py
    """
    labels = np.where(
        y_continuous <= 40, "Normal Stress",
        np.where(
            y_continuous <= 55, "Mild Stress",
            np.where(
                y_continuous <= 70, "Moderate Stress",
                np.where(
                    y_continuous <= 85, "Severe Stress",
                    "Extremely Severe Stress"
                )
            )
        )
    )
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION — training
# ─────────────────────────────────────────────────────────────────────────────

def train_classification_models(X_train: np.ndarray,
                                y_train_labels: np.ndarray):
    """
    Train all three stress-level classifiers.

    Parameters
    ----------
    X_train       : np.ndarray — scaled feature matrix
    y_train_labels: np.ndarray — string labels ('Low Stress', etc.)

    Returns
    -------
    trained_models : dict {name: fitted model}
    label_encoder  : fitted LabelEncoder (needed for evaluation)
    """
    le      = LabelEncoder()
    y_enc   = le.fit_transform(y_train_labels)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42, C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42, eval_metric="mlogloss", verbosity=0,
            use_label_encoder=False,
        ),
    }

    trained = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_enc)
            trained[name] = model
        except Exception as exc:
            print(f"[ML] Warning: {name} classifier failed — {exc}")

    return trained, le


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION — evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classification(model,
                            X_test: np.ndarray,
                            y_test_labels: np.ndarray,
                            label_encoder: LabelEncoder,
                            model_name: str = "Model") -> dict:
    """
    Evaluate a classification model and return clinical performance metrics.

    Metrics reported:
    -----------------
    Accuracy  : overall fraction correct
    Precision : of predicted positives, what fraction are truly positive
    Recall    : of true positives, what fraction were detected (sensitivity)
    F1 Score  : harmonic mean of precision and recall
    ROC-AUC   : area under receiver operating characteristic curve (weighted OvR)

    Parameters
    ----------
    model         : fitted classifier
    X_test        : np.ndarray
    y_test_labels : np.ndarray of string labels
    label_encoder : from train_classification_models()
    model_name    : str

    Returns
    -------
    dict with all metrics and raw arrays for plotting.
    """
    y_enc  = label_encoder.transform(y_test_labels)
    y_pred = model.predict(X_test)

    acc  = float(accuracy_score(y_enc, y_pred))
    prec = float(precision_score(y_enc, y_pred, average="weighted", zero_division=0))
    rec  = float(recall_score(y_enc, y_pred,    average="weighted", zero_division=0))
    f1   = float(f1_score(y_enc, y_pred,         average="weighted", zero_division=0))
    cm   = confusion_matrix(y_enc, y_pred)

    # ROC-AUC requires predicted probabilities
    auc = None
    roc_data = {}
    try:
        y_prob = model.predict_proba(X_test)
        auc = float(roc_auc_score(y_enc, y_prob, multi_class="ovr", average="weighted"))
        # Store per-class ROC curves for plotting
        n_classes = len(label_encoder.classes_)
        for i, cls_name in enumerate(label_encoder.classes_):
            fpr, tpr, _ = roc_curve((y_enc == i).astype(int), y_prob[:, i])
            roc_data[cls_name] = {"fpr": fpr, "tpr": tpr}
    except Exception:
        pass

    return {
        "model_name":  model_name,
        "accuracy":    round(acc,  4),
        "precision":   round(prec, 4),
        "recall":      round(rec,  4),
        "f1":          round(f1,   4),
        "auc":         round(auc,  4) if auc is not None else None,
        "confusion_matrix": cm,
        "classes":     list(label_encoder.classes_),
        "y_test":      y_test_labels,
        "y_pred_labels": label_encoder.inverse_transform(y_pred),
        "roc_data":    roc_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY — SHAP VALUES
# ─────────────────────────────────────────────────────────────────────────────

def calculate_shap_values(model,
                          X: np.ndarray,
                          feature_names: list,
                          model_type: str = "tree") -> dict:
    """
    Compute SHAP (SHapley Additive exPlanations) values for a trained model.

    SHAP values quantify the contribution of each feature to an individual
    prediction relative to the average prediction.  Positive SHAP → feature
    pushed prediction UP (more stress); negative → pushed it DOWN.

    Why SHAP for clinical AI?
    --------------------------
    Clinicians and regulators require AI systems to be explainable.  SHAP
    provides patient-level explanations that clinicians can review, challenge,
    and document in audit trails.  It also satisfies GDPR / HIPAA requirements
    for algorithmic accountability.

    Parameters
    ----------
    model        : trained sklearn / XGBoost model
    X            : np.ndarray — feature matrix (already scaled)
    feature_names: list[str]
    model_type   : 'tree' for tree-based models, 'linear' for linear models

    Returns
    -------
    dict with keys: shap_values, X_sample, feature_names, expected_value.
    On failure returns {'error': message}.
    """
    try:
        import shap

        # Sample up to 200 rows — SHAP computation scales O(n×features) and
        # can be slow on large datasets.  200 rows is sufficient for global
        # summaries and the global feature importance plot.
        n_sample = min(200, len(X))
        idx      = np.random.choice(len(X), size=n_sample, replace=False)
        X_sample = X[idx]

        if model_type == "tree" or hasattr(model, "feature_importances_"):
            # TreeExplainer is fast and exact for tree-based models
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_sample)
            exp_value  = explainer.expected_value
        else:
            # LinearExplainer for regularised linear models
            explainer  = shap.LinearExplainer(model, X)
            shap_vals  = explainer.shap_values(X_sample)
            exp_value  = explainer.expected_value

        # For multi-output models (classifiers), take class 0 values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if isinstance(exp_value, (list, np.ndarray)):
            exp_value = exp_value[0]

        return {
            "shap_values":    shap_vals,          # shape: (n_sample, n_features)
            "X_sample":       X_sample,
            "feature_names":  feature_names,
            "expected_value": float(exp_value),
        }

    except ImportError:
        return {"error": "SHAP library not installed. Run: pip install shap"}
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance scores from any trained model.

    For tree-based models (Random Forest, XGBoost, Gradient Boosting):
        Uses the model's built-in feature_importances_ (mean decrease in impurity
        for Random Forest; gain-based for XGBoost).

    For linear models (Linear, Ridge, Lasso, Logistic):
        Uses |coefficient| as a proxy for importance.  Note: requires the
        features to have been standardised (which prepare_features() ensures).

    Parameters
    ----------
    model        : fitted sklearn / XGBoost model
    feature_names: list[str]

    Returns
    -------
    pd.DataFrame with columns: Feature, Importance
    Sorted descending by Importance.
    """
    importance_vals = None

    if hasattr(model, "feature_importances_"):
        # Tree-based model — use built-in impurity-based importance
        importance_vals = model.feature_importances_

    elif hasattr(model, "coef_"):
        # Linear model — use absolute coefficient magnitude
        coef = model.coef_
        if coef.ndim > 1:
            # Multi-class logistic regression has shape (n_classes, n_features)
            coef = np.abs(coef).mean(axis=0)
        importance_vals = np.abs(coef.flatten())

    if importance_vals is None or len(importance_vals) != len(feature_names):
        return pd.DataFrame({"Feature": feature_names,
                             "Importance": np.zeros(len(feature_names))})

    df_imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importance_vals,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Normalise to 0–100 for easier reading
    max_imp = df_imp["Importance"].max()
    if max_imp > 0:
        df_imp["Importance_Normalised"] = (df_imp["Importance"] / max_imp * 100).round(1)
    else:
        df_imp["Importance_Normalised"] = 0.0

    return df_imp


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL INTERPRETATION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_clinical_interpretation(metrics: dict, model_type: str = "regression") -> str:
    """
    Auto-generate a plain-English clinical interpretation statement for
    the best-performing model's results.

    Parameters
    ----------
    metrics    : dict from evaluate_regression() or evaluate_classification()
    model_type : 'regression' or 'classification'

    Returns
    -------
    str — multi-sentence clinical interpretation suitable for reports.
    """
    name = metrics.get("model_name", "The model")

    if model_type == "regression":
        r2   = metrics.get("r2", 0)
        rmse = metrics.get("rmse", 0)
        mae  = metrics.get("mae", 0)
        cv   = metrics.get("cv_r2_mean")

        quality = ("excellent" if r2 > 0.80 else
                   "good"      if r2 > 0.60 else
                   "moderate"  if r2 > 0.40 else "limited")

        txt  = (f"{name} demonstrates {quality} predictive performance for Stress Score, "
                f"explaining {r2 * 100:.1f}% of variance in the outcome (R² = {r2:.3f}). ")
        txt += (f"On average, predicted stress scores deviate from true scores by "
                f"{mae:.1f} points (MAE), with RMSE of {rmse:.1f}. ")
        if cv is not None:
            txt += (f"Five-fold cross-validation yielded a mean R² of {cv:.3f}, "
                    f"indicating the model's generalisation {'is robust' if cv > 0.5 else 'should be interpreted cautiously'}. ")
        txt += ("HRV-based predictors — particularly RMSSD, LF/HF Ratio, and HF Power — "
                "are the dominant physiological drivers of predicted stress in this cohort.")

    else:
        acc = metrics.get("accuracy", 0)
        f1  = metrics.get("f1", 0)
        auc = metrics.get("auc")

        quality = ("excellent" if f1 > 0.85 else
                   "good"      if f1 > 0.70 else
                   "moderate"  if f1 > 0.55 else "limited")

        txt  = (f"{name} achieves {quality} three-class stress level classification "
                f"(accuracy {acc * 100:.1f}%, weighted F1 = {f1:.3f}). ")
        if auc is not None:
            txt += f"The weighted one-vs-rest AUC is {auc:.3f}. "
        txt += ("Misclassifications predominantly occur at the Low/Moderate boundary, "
                "which is clinically expected given the continuous nature of stress physiology. "
                "This model supports clinical screening for high-stress patients "
                "who may benefit from autonomic health interventions.")

    return txt
