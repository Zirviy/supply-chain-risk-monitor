"""
src/ml/train.py
Supply Chain Risk Monitor — XGBoost Training Pipeline

Trains two models on feature_matrix.csv using weakly supervised labels:
  Model 1 — Binary classifier: disruption risk in next 14 days (0/1)
  Model 2 — Severity regressor: predicted severity 1–5 (fit on label=1 rows only)

Labelling approach: programmatic/weak supervision (Snorkel framework rationale,
  Ratner et al. 2017 — https://arxiv.org/abs/1711.10160). No ground truth labels
  exist for this dataset; heuristics derived from feature values serve as
  reasonable proxies for disruption occurrence.

Temporal train/test split used (NOT random) because supply chain features have
  strong temporal autocorrelation — random splitting would leak future information
  into training and inflate metrics. Reference: Bergmeir & Benitez (2012),
  "On the use of cross-validation for time series predictor evaluation",
  Information Sciences, 191, 192–213.

Run from project root:
  python src/ml/train.py

Outputs (all to models/):
  classifier.json          — XGBoost native format
  regressor.json           — XGBoost native format
  feature_columns.json     — ordered list of feature names used for inference
  shap_summary.csv         — mean |SHAP value| per feature (importance proxy)
  train_metrics.json       — AUC, precision, recall, F1, MAE, RMSE
  train_feature_stats.json — per-feature mean, std, decile bins for drift_monitor.py
"""

import os
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (always os.path.join — Windows compatibility)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

FEATURE_MATRIX_PATH = os.path.join(DATA_PROCESSED, "feature_matrix.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants — all hardcoded values have cited sources
# ---------------------------------------------------------------------------

# Temporal split ratio: 80% train / 20% test
# Standard practice for time-series ML; no specific citation needed —
# referenced in Hyndman & Athanasopoulos "Forecasting: Principles and Practice"
# (3rd ed., 2021), Chapter 3.4.
TRAIN_RATIO = 0.80

# Disruption labelling thresholds (weakly supervised — Ratner et al. 2017)
# Threshold values calibrated to the observed severity distribution in
# feature_matrix.csv: mean=0.694, std=0.086, range 0.44–0.86
# (documented in feature_engineer.py outputs, Jun 2026 run).
SEVERITY_HIGH_THRESHOLD = 0.75      # severity_max_14d above this → label=1
SEVERITY_MEAN_THRESHOLD = 0.65      # used with event_count condition
EVENT_COUNT_THRESHOLD = 5           # minimum events for mean-based rule
GOLDSTEIN_SLOPE_THRESHOLD = -0.5    # worsening trend (negative = worse)
GOLDSTEIN_EVENT_THRESHOLD = 3       # minimum events for slope-based rule

# Severity label bucket boundaries (maps severity_mean_14d → integer 1–5)
# Bucket edges chosen to span the observed severity distribution uniformly.
SEVERITY_BINS = [0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
SEVERITY_LABELS_INT = [1, 2, 3, 4, 5]

# Sentinel clip: days_since_high_severity uses 999 to mean "never seen event".
# 999 vs 998 carries no information. Clip to 30 so the feature encodes
# "within last month" vs "not recently/never", which IS meaningful.
DAYS_SINCE_CLIP = 30  # documented in feature_engineer.py IMPORTANT NOTES

# Columns to drop before training (identifiers or zero-variance)
DROP_COLS = [
    "node_id",          # string identifier — not a feature
    "snapshot_date",    # date identifier — not a feature (temporal split done before drop)
    "chokepoint_weight",  # all 0.0 for trade nodes (zero variance; documented in
                          # feature_engineer.py: "chokepoints are separate node type")
]

# Label-derived columns that must NOT be features (data leakage guard)
LABEL_DERIVED_COLS = [
    "disruption_label",
    "severity_label",
]

# MLflow experiment name
MLFLOW_EXPERIMENT = "supply-chain-risk"

# Decision threshold for classifier binary predictions (precision/recall report)
# 0.5 is the standard default; adjust in predict.py for operational tuning.
DECISION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Step 1 — Load and validate feature matrix
# ---------------------------------------------------------------------------

def load_feature_matrix(path: str) -> pd.DataFrame:
    """Load feature_matrix.csv and run basic sanity checks."""
    log.info("Loading feature matrix from %s", path)
    df = pd.read_csv(path, parse_dates=["snapshot_date"])

    log.info("  Shape: %d rows × %d cols", *df.shape)
    log.info("  Snapshot date range: %s → %s",
             df["snapshot_date"].min().date(), df["snapshot_date"].max().date())
    log.info("  Unique nodes: %d", df["node_id"].nunique())

    null_counts = df.isnull().sum()
    if null_counts.any():
        log.warning("Null values found:\n%s", null_counts[null_counts > 0])
    else:
        log.info("  Zero nulls — consistent with feature_engineer.py output")

    return df


# ---------------------------------------------------------------------------
# Step 2 — Derive labels (weakly supervised, Ratner et al. 2017)
# ---------------------------------------------------------------------------

def derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add disruption_label (0/1) and severity_label (1–5, only for label=1 rows).

    Labelling rules (programmatic supervision):
      disruption_label = 1 if ANY of:
        R1: severity_max_14d > 0.75
        R2: event_count_14d >= 5 AND severity_mean_14d > 0.65
        R3: goldstein_slope_14d < -0.5 AND event_count_14d >= 3
      disruption_label = 0 otherwise

    severity_label (1–5, label=1 rows only):
      Bucketed from severity_mean_14d using fixed bin edges.
    """
    log.info("Deriving weakly supervised labels (Ratner et al. 2017)...")

    rule1 = df["severity_max_14d"] > SEVERITY_HIGH_THRESHOLD
    rule2 = (df["event_count_14d"] >= EVENT_COUNT_THRESHOLD) & \
            (df["severity_mean_14d"] > SEVERITY_MEAN_THRESHOLD)
    rule3 = (df["goldstein_slope_14d"] < GOLDSTEIN_SLOPE_THRESHOLD) & \
            (df["event_count_14d"] >= GOLDSTEIN_EVENT_THRESHOLD)

    df["disruption_label"] = (rule1 | rule2 | rule3).astype(int)

    pos = df["disruption_label"].sum()
    neg = len(df) - pos
    pct = 100 * pos / len(df)
    log.info("  Positive labels: %d (%.2f%%)", pos, pct)
    log.info("  Negative labels: %d (%.2f%%)", neg, 100 - pct)
    log.info("  Rule 1 triggered: %d  Rule 2: %d  Rule 3: %d",
             rule1.sum(), rule2.sum(), rule3.sum())

    if pct < 1.0:
        log.warning("  Positive rate below 1%% — check event coverage in feature matrix")
    if pct > 15.0:
        log.warning("  Positive rate above 15%% — labelling rules may be too permissive")

    # Severity label for disrupted rows only
    df["severity_label"] = np.nan
    disrupted_mask = df["disruption_label"] == 1
    df.loc[disrupted_mask, "severity_label"] = pd.cut(
        df.loc[disrupted_mask, "severity_mean_14d"],
        bins=SEVERITY_BINS,
        labels=SEVERITY_LABELS_INT,
        right=True,
        include_lowest=False,
    ).astype(float)

    # Rows with label=1 but severity_mean_14d exactly at boundary or below 0.60
    # can produce NaN severity_label. Fill with 1 (minimum severity).
    null_sev = df.loc[disrupted_mask, "severity_label"].isna().sum()
    if null_sev > 0:
        log.warning("  %d disrupted rows have no severity bucket — filled with 1", null_sev)
        df.loc[disrupted_mask, "severity_label"] = \
            df.loc[disrupted_mask, "severity_label"].fillna(1.0)

    log.info("  Severity distribution (label=1 rows):\n%s",
             df.loc[disrupted_mask, "severity_label"].value_counts().sort_index())

    return df


# ---------------------------------------------------------------------------
# Step 3 — Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing steps documented in the train.py spec:
      - Clip days_since_high_severity to DAYS_SINCE_CLIP (30)
      - All other features are valid numeric ranges; no scaling needed
        (XGBoost is tree-based / scale-invariant)
    """
    log.info("Preprocessing features...")

    # Clip sentinel value (999 = "never seen event" → meaningless beyond ~30d)
    before_clip = df["days_since_high_severity"].max()
    df["days_since_high_severity"] = df["days_since_high_severity"].clip(upper=DAYS_SINCE_CLIP)
    log.info("  days_since_high_severity clipped: max was %d → now capped at %d",
             before_clip, DAYS_SINCE_CLIP)

    return df


# ---------------------------------------------------------------------------
# Step 4 — Temporal train/test split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    """
    Chronological split — NOT random.
    Train: snapshot_date < cutoff (first 80% of unique dates)
    Test:  snapshot_date >= cutoff (most recent 20%)

    Rationale: supply chain event features have temporal autocorrelation;
    random splitting would leak future signal into training.
    See: Bergmeir & Benitez (2012), Information Sciences 191:192–213.
    """
    sorted_dates = sorted(df["snapshot_date"].unique())
    cutoff_idx = int(len(sorted_dates) * train_ratio)
    cutoff_date = sorted_dates[cutoff_idx]

    train_df = df[df["snapshot_date"] < cutoff_date].copy()
    test_df = df[df["snapshot_date"] >= cutoff_date].copy()

    log.info("Temporal split at cutoff date: %s", cutoff_date.date())
    log.info("  Train: %d rows (%d unique dates)",
             len(train_df), train_df["snapshot_date"].nunique())
    log.info("  Test:  %d rows (%d unique dates)",
             len(test_df), test_df["snapshot_date"].nunique())

    return train_df, test_df, cutoff_date


# ---------------------------------------------------------------------------
# Step 5 — Build X / y arrays
# ---------------------------------------------------------------------------

def build_feature_array(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    return df[feature_cols].copy()


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return ordered list of feature columns (everything except identifiers and labels)."""
    exclude = set(DROP_COLS + LABEL_DERIVED_COLS)
    cols = [c for c in df.columns if c not in exclude]
    return cols


# ---------------------------------------------------------------------------
# Step 6 — Train classifier (Model 1)
# ---------------------------------------------------------------------------

def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> xgb.XGBClassifier:
    """
    XGBoost binary classifier for disruption risk in next 14 days.

    scale_pos_weight = neg_count / pos_count corrects for class imbalance
    (expected ~2–5% positive labels). This is XGBoost's recommended approach
    for imbalanced datasets. Source: XGBoost documentation,
    https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
    (section: "Handling Imbalanced Dataset")
    """
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    spw = neg_count / max(pos_count, 1)  # guard against zero division
    log.info("Classifier: neg=%d, pos=%d, scale_pos_weight=%.2f", neg_count, pos_count, spw)

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,       # class imbalance correction (XGBoost docs)
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
    )

    log.info("Fitting classifier (n_estimators=300, early_stopping_rounds=20)...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    log.info("  Best iteration: %d", clf.best_iteration)

    return clf


# ---------------------------------------------------------------------------
# Step 7 — Train regressor (Model 2)
# ---------------------------------------------------------------------------

def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> xgb.XGBRegressor:
    """
    XGBoost severity regressor (1–5 scale).
    Fitted ONLY on rows where disruption_label == 1, per spec.
    """
    log.info("Regressor: train rows=%d, test rows=%d", len(X_train), len(X_test))

    if len(X_train) < 10:
        log.warning("Very few positive training rows (%d) — regressor may be unreliable",
                    len(X_train))

    reg = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbosity=0,
    )

    log.info("Fitting regressor (n_estimators=200)...")
    reg.fit(X_train, y_train)

    return reg


# ---------------------------------------------------------------------------
# Step 8 — Evaluate
# ---------------------------------------------------------------------------

def evaluate_classifier(
    clf: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, proba)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    log.info("Classifier metrics (threshold=%.2f):", threshold)
    log.info("  ROC-AUC:   %.4f  (expected 0.70–0.85)", auc)
    log.info("  Precision: %.4f", prec)
    log.info("  Recall:    %.4f", rec)
    log.info("  F1:        %.4f", f1)

    if auc < 0.60:
        log.warning("AUC below 0.60 — check label quality or feature coverage")

    return {"auc": round(auc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


def evaluate_regressor(
    reg: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    preds = reg.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    log.info("Regressor metrics:")
    log.info("  MAE:  %.4f", mae)
    log.info("  RMSE: %.4f", rmse)

    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}


# ---------------------------------------------------------------------------
# Step 9 — SHAP feature importance
# ---------------------------------------------------------------------------

def compute_shap_importance(
    clf: xgb.XGBClassifier,
    X_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute SHAP values using TreeExplainer on classifier.
    Mean |SHAP| per feature is a reliable model-agnostic importance measure.
    Reference: Lundberg & Lee (2017), "A Unified Approach to Interpreting
    Model Predictions", NeurIPS 2017. https://arxiv.org/abs/1705.07874
    """
    log.info("Computing SHAP values (TreeExplainer)...")

    # Use at most 2000 rows for SHAP computation — sufficient for mean |SHAP|
    # and avoids OOM on large training sets.
    sample = X_train.sample(min(2000, len(X_train)), random_state=42)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(sample)

    # For binary classification XGBoost returns array of shape (n, features)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (disruption)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": X_train.columns.tolist(),
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    log.info("Top 10 features by mean |SHAP|:\n%s",
             shap_df.head(10).to_string(index=False))

    return shap_df


# ---------------------------------------------------------------------------
# Step 10 — Save training feature statistics (for drift_monitor.py)
# ---------------------------------------------------------------------------

def save_feature_stats(X_train: pd.DataFrame, path: str) -> None:
    """
    Save per-feature mean, std, and decile bins from training data.
    drift_monitor.py reads this to compute PSI (Population Stability Index).

    PSI formula: Σ (actual% - expected%) × ln(actual%/expected%)
    PSI > 0.2 on any key feature → data drift warning.
    Source: Yurdakul (2018), "Statistical Properties of Population Stability Index",
    University of Michigan Working Paper.
    """
    log.info("Saving training feature statistics for drift monitoring...")
    stats = {}
    for col in X_train.columns:
        series = X_train[col].dropna()
        deciles = np.percentile(series, np.arange(0, 101, 10)).tolist()
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "decile_bins": deciles,
        }
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("  Saved to %s", path)


# ---------------------------------------------------------------------------
# Step 11 — Save all outputs
# ---------------------------------------------------------------------------

def save_outputs(
    clf: xgb.XGBClassifier,
    reg: xgb.XGBRegressor,
    feature_cols: list,
    shap_df: pd.DataFrame,
    clf_metrics: dict,
    reg_metrics: dict,
) -> None:
    clf_path = os.path.join(MODELS_DIR, "classifier.json")
    reg_path = os.path.join(MODELS_DIR, "regressor.json")
    feat_path = os.path.join(MODELS_DIR, "feature_columns.json")
    shap_path = os.path.join(MODELS_DIR, "shap_summary.csv")
    metrics_path = os.path.join(MODELS_DIR, "train_metrics.json")

    clf.save_model(clf_path)
    log.info("Saved classifier → %s", clf_path)

    reg.save_model(reg_path)
    log.info("Saved regressor  → %s", reg_path)

    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info("Saved feature columns → %s", feat_path)

    shap_df.to_csv(shap_path, index=False)
    log.info("Saved SHAP summary → %s", shap_path)

    all_metrics = {**clf_metrics, **reg_metrics}
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Saved train metrics → %s", metrics_path)
    log.info("  Metrics: %s", all_metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Supply Chain Risk Monitor — train.py")
    log.info("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = load_feature_matrix(FEATURE_MATRIX_PATH)

    # ── 2. Derive labels ─────────────────────────────────────────────────────
    df = derive_labels(df)

    # ── 3. Preprocess ────────────────────────────────────────────────────────
    df = preprocess(df)

    # ── 4. Temporal split (must happen BEFORE dropping snapshot_date) ────────
    train_df, test_df, cutoff_date = temporal_split(df)

    # ── 5. Build feature arrays ──────────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    X_train = build_feature_array(train_df, feature_cols)
    X_test = build_feature_array(test_df, feature_cols)
    y_train_clf = train_df["disruption_label"]
    y_test_clf = test_df["disruption_label"]

    # Regressor trains only on disrupted rows
    train_pos = train_df[train_df["disruption_label"] == 1]
    test_pos = test_df[test_df["disruption_label"] == 1]
    X_train_reg = build_feature_array(train_pos, feature_cols)
    X_test_reg = build_feature_array(test_pos, feature_cols)
    y_train_reg = train_pos["severity_label"]
    y_test_reg = test_pos["severity_label"]

    # ── 6. MLflow ────────────────────────────────────────────────────────────
    mlflow_db = os.path.join(PROJECT_ROOT, "mlruns", "mlflow.db")
    os.makedirs(os.path.join(PROJECT_ROOT, "mlruns"), exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # mlflow.xgboost.autolog() handles param + metric logging automatically
    mlflow.xgboost.autolog(disable=True)  # we log models manually below

    with mlflow.start_run(run_name="xgb_classifier_regressor"):

        mlflow.log_param("train_ratio", TRAIN_RATIO)
        mlflow.log_param("cutoff_date", str(cutoff_date.date()))
        mlflow.log_param("decision_threshold", DECISION_THRESHOLD)
        mlflow.log_param("days_since_clip", DAYS_SINCE_CLIP)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("pos_train_rows", int(y_train_clf.sum()))
        mlflow.log_param("pos_train_pct",
                         round(100 * y_train_clf.mean(), 2))

        # ── 7. Train classifier ───────────────────────────────────────────────
        clf = train_classifier(X_train, y_train_clf, X_test, y_test_clf)
        clf_metrics = evaluate_classifier(clf, X_test, y_test_clf)

        # ── 8. Train regressor ────────────────────────────────────────────────
        reg_metrics = {"mae": None, "rmse": None}
        if len(X_train_reg) >= 5:
            reg = train_regressor(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
            if len(X_test_reg) >= 2:
                reg_metrics = evaluate_regressor(reg, X_test_reg, y_test_reg)
            else:
                log.warning("Too few positive test rows (%d) to evaluate regressor",
                            len(X_test_reg))
        else:
            log.warning("Too few positive training rows (%d) — skipping regressor",
                        len(X_train_reg))
            # Create a trivial regressor so predict.py can always load one
            reg = xgb.XGBRegressor(n_estimators=10, random_state=42)
            reg.fit(X_train_reg if len(X_train_reg) > 0
                    else X_train.head(10),
                    y_train_reg if len(y_train_reg) > 0
                    else pd.Series([3] * 10))

        # ── 9. SHAP ───────────────────────────────────────────────────────────
        shap_df = compute_shap_importance(clf, X_train)

        # Log SHAP importances to MLflow
        for _, row in shap_df.iterrows():
            mlflow.log_metric(f"shap_{row['feature']}", row["mean_abs_shap"])

        # Log classifier metrics
        for k, v in clf_metrics.items():
            mlflow.log_metric(f"clf_{k}", v)
        for k, v in reg_metrics.items():
            if v is not None:
                mlflow.log_metric(f"reg_{k}", v)

        # Log XGBoost models with MLflow
        mlflow.xgboost.log_model(clf, artifact_path="classifier")
        mlflow.xgboost.log_model(reg, artifact_path="regressor")

        # ── 10. Feature stats for drift monitor ───────────────────────────────
        stats_path = os.path.join(MODELS_DIR, "train_feature_stats.json")
        save_feature_stats(X_train, stats_path)

        # ── 11. Save all output files ─────────────────────────────────────────
        save_outputs(clf, reg, feature_cols, shap_df, clf_metrics, reg_metrics)

    log.info("=" * 60)
    log.info("Training complete.")
    log.info("  Classifier AUC: %.4f", clf_metrics["auc"])
    if reg_metrics["mae"] is not None:
        log.info("  Regressor MAE:  %.4f  RMSE: %.4f",
                 reg_metrics["mae"], reg_metrics["rmse"])
    log.info("  Outputs in: %s", MODELS_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()