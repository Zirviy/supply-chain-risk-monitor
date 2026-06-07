"""
src/ml/predict.py
Supply Chain Risk Monitor — Inference Pipeline

Loads trained XGBoost models from models/, scores all nodes using the most
recent snapshot from feature_matrix.csv, writes risk_scores.csv.

Run from project root:
  python src/ml/predict.py

Inputs:
  models/classifier.json
  models/regressor.json
  models/feature_columns.json
  data/processed/feature_matrix.csv
  data/processed/supply_chain_nodes_enriched.csv

Output:
  data/processed/risk_scores.csv
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

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
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

FEATURE_MATRIX_PATH = os.path.join(DATA_PROCESSED, "feature_matrix.csv")
NODES_ENRICHED_PATH = os.path.join(DATA_PROCESSED, "supply_chain_nodes_enriched.csv")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.json")
REGRESSOR_PATH = os.path.join(MODELS_DIR, "regressor.json")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
OUTPUT_PATH = os.path.join(DATA_PROCESSED, "risk_scores.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Clip sentinel value — must match train.py exactly
DAYS_SINCE_CLIP = 30

# Risk tier thresholds — from architecture spec
# Critical/>0.70, High/0.50–0.70, Medium/0.30–0.50, Low/<0.30
TIER_CRITICAL = 0.70
TIER_HIGH = 0.50
TIER_MEDIUM = 0.30

# Severity clipping — regressor output clipped to valid 1–5 range
SEVERITY_MIN = 1.0
SEVERITY_MAX = 5.0


# ---------------------------------------------------------------------------
# Step 1 — Load models and feature columns
# ---------------------------------------------------------------------------

def load_models():
    log.info("Loading models from %s", MODELS_DIR)

    clf = xgb.XGBClassifier()
    clf.load_model(CLASSIFIER_PATH)
    log.info("  Loaded classifier: %s", CLASSIFIER_PATH)

    reg = xgb.XGBRegressor()
    reg.load_model(REGRESSOR_PATH)
    log.info("  Loaded regressor:  %s", REGRESSOR_PATH)

    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    log.info("  Feature columns: %d features", len(feature_cols))

    return clf, reg, feature_cols


# ---------------------------------------------------------------------------
# Step 2 — Load and prepare feature matrix (most recent snapshot only)
# ---------------------------------------------------------------------------

def load_latest_snapshot(feature_cols: list) -> pd.DataFrame:
    log.info("Loading feature matrix from %s", FEATURE_MATRIX_PATH)
    df = pd.read_csv(FEATURE_MATRIX_PATH, parse_dates=["snapshot_date"])
    log.info("  Full matrix: %d rows × %d cols", *df.shape)

    # Filter to most recent snapshot date
    latest_date = df["snapshot_date"].max()
    df = df[df["snapshot_date"] == latest_date].copy()
    log.info("  Snapshot date used: %s", latest_date.date())
    log.info("  Rows for scoring: %d nodes", len(df))

    # Preprocessing — must match train.py exactly
    df["days_since_high_severity"] = df["days_since_high_severity"].clip(upper=DAYS_SINCE_CLIP)

    # Validate all required feature columns are present
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature matrix missing columns: {missing}")

    return df, latest_date


# ---------------------------------------------------------------------------
# Step 3 — Load node metadata
# ---------------------------------------------------------------------------

def load_node_metadata() -> pd.DataFrame:
    """
    Load supply_chain_nodes_enriched.csv and deduplicate.
    130 duplicate node_ids known — keep row with highest static_risk_score.
    Documented in node_mapper.py outputs.
    """
    log.info("Loading node metadata from %s", NODES_ENRICHED_PATH)
    nodes = pd.read_csv(NODES_ENRICHED_PATH)
    log.info("  Raw rows: %d", len(nodes))

    nodes = (
        nodes
        .sort_values("static_risk_score", ascending=False)
        .drop_duplicates(subset="node_id", keep="first")
        .reset_index(drop=True)
    )
    log.info("  After dedup: %d unique nodes", len(nodes))

    return nodes[["node_id", "country", "sector", "commodity", "lat", "lon"]]


# ---------------------------------------------------------------------------
# Step 4 — Run inference
# ---------------------------------------------------------------------------

def assign_risk_tier(prob: float) -> str:
    """
    Assign risk tier from disruption probability.
    Thresholds from architecture spec:
      Critical  > 0.70
      High      0.50–0.70
      Medium    0.30–0.50
      Low       < 0.30
    """
    if prob > TIER_CRITICAL:
        return "Critical"
    elif prob > TIER_HIGH:
        return "High"
    elif prob > TIER_MEDIUM:
        return "Medium"
    else:
        return "Low"


def run_inference(
    df: pd.DataFrame,
    clf: xgb.XGBClassifier,
    reg: xgb.XGBRegressor,
    feature_cols: list,
    latest_date,
    nodes: pd.DataFrame,
) -> pd.DataFrame:

    X = df[feature_cols].copy()

    # Classifier — disruption probability
    log.info("Running classifier inference on %d nodes...", len(X))
    disruption_prob = clf.predict_proba(X)[:, 1]

    # Regressor — severity prediction (indicative; clipped to 1–5)
    log.info("Running regressor inference...")
    severity_pred = reg.predict(X)
    severity_pred = np.clip(severity_pred, SEVERITY_MIN, SEVERITY_MAX)

    # Assemble results
    results = pd.DataFrame({
        "node_id": df["node_id"].values,
        "snapshot_date": latest_date.date(),
        "disruption_prob": np.round(disruption_prob, 4),
        "severity_pred": np.round(severity_pred, 1),
        "risk_tier": [assign_risk_tier(p) for p in disruption_prob],
        "static_risk_score": df["static_risk_score"].values,
    })

    # Merge node metadata
    results = results.merge(nodes, on="node_id", how="left")

    # Final column order per spec
    results = results[[
        "node_id", "snapshot_date", "disruption_prob", "severity_pred",
        "risk_tier", "country", "sector", "commodity",
        "static_risk_score", "lat", "lon",
    ]]

    # Sort by disruption_prob descending
    results = results.sort_values("disruption_prob", ascending=False).reset_index(drop=True)

    return results


# ---------------------------------------------------------------------------
# Step 5 — Log summary
# ---------------------------------------------------------------------------

def log_summary(results: pd.DataFrame) -> None:
    tier_counts = results["risk_tier"].value_counts()

    log.info("=" * 60)
    log.info("Risk tier summary:")
    for tier in ["Critical", "High", "Medium", "Low"]:
        count = tier_counts.get(tier, 0)
        log.info("  %-10s %d nodes", tier, count)

    log.info("Top 10 nodes by disruption probability:")
    top10 = results.head(10)
    for _, row in top10.iterrows():
        log.info("  %-20s  prob=%.4f  tier=%-8s  country=%s  sector=%s",
                 row["node_id"], row["disruption_prob"],
                 row["risk_tier"], row["country"], row["sector"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Supply Chain Risk Monitor — predict.py")
    log.info("=" * 60)

    # Load models
    clf, reg, feature_cols = load_models()

    # Load data
    df, latest_date = load_latest_snapshot(feature_cols)
    nodes = load_node_metadata()

    # Inference
    results = run_inference(df, clf, reg, feature_cols, latest_date, nodes)

    # Log summary
    log_summary(results)

    # Save
    results.to_csv(OUTPUT_PATH, index=False)
    log.info("=" * 60)
    log.info("Saved risk_scores.csv → %s", OUTPUT_PATH)
    log.info("%d nodes scored.", len(results))
    log.info("=" * 60)


if __name__ == "__main__":
    main()