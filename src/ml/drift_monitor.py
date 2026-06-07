"""
src/ml/drift_monitor.py
Supply Chain Risk Monitor — Data Drift Monitor

Computes Population Stability Index (PSI) comparing the last 7 days of
feature distributions against the training distribution saved by train.py.

PSI formula: Σ (actual% - expected%) × ln(actual% / expected%)
PSI interpretation:
  PSI < 0.10  — no significant drift
  PSI 0.10–0.20 — moderate drift, monitor
  PSI > 0.20  — significant drift, consider retraining

Source: Yurdakul (2018), "Statistical Properties of Population Stability
Index", University of Michigan Working Paper.
https://scholarworks.wmich.edu/dissertations/3208

Run from project root:
  python src/ml/drift_monitor.py

Inputs:
  data/processed/feature_matrix.csv
  models/train_feature_stats.json   ← written by train.py

Output:
  logs/drift.log  ← appended each run (PSI per feature + warnings)
  Prints summary to console.
"""

import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging — dual output: console + drift.log
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOGS_DIR, "drift.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

FEATURE_MATRIX_PATH = os.path.join(DATA_PROCESSED, "feature_matrix.csv")
TRAIN_STATS_PATH = os.path.join(MODELS_DIR, "train_feature_stats.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rolling window for "current" distribution
RECENT_DAYS = 7

# PSI threshold for drift warning
# Source: Yurdakul (2018) — PSI > 0.20 indicates significant population shift
PSI_WARN_THRESHOLD = 0.20
PSI_MONITOR_THRESHOLD = 0.10

# Clip sentinel — must match train.py
DAYS_SINCE_CLIP = 30

# Dynamic features: vary across snapshots for the same node.
# PSI is appropriate here — we're asking whether the event/signal
# distribution has shifted since the model was trained.
DYNAMIC_FEATURES = [
    "event_count_7d",
    "event_count_14d",
    "event_count_30d",
    "goldstein_mean_14d",
    "goldstein_slope_14d",
    "mention_accel",
    "severity_mean_14d",
    "severity_max_14d",
    "severity_weighted_mean_14d",
    "days_since_high_severity",
]

# Static-per-node features: set once by node_mapper.py / feature_engineer.py,
# identical across all 372 snapshots for a given node.
#
# PSI is NOT appropriate for these features. PSI measures temporal distribution
# shift, but static features cannot shift between pipeline runs by definition —
# they are node properties (FSI country risk, trade HHI, commodity importance)
# that only change when the entire reference data pipeline is rebuilt.
# Applying PSI to them compares 952 node values against decile bin edges
# that were computed from those same 952 values, which produces numerically
# unstable results when the distribution is discrete or heavily zero-inflated.
#
# Instead: a simple mean-deviation check. If the recent node-level mean
# deviates from the training mean by more than STATIC_MEAN_WARN_PCT,
# it indicates the reference data was rebuilt with different source data.
STATIC_PER_NODE_FEATURES = [
    "upstream_exposure",
    "static_risk_score",
    "vulnerability_score",
    "country_risk_fsi",
    "hhi",
    "commodity_importance",
]

# Flag a static feature if its mean has shifted by more than this fraction.
# 5% threshold: a meaningful change in reference data (e.g. new FSI release,
# updated trade vulnerability scores). Tighter than PSI's 0.20 threshold
# because static features should be perfectly stable between pipeline runs.
STATIC_MEAN_WARN_PCT = 0.05


# ---------------------------------------------------------------------------
# PSI computation (dynamic features only)
# ---------------------------------------------------------------------------

def compute_psi(
    actual: np.ndarray,
    expected_bins: list,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute PSI for a dynamic feature.

    Buckets observations using the training decile edges as bin boundaries.
    Expected distribution: 10% per decile bucket (uniform by construction).
    Actual distribution: recent data bucketed by the same edges.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    Source: Yurdakul (2018)
    """
    raw_edges = np.array(expected_bins, dtype=float)  # 11 values

    if len(raw_edges) < 2:
        return 0.0

    # Use the 9 interior decile points as cut boundaries (10th–90th percentile).
    # Each of the 10 resulting buckets contains exactly 10% of training data
    # by construction — no reconstruction needed.
    # We use np.digitize which handles edge values correctly.
    interior = raw_edges[1:-1]  # 9 cut points

    if len(interior) == 0:
        return 0.0

    # Remove duplicate cut points — when the distribution is heavily
    # zero-inflated, multiple interior points may be identical.
    # Keep unique cuts only; remaining buckets absorb their mass.
    unique_cuts = np.unique(interior)

    if len(unique_cuts) == 0:
        return 0.0

    n_buckets = len(unique_cuts) + 1

    # Expected: count how many of the 10 original decile buckets map into
    # each surviving bucket after deduplication.
    # Original buckets: [raw[0],raw[1]), [raw[1],raw[2]), ..., [raw[9],raw[10]]
    # Each carries mass 1/10.
    expected_counts = np.zeros(n_buckets)
    for i in range(len(raw_edges) - 1):
        # Representative value for original bucket i: its midpoint
        lo, hi = raw_edges[i], raw_edges[i + 1]
        rep = (lo + hi) / 2.0 if lo != hi else lo
        # Which surviving bucket does this representative value fall into?
        bucket_idx = int(np.searchsorted(unique_cuts, rep, side="right"))
        bucket_idx = min(bucket_idx, n_buckets - 1)
        expected_counts[bucket_idx] += 1

    expected_pct = expected_counts / expected_counts.sum()

    # Actual: bucket the recent data using the same unique cut points
    actual_bucket_ids = np.searchsorted(unique_cuts, actual, side="right")
    actual_bucket_ids = np.clip(actual_bucket_ids, 0, n_buckets - 1)
    actual_counts = np.bincount(actual_bucket_ids, minlength=n_buckets).astype(float)
    actual_pct = actual_counts / (actual_counts.sum() + epsilon)

    # Epsilon guard against log(0)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# Step 1 — Load inputs
# ---------------------------------------------------------------------------

def load_inputs():
    log.info("Loading feature matrix from %s", FEATURE_MATRIX_PATH)
    df = pd.read_csv(FEATURE_MATRIX_PATH, parse_dates=["snapshot_date"])
    log.info("  Shape: %d rows × %d cols", *df.shape)

    log.info("Loading training feature stats from %s", TRAIN_STATS_PATH)
    with open(TRAIN_STATS_PATH) as f:
        train_stats = json.load(f)
    log.info("  Stats loaded for %d features", len(train_stats))

    return df, train_stats


# ---------------------------------------------------------------------------
# Step 2 — Extract recent window
# ---------------------------------------------------------------------------

def get_recent_data(df: pd.DataFrame) -> pd.DataFrame:
    latest_date = df["snapshot_date"].max()
    cutoff = latest_date - timedelta(days=RECENT_DAYS - 1)
    recent = df[df["snapshot_date"] >= cutoff].copy()

    log.info("Recent window: %s → %s (%d rows, %d unique dates)",
             cutoff.date(), latest_date.date(),
             len(recent), recent["snapshot_date"].nunique())

    # Clip sentinel — must match train.py
    recent["days_since_high_severity"] = recent["days_since_high_severity"].clip(
        upper=DAYS_SINCE_CLIP
    )

    return recent


# ---------------------------------------------------------------------------
# Step 3a — PSI check for dynamic features
# ---------------------------------------------------------------------------

def run_psi_check(recent: pd.DataFrame, train_stats: dict) -> pd.DataFrame:
    results = []

    for feature in DYNAMIC_FEATURES:
        if feature not in train_stats:
            log.warning("  Feature '%s' not in train_stats — skipping", feature)
            continue
        if feature not in recent.columns:
            log.warning("  Feature '%s' not in feature matrix — skipping", feature)
            continue

        actual = recent[feature].dropna().values
        expected_bins = train_stats[feature]["decile_bins"]
        psi = compute_psi(actual, expected_bins)

        if psi > PSI_WARN_THRESHOLD:
            status = "DRIFT"
        elif psi > PSI_MONITOR_THRESHOLD:
            status = "MONITOR"
        else:
            status = "OK"

        results.append({
            "feature": feature,
            "check_type": "PSI",
            "value": round(psi, 4),
            "status": status,
            "train_ref": round(train_stats[feature]["mean"], 6),
            "recent_val": round(float(actual.mean()), 6) if len(actual) > 0 else None,
            "detail": f"PSI={psi:.4f}",
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Step 3b — Mean-deviation check for static features
# ---------------------------------------------------------------------------

def run_static_check(recent: pd.DataFrame, train_stats: dict) -> pd.DataFrame:
    """
    For static-per-node features, compare the node-level mean in the recent
    window against the training mean. A deviation > STATIC_MEAN_WARN_PCT
    indicates the reference data pipeline was rebuilt with different inputs
    (e.g. new FSI release, updated ISB trade data).

    Uses node-level deduplication (one row per node) to avoid inflating
    counts from the 372-snapshot repetition.
    """
    recent_node = recent.drop_duplicates(subset="node_id", keep="first")
    results = []

    for feature in STATIC_PER_NODE_FEATURES:
        if feature not in train_stats:
            log.warning("  Static feature '%s' not in train_stats — skipping", feature)
            continue
        if feature not in recent_node.columns:
            log.warning("  Static feature '%s' not in feature matrix — skipping", feature)
            continue

        actual = recent_node[feature].dropna().values
        train_mean = train_stats[feature]["mean"]
        recent_mean = float(actual.mean()) if len(actual) > 0 else 0.0

        if train_mean == 0:
            deviation = 0.0
        else:
            deviation = abs(recent_mean - train_mean) / abs(train_mean)

        if deviation > STATIC_MEAN_WARN_PCT:
            status = "DRIFT"
            detail = f"mean shifted {deviation*100:.1f}% from training"
        else:
            status = "OK"
            detail = f"mean stable (deviation={deviation*100:.2f}%)"

        results.append({
            "feature": feature,
            "check_type": "MEAN_DEV",
            "value": round(deviation, 4),
            "status": status,
            "train_ref": round(train_mean, 6),
            "recent_val": round(recent_mean, 6),
            "detail": detail,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Step 4 — Log results
# ---------------------------------------------------------------------------

def log_results(psi_df: pd.DataFrame, static_df: pd.DataFrame) -> None:
    all_results = pd.concat([psi_df, static_df], ignore_index=True)
    drifted = all_results[all_results["status"] == "DRIFT"]
    monitored = all_results[all_results["status"] == "MONITOR"]

    log.info("=" * 60)
    log.info("Drift Report — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    # --- Dynamic features (PSI) ---
    log.info("Dynamic features — PSI check (%d features):", len(psi_df))
    log.info("  %-35s  %6s  %8s  %12s  %12s",
             "Feature", "PSI", "Status", "Train Mean", "Recent Mean")
    log.info("  " + "-" * 80)
    psi_sorted = psi_df.sort_values("value", ascending=False)
    for _, row in psi_sorted.iterrows():
        log.info("  %-35s  %6.4f  %8s  %12.6f  %12.6f",
                 row["feature"], row["value"], row["status"],
                 row["train_ref"], row["recent_val"] or 0.0)

    log.info("")

    # --- Static features (mean deviation) ---
    log.info("Static features — mean deviation check (%d features):", len(static_df))
    log.info("  %-35s  %8s  %8s  %12s  %12s  %s",
             "Feature", "Dev%", "Status", "Train Mean", "Recent Mean", "Detail")
    log.info("  " + "-" * 90)
    for _, row in static_df.iterrows():
        log.info("  %-35s  %7.2f%%  %8s  %12.6f  %12.6f  %s",
                 row["feature"], row["value"] * 100, row["status"],
                 row["train_ref"], row["recent_val"] or 0.0, row["detail"])

    log.info("")

    # --- Summary ---
    psi_drifted = psi_df[psi_df["status"] == "DRIFT"]
    psi_monitored = psi_df[psi_df["status"] == "MONITOR"]
    static_drifted = static_df[static_df["status"] == "DRIFT"]

    log.info("Summary:")
    log.info("  PSI DRIFT   (> 0.20): %d dynamic features", len(psi_drifted))
    log.info("  PSI MONITOR (> 0.10): %d dynamic features", len(psi_monitored))
    log.info("  Static mean drift (> 5%%): %d features", len(static_drifted))

    if len(drifted) > 0:
        log.warning("")
        log.warning("⚠ DRIFT DETECTED:")
        for _, row in drifted.iterrows():
            log.warning("  [%s] %s — %s", row["check_type"], row["feature"], row["detail"])
        log.warning("Recommendation: inspect recent data; consider retraining.")
    else:
        log.info("No significant drift detected. Models are stable.")

    log.info("Drift log appended to: %s", LOG_PATH)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Supply Chain Risk Monitor — drift_monitor.py")
    log.info("=" * 60)

    df, train_stats = load_inputs()
    recent = get_recent_data(df)
    psi_df = run_psi_check(recent, train_stats)
    static_df = run_static_check(recent, train_stats)
    log_results(psi_df, static_df)


if __name__ == "__main__":
    main()