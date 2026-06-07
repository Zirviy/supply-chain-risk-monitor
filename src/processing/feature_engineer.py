"""
feature_engineer.py  —  builds the ML feature matrix
Reads:
  data/processed/processed_events.csv          (from event_processor.py)
  data/processed/supply_chain_nodes_enriched.csv (from node_mapper.py)
  data/processed/node_graph_edges.csv           (from node_mapper.py)
Outputs:
  data/processed/feature_matrix.csv

Feature groups per (node_id, snapshot_date):
  1. Rolling event counts     — 7d, 14d, 30d windows
  2. Goldstein trend          — mean + linear slope over 14d
  3. Mention acceleration     — 7d / 30d mention ratio
  4. Severity aggregates      — mean, max, mention-weighted mean over 14d
  5. Network exposure         — upstream risk propagation via graph edges
  6. Seasonal flags           — monsoon, typhoon, sanctions cycle
  7. Recency                  — days since last high-severity event at node
  8. Node static features     — static_risk_score, vulnerability, FSI, HHI,
                                commodity_importance, chokepoint_weight
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))
OUT  = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────
# SEASONAL FLAG LOGIC
# monsoon  — IMD (India Meteorological Department): Jun–Sep
# typhoon  — JMA (Japan Meteorological Agency) Western Pacific: Jul–Nov
# sanctions cycle — IMF/World Bank fiscal windows: Q1 (Jan–Mar) + Q4 (Oct–Dec)
# ─────────────────────────────────────────────
def get_seasonal_flags(month: int) -> dict:
    return {
        "flag_monsoon_season":  int(month in {6, 7, 8, 9}),
        "flag_typhoon_season":  int(month in {7, 8, 9, 10, 11}),
        "flag_sanctions_cycle": int(month in {1, 2, 3, 10, 11, 12}),
    }


# ─────────────────────────────────────────────
# GOLDSTEIN SLOPE  (OLS linear trend over window)
# Negative slope = worsening (more destabilizing over time).
# ─────────────────────────────────────────────
def compute_slope(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    x  = np.arange(len(values), dtype=float)
    y  = values.values.astype(float)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm * xm).sum()
    return float((xm * ym).sum() / denom) if denom != 0 else 0.0


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    events_path = os.path.join(OUT, "processed_events.csv")
    nodes_path  = os.path.join(OUT, "supply_chain_nodes_enriched.csv")
    edges_path  = os.path.join(OUT, "node_graph_edges.csv")

    for p in [events_path, nodes_path, edges_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing required file: {p}\n"
                f"Run preceding pipeline steps first."
            )

    events = pd.read_csv(events_path, low_memory=False)
    nodes  = pd.read_csv(nodes_path,  low_memory=False)
    edges  = pd.read_csv(edges_path,  low_memory=False)

    # ── Diagnose edges columns so we can map them correctly
    print(f"  Edges columns ({len(edges.columns)}): {edges.columns.tolist()}")
    print(f"  Edges sample:\n{edges.head(3).to_string()}")

    # ── Parse event dates
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["event_date"])

    # ── Only matched events contribute to node features
    matched = events[events["matched_node_id"].notna()].copy()
    matched["matched_node_id"] = matched["matched_node_id"].astype(str)

    print(f"\n  Events: {len(events)} total | {len(matched)} matched to nodes")
    print(f"  Nodes:  {len(nodes)} | Edges: {len(edges)}")
    print(f"  Event date range: {events['event_date'].min().date()} → "
          f"{events['event_date'].max().date()}")

    return matched, nodes, edges


# ─────────────────────────────────────────────
# EDGE COLUMN RESOLVER
# node_mapper.py may have written edges with different column names.
# We detect whatever names are present and normalise to source/target/weight.
# ─────────────────────────────────────────────
def normalise_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [source, target, weight].
    Handles common naming variants produced by node_mapper.py:
      source_id / target_id / edge_weight
      from / to / weight
      node_from / node_to / w
      source / target / weight   (already correct)
    """
    cols = {c.lower(): c for c in edges.columns}

    # Candidate names for each role — most specific first
    source_candidates = ["source_node_id", "source_id", "source",
                         "from", "node_from", "src", "origin", "from_node"]
    target_candidates = ["target_node_id", "target_id", "target",
                         "to", "node_to", "dst", "destination", "to_node"]
    weight_candidates = ["weight", "edge_weight", "w", "value",
                         "trade_weight", "flow", "strength"]

    def find_col(candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        return None

    src_col = find_col(source_candidates)
    tgt_col = find_col(target_candidates)
    wgt_col = find_col(weight_candidates)

    if src_col is None or tgt_col is None:
        # Last resort: assume first two columns are source/target
        c = edges.columns.tolist()
        src_col = c[0]
        tgt_col = c[1]
        wgt_col = c[2] if len(c) > 2 else None
        print(f"  [WARN] Could not detect source/target columns by name. "
              f"Assuming col[0]=source ({src_col}), col[1]=target ({tgt_col})")

    out = pd.DataFrame()
    out["source"] = edges[src_col].astype(str)
    out["target"] = edges[tgt_col].astype(str)
    out["weight"] = (
        pd.to_numeric(edges[wgt_col], errors="coerce").fillna(0.0)
        if wgt_col else pd.Series(1.0, index=edges.index)
    )

    print(f"  Edge columns mapped: '{src_col}' → source | "
          f"'{tgt_col}' → target | "
          f"'{wgt_col}' → weight")
    return out


# ─────────────────────────────────────────────
# UPSTREAM EXPOSURE MAP
# For each node, compute: Σ (upstream_node.static_risk_score × edge.weight)
# where "upstream" = nodes that SUPPLY INTO this node (source → target edges).
# Source: supply chain network propagation model (Tang 2006,
#         Int. Journal of Production Economics).
# ─────────────────────────────────────────────
def build_upstream_exposure(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    edges = normalise_edges(edges)

    node_risk = dict(zip(
        nodes["node_id"].astype(str),
        nodes["static_risk_score"].fillna(0).astype(float)
    ))

    exposure = {}
    for target, group in edges.groupby("target"):
        score = sum(
            node_risk.get(str(row["source"]), 0.0) * row["weight"]
            for _, row in group.iterrows()
        )
        exposure[str(target)] = round(score, 6)

    covered = sum(1 for v in exposure.values() if v > 0)
    print(f"  Nodes with non-zero upstream exposure: {covered}")
    return exposure


# ─────────────────────────────────────────────
# FEATURE COMPUTATION — one (node, snapshot_date) cell
# ─────────────────────────────────────────────
def compute_node_features(
    node_id: str,
    snapshot_date: pd.Timestamp,
    node_events: pd.DataFrame,
    node_meta: dict,
    upstream_exposure: float,
) -> dict:
    feat = {
        "node_id":       node_id,
        "snapshot_date": snapshot_date.date(),
    }

    d7  = snapshot_date - timedelta(days=7)
    d14 = snapshot_date - timedelta(days=14)
    d30 = snapshot_date - timedelta(days=30)

    # Events strictly before snapshot_date
    # Guard: empty DataFrame from nodes with no events has no columns
    if node_events.empty or "event_date" not in node_events.columns:
        node_events = pd.DataFrame(columns=["event_date", "GoldsteinScale",
                                            "NumMentions", "preliminary_severity"])
    past = node_events[node_events["event_date"] < snapshot_date]
    w7   = past[past["event_date"] >= d7]
    w14  = past[past["event_date"] >= d14]
    w30  = past[past["event_date"] >= d30]

    # ── 1. Rolling event counts
    feat["event_count_7d"]  = len(w7)
    feat["event_count_14d"] = len(w14)
    feat["event_count_30d"] = len(w30)

    # ── 2. Goldstein mean + OLS slope (14d)
    gs_14 = w14["GoldsteinScale"].dropna() if "GoldsteinScale" in w14.columns else pd.Series(dtype=float)
    feat["goldstein_mean_14d"]  = round(float(gs_14.mean()), 4)  if len(gs_14) > 0 else 0.0
    feat["goldstein_slope_14d"] = round(compute_slope(gs_14), 6) if len(gs_14) >= 2 else 0.0

    # ── 3. Mention acceleration  (7d mentions / 30d mentions)
    #    Captures whether media coverage is intensifying recently vs baseline.
    m7  = float(w7["NumMentions"].fillna(0).sum())  if "NumMentions" in w7.columns  else 0.0
    m30 = float(w30["NumMentions"].fillna(0).sum()) if "NumMentions" in w30.columns else 0.0
    feat["mention_accel"] = round(m7 / max(m30, 1.0), 4)

    # ── 4. Severity aggregates (14d)
    sev_14 = w14["preliminary_severity"].dropna() if "preliminary_severity" in w14.columns else pd.Series(dtype=float)
    feat["severity_mean_14d"] = round(float(sev_14.mean()), 4) if len(sev_14) > 0 else 0.0
    feat["severity_max_14d"]  = round(float(sev_14.max()),  4) if len(sev_14) > 0 else 0.0

    # Mention-weighted severity mean  (high-coverage events count more)
    if (len(w14) > 0
            and "NumMentions" in w14.columns
            and "preliminary_severity" in w14.columns):
        wts  = w14["NumMentions"].fillna(1).clip(1, None).astype(float)
        sevs = w14["preliminary_severity"].fillna(0).astype(float)
        feat["severity_weighted_mean_14d"] = round(
            float((sevs * wts).sum() / wts.sum()), 4
        )
    else:
        feat["severity_weighted_mean_14d"] = 0.0

    # ── 5. Network upstream exposure (static per node, not time-varying)
    feat["upstream_exposure"] = round(upstream_exposure, 6)

    # ── 6. Seasonal flags  (based on snapshot month)
    feat.update(get_seasonal_flags(snapshot_date.month))

    # ── 7. Days since last high-severity event (severity > 0.70)
    if "preliminary_severity" in past.columns and len(past) > 0:
        high_sev = past[past["preliminary_severity"] > 0.70]
        if len(high_sev) > 0:
            feat["days_since_high_severity"] = (
                snapshot_date - high_sev["event_date"].max()
            ).days
        else:
            feat["days_since_high_severity"] = 999
    else:
        feat["days_since_high_severity"] = 999

    # ── 8. Node static features  (from supply_chain_nodes_enriched.csv)
    feat["static_risk_score"]    = float(node_meta.get("static_risk_score",    0) or 0)
    feat["vulnerability_score"]  = float(node_meta.get("vulnerability_score",  0) or 0)
    feat["country_risk_fsi"]     = float(node_meta.get("country_risk_fsi",     0) or 0)
    feat["hhi"]                  = float(node_meta.get("hhi",                  0) or 0)
    feat["commodity_importance"] = float(node_meta.get("commodity_importance", 0) or 0)
    feat["chokepoint_weight"]    = float(node_meta.get("chokepoint_weight",    0) or 0)

    return feat


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def build_feature_matrix(snapshot_dates: list = None) -> pd.DataFrame:
    """
    Compute the full feature matrix.
    snapshot_dates: list of pd.Timestamps (defaults to one per day in event range).
    """
    print("=" * 60)
    print("Feature Engineer")
    print("=" * 60)

    matched_events, nodes, edges = load_data()

    # ── Upstream exposure map (computed once — graph is static)
    print("\nBuilding upstream exposure map...")
    upstream_map = build_upstream_exposure(nodes, edges)

    # ── Node metadata lookup — deduplicate node_id first (keep highest static_risk_score row)
    nodes_dedup = (
        nodes.sort_values("static_risk_score", ascending=False)
             .drop_duplicates(subset=["node_id"], keep="first")
    )
    dupes = len(nodes) - len(nodes_dedup)
    if dupes > 0:
        print(f"  [INFO] Dropped {dupes} duplicate node_id rows from nodes file (kept highest risk score row)")
    node_meta = {
        str(k): v
        for k, v in nodes_dedup.set_index("node_id").to_dict("index").items()
    }

    # ── Snapshot dates
    if snapshot_dates is None:
        min_date = matched_events["event_date"].min().normalize()
        max_date = (
            matched_events["event_date"].max().normalize() + timedelta(days=1)
        )
        snapshot_dates = pd.date_range(min_date, max_date, freq="D").tolist()

    print(f"\nSnapshot range: {snapshot_dates[0].date()} → "
          f"{snapshot_dates[-1].date()} ({len(snapshot_dates)} snapshots)")

    # ── Only build features for trade nodes (chokepoints have no event stream)
    trade_node_ids = (
        nodes[nodes["node_type"] == "trade"]["node_id"]
        .astype(str).unique().tolist()
    )
    print(f"Trade nodes: {len(trade_node_ids)}")
    print(f"Total cells: {len(trade_node_ids) * len(snapshot_dates):,}")

    # ── Group events by node for O(1) lookup
    events_by_node = {
        nid: grp.reset_index(drop=True)
        for nid, grp in matched_events.groupby("matched_node_id")
    }

    # ── Build features
    rows     = []
    total    = len(trade_node_ids) * len(snapshot_dates)
    done     = 0
    log_step = max(1, len(snapshot_dates) * max(1, len(trade_node_ids) // 20))

    for node_id in trade_node_ids:
        nid_str     = str(node_id)
        node_events = events_by_node.get(nid_str, pd.DataFrame())
        meta        = node_meta.get(nid_str, {})
        exposure    = upstream_map.get(nid_str, 0.0)

        for snap in snapshot_dates:
            rows.append(
                compute_node_features(nid_str, snap, node_events, meta, exposure)
            )
            done += 1
            if done % log_step == 0:
                print(f"  Progress: {done:,}/{total:,} ({100*done/total:.0f}%)")

    feature_matrix = pd.DataFrame(rows)

    # ── Quality checks
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    nulls = feature_matrix.isnull().sum()
    bad   = nulls[nulls > 0]
    if len(bad):
        print(f"  Null counts:\n{bad.to_string()}")
    else:
        print("  No nulls — matrix complete")

    return feature_matrix


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    fm = build_feature_matrix()

    if fm.empty:
        print("Feature matrix is empty — check upstream outputs.")
    else:
        out_path = os.path.join(OUT, "feature_matrix.csv")
        fm.to_csv(out_path, index=False)
        print(f"\nDONE → {out_path}")
        print(f"Rows: {len(fm):,} | Cols: {len(fm.columns)}")

        print(f"\nFeature summary (numeric cols):")
        num_cols = fm.select_dtypes(include=[np.number]).columns.tolist()
        print(fm[num_cols].describe().round(4).to_string())

        print(f"\nTop 10 nodes by 14d severity mean (latest snapshot):")
        latest = fm[fm["snapshot_date"] == fm["snapshot_date"].max()].copy()
        top = latest.nlargest(10, "severity_mean_14d")[
            ["node_id", "severity_mean_14d", "severity_max_14d",
             "event_count_14d", "static_risk_score", "upstream_exposure"]
        ]
        print(top.to_string(index=False))

        print(f"\nTop 10 nodes by upstream exposure:")
        top_exp = latest.nlargest(10, "upstream_exposure")[
            ["node_id", "upstream_exposure", "static_risk_score",
             "event_count_14d", "severity_mean_14d"]
        ]
        print(top_exp.to_string(index=False))