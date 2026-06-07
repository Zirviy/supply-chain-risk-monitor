"""
src/dashboard/app.py
Supply Chain Risk Monitor — Streamlit Dashboard

3 pages:
  Page 1 — Global Risk Map    : Folium map, nodes coloured by risk tier
  Page 2 — Risk Trend Explorer: Plotly line chart per node over time
  Page 3 — Event Feed         : Live event table with filters

Run from project root:
  streamlit run src/dashboard/app.py
"""

import os
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

RISK_SCORES_PATH    = os.path.join(DATA_PROCESSED, "risk_scores.csv")
EVENTS_PATH         = os.path.join(DATA_PROCESSED, "processed_events.csv")
NODES_ENRICHED_PATH = os.path.join(DATA_PROCESSED, "supply_chain_nodes_enriched.csv")
FEATURE_MATRIX_PATH = os.path.join(DATA_PROCESSED, "feature_matrix.csv")

# ---------------------------------------------------------------------------
# Colour map — risk tier → hex
# ---------------------------------------------------------------------------
TIER_COLORS = {
    "Critical": "#d32f2f",
    "High":     "#f57c00",
    "Medium":   "#fbc02d",
    "Low":      "#388e3c",
}

TIER_ORDER = ["Critical", "High", "Medium", "Low"]

# ---------------------------------------------------------------------------
# Country → (lat, lon) lookup
# node_mapper.py leaves lat/lon as NaN for all trade nodes.
# This table covers every country that appears in the ISB trade data.
# Coordinates = geographic centroid of the country.
# ---------------------------------------------------------------------------
COUNTRY_COORDS = {
    "Afghanistan": (33.93, 67.71), "Albania": (41.15, 20.17),
    "Algeria": (28.03, 1.66), "Angola": (11.20, 17.87),
    "Argentina": (-38.42, -63.62), "Armenia": (40.07, 45.04),
    "Australia": (-25.27, 133.78), "Austria": (47.52, 14.55),
    "Azerbaijan": (40.14, 47.58), "Bahrain": (26.00, 50.55),
    "Bangladesh": (23.68, 90.36), "Belarus": (53.71, 27.95),
    "Belgium": (50.50, 4.47), "Benin": (9.31, 2.32),
    "Bolivia": (-16.29, -63.59), "Bosnia": (43.92, 17.68),
    "Brazil": (-14.24, -51.93), "Bulgaria": (42.73, 25.49),
    "Cambodia": (12.57, 104.99), "Cameroon": (3.85, 11.50),
    "Canada": (56.13, -106.35), "Chile": (-35.68, -71.54),
    "China": (35.86, 104.20), "Colombia": (4.57, -74.30),
    "Congo": (-0.23, 15.83), "Costa Rica": (9.75, -83.75),
    "Croatia": (45.10, 15.20), "Cuba": (21.52, -77.78),
    "Czech Republic": (49.82, 15.47), "Denmark": (56.26, 9.50),
    "Ecuador": (-1.83, -78.18), "Egypt": (26.82, 30.80),
    "Ethiopia": (9.15, 40.49), "Finland": (61.92, 25.75),
    "France": (46.23, 2.21), "Gabon": (-0.80, 11.61),
    "Germany": (51.17, 10.45), "Ghana": (7.95, -1.02),
    "Greece": (39.07, 21.82), "Guatemala": (15.78, -90.23),
    "Hong Kong": (22.32, 114.17), "Hungary": (47.16, 19.50),
    "India": (20.59, 78.96), "Indonesia": (-0.79, 113.92),
    "Iran": (32.43, 53.69), "Iraq": (33.22, 43.68),
    "Ireland": (53.41, -8.24), "Israel": (31.05, 34.85),
    "Italy": (41.87, 12.57), "Ivory Coast": (7.54, -5.55),
    "Jamaica": (18.11, -77.30), "Japan": (36.20, 138.25),
    "Jordan": (30.59, 36.24), "Kazakhstan": (48.02, 66.92),
    "Kenya": (-0.02, 37.91), "Kuwait": (29.31, 47.48),
    "Kyrgyzstan": (41.20, 74.77), "Laos": (19.86, 102.50),
    "Lebanon": (33.85, 35.86), "Libya": (26.34, 17.23),
    "Malaysia": (4.21, 101.98), "Mexico": (23.63, -102.55),
    "Morocco": (31.79, -7.09), "Mozambique": (-18.67, 35.53),
    "Myanmar": (21.91, 95.96), "Netherlands": (52.13, 5.29),
    "New Zealand": (-40.90, 174.89), "Nigeria": (9.08, 8.68),
    "North Korea": (40.34, 127.51), "Norway": (60.47, 8.47),
    "Oman": (21.47, 55.98), "Pakistan": (30.38, 69.35),
    "Panama": (8.54, -80.78), "Peru": (-9.19, -75.02),
    "Philippines": (12.88, 121.77), "Poland": (51.92, 19.15),
    "Portugal": (39.40, -8.22), "Qatar": (25.35, 51.18),
    "Romania": (45.94, 24.97), "Russia": (61.52, 105.32),
    "Saudi Arabia": (23.89, 45.08), "Senegal": (14.50, -14.45),
    "Serbia": (44.02, 21.01), "Singapore": (1.35, 103.82),
    "Slovakia": (48.67, 19.70), "South Africa": (-30.56, 22.94),
    "South Korea": (35.91, 127.77), "Spain": (40.46, -3.75),
    "Sri Lanka": (7.87, 80.77), "Sudan": (12.86, 30.22),
    "Sweden": (60.13, 18.64), "Switzerland": (46.82, 8.23),
    "Syria": (34.80, 38.99), "Taiwan": (23.70, 121.00),
    "Tanzania": (-6.37, 34.89), "Thailand": (15.87, 100.99),
    "Tunisia": (33.89, 9.54), "Turkey": (38.96, 35.24),
    "Turkmenistan": (38.97, 59.56), "UAE": (23.42, 53.85),
    "Uganda": (1.37, 32.29), "Ukraine": (48.38, 31.17),
    "United Kingdom": (55.38, -3.44), "United States": (37.09, -95.71),
    "Uruguay": (-32.52, -55.77), "Uzbekistan": (41.38, 64.59),
    "Venezuela": (6.42, -66.59), "Vietnam": (14.06, 108.28),
    "Yemen": (15.55, 48.52), "Zambia": (-13.13, 27.85),
    "Zimbabwe": (-19.01, 29.15),
    # Common short aliases
    "USA": (37.09, -95.71), "UK": (55.38, -3.44),
    "S Korea": (35.91, 127.77),
    # UN official long-form names used in ISB data
    "Viet Nam": (14.06, 108.28),
    "Türkiye": (38.96, 35.24),
    "Turkiye": (38.96, 35.24),
    "Russian Federation": (61.52, 105.32),
    "Republic of Korea": (35.91, 127.77),
    "Democratic People's Republic of Korea": (40.34, 127.51),
    "Islamic Republic of Iran": (32.43, 53.69),
    "Syrian Arab Republic": (34.80, 38.99),
    "Lao People's Democratic Republic": (19.86, 102.50),
    "Taiwan, Province of China": (23.70, 121.00),
    "United Kingdom of Great Britain and Northern Ireland": (55.38, -3.44),
    "Kingdom of the Netherlands": (52.13, 5.29),
    "Bosnia and Herzegovina": (43.92, 17.68),
    "Republic of Moldova": (47.41, 28.37),
    "North Macedonia": (41.61, 21.75),
    "Czechia": (49.82, 15.47),
    "Brunei Darussalam": (4.54, 114.73),
    "Timor-Leste": (-8.87, 125.73),
    "Papua New Guinea": (-6.31, 143.96),
    "Solomon Islands": (-9.43, 160.03),
    "Vanuatu": (-15.38, 166.96),
    "Fiji": (-17.71, 178.07),
    "Samoa": (-13.76, -172.10),
    "Tonga": (-21.18, -175.20),
    "Kiribati": (-3.37, -168.73),
    "Tuvalu": (-7.11, 177.65),
    "Nauru": (-0.53, 166.93),
    "Palau": (7.52, 134.58),
    "Marshall Islands": (7.13, 171.18),
    "Federated States of Micronesia": (7.43, 150.55),
    "Macao": (22.17, 113.55),
    "Maldives": (3.20, 73.22),
    "Bhutan": (27.51, 90.43),
    "Nepal": (28.39, 84.12),
    "Georgia": (42.32, 43.36),
    "Armenia": (40.07, 45.04),
    "Tajikistan": (38.86, 71.28),
    "Kyrgyzstan": (41.20, 74.77),
    "Uzbekistan": (41.38, 64.59),
    "Turkmenistan": (38.97, 59.56),
    "Kazakhstan": (48.02, 66.92),
    "Azerbaijan": (40.14, 47.58),
    "Estonia": (58.60, 25.01),
    "Latvia": (56.88, 24.60),
    "Lithuania": (55.17, 23.88),
    "Luxembourg": (49.82, 6.13),
    "Iceland": (64.96, -19.02),
    "Liechtenstein": (47.17, 9.56),
    "Monaco": (43.74, 7.41),
    "Andorra": (42.55, 1.60),
    "San Marino": (43.94, 12.46),
    "Holy See(Vatican City)": (41.90, 12.45),
    "Gibraltar": (36.14, -5.35),
    "Cyprus": (35.13, 33.43),
    "Malta": (35.94, 14.37),
    "Montenegro": (42.71, 19.37),
    "Albania": (41.15, 20.17),
    "North Korea": (40.34, 127.51),
    "Serbia": (44.02, 21.01),
    "Serbia and Montenegro": (43.52, 20.46),
    "Netherlands Antilles": (12.23, -69.06),
    "French Polynesia": (-17.68, -149.41),
    "New Caledonia": (-20.90, 165.62),
    "Guam": (13.44, 144.79),
    "American Samoa": (-14.27, -170.13),
    "Faroe Islands": (61.89, -6.91),
    "Guernsey": (49.47, -2.59),
    "Jersey": (49.21, -2.13),
    "Tokelau": (-9.17, -171.84),
    "Norfolk Island": (-29.03, 167.95),
    "Pitcairn": (-25.07, -130.10),
    "Christmas Island": (-10.49, 105.62),
    "Cocos (Keeling) Islands": (-12.16, 96.87),
    "Wallis and Futuna": (-13.77, -177.16),
    "Cook Islands": (-21.24, -159.78),
    "Niue": (-19.05, -169.86),
    "State of Palestine": (31.95, 35.23),
    "Heard Island and McDonald Islands": (-53.08, 73.50),
    "Mongolia": (46.86, 103.85),
    "Malaysia": (4.21, 101.98),
}


def _inject_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills lat/lon from COUNTRY_COORDS lookup wherever they are NaN.
    Uses the 'country' column to look up coordinates.
    """
    mask = df["lat"].isna() | df["lon"].isna()
    if not mask.any():
        return df
    coords = df.loc[mask, "country"].map(
        lambda c: COUNTRY_COORDS.get(str(c), (float("nan"), float("nan")))
    )
    df.loc[mask, "lat"] = coords.map(lambda x: x[0])
    df.loc[mask, "lon"] = coords.map(lambda x: x[1])
    return df


# ---------------------------------------------------------------------------
# Data loaders — all cached
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_risk_scores() -> pd.DataFrame:
    """
    Loads risk_scores.csv.
    Injects lat/lon from COUNTRY_COORDS for trade nodes (node_mapper
    leaves these as NaN — coordinates were never populated in pipeline).
    Fills missing insight_text with empty string.
    Does NOT drop rows — NaN coords handled per-use-site.
    """
    scores = pd.read_csv(RISK_SCORES_PATH)

    # Ensure insight_text column exists and nulls are empty strings
    if "insight_text" not in scores.columns:
        scores["insight_text"] = ""
    scores["insight_text"] = scores["insight_text"].fillna("").astype(str)
    scores["insight_text"] = scores["insight_text"].replace("nan", "")

    # Inject coordinates from country lookup (overrides NaN lat/lon from pipeline)
    scores = _inject_coords(scores)

    return scores


@st.cache_data(ttl=300)
def load_nodes_enriched() -> pd.DataFrame:
    """
    Loads supply_chain_nodes_enriched.csv.
    Deduplicates by keeping highest static_risk_score per node_id.
    130 known duplicate node_ids — always dedup.
    Injects lat/lon from country lookup.
    """
    nodes = pd.read_csv(NODES_ENRICHED_PATH)
    nodes = (
        nodes
        .sort_values("static_risk_score", ascending=False)
        .drop_duplicates(subset="node_id", keep="first")
        .reset_index(drop=True)
    )
    nodes = _inject_coords(nodes)
    return nodes


@st.cache_data(ttl=300)
def load_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_PATH, low_memory=False)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    return events


@st.cache_data(ttl=600)
def load_feature_matrix_for_node(node_id: str) -> pd.DataFrame:
    """
    Loads only rows for a specific node_id from feature_matrix.csv.
    Reads full file but filters immediately — avoids holding 354k rows in memory.
    Uses chunksize for memory efficiency on the large file.
    """
    chunks = []
    for chunk in pd.read_csv(
        FEATURE_MATRIX_PATH,
        chunksize=10000,
        parse_dates=["snapshot_date"],
        low_memory=False,
    ):
        filtered = chunk[chunk["node_id"] == node_id]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values("snapshot_date")
    return df


# ---------------------------------------------------------------------------
# Page 1 — Global Risk Map
# ---------------------------------------------------------------------------

def page_risk_map() -> None:
    st.header("Global Supply Chain Risk Map")

    scores = load_risk_scores()

    # Summary metrics row
    tier_counts = scores["risk_tier"].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔴 Critical", tier_counts.get("Critical", 0))
    col2.metric("🟠 High",     tier_counts.get("High",     0))
    col3.metric("🟡 Medium",   tier_counts.get("Medium",   0))
    col4.metric("🟢 Low",      tier_counts.get("Low",      0))

    snap_date = scores["snapshot_date"].iloc[0] if not scores.empty else "N/A"
    mappable  = scores["lat"].notna() & scores["lon"].notna()
    st.caption(
        f"Snapshot date: {snap_date}  |  "
        f"Total nodes: {len(scores)}  |  Mappable: {mappable.sum()}"
    )

    # Tier filter
    selected_tiers = st.multiselect(
        "Filter by risk tier",
        options=TIER_ORDER,
        default=TIER_ORDER,
    )
    # Only pass rows with valid coordinates to Folium — avoids NaN crash
    filtered = scores[scores["risk_tier"].isin(selected_tiers) & mappable]

    # Build Folium map
    m = folium.Map(
        location=[20, 78],   # India-centered, global view
        zoom_start=3,
        tiles="CartoDB positron",
    )

    for _, row in filtered.iterrows():
        tier   = row.get("risk_tier", "Medium")
        color  = TIER_COLORS.get(tier, "#fbc02d")
        prob   = float(row.get("disruption_prob", 0))
        radius = 6 + prob * 10   # bigger circle = higher risk

        # Build popup HTML
        insight = str(row.get("insight_text", "")).strip()
        insight_html = (
            f"<br><b>AI Insight:</b><br>"
            f"<div style='max-width:300px;font-size:11px;'>{insight[:500]}</div>"
            if insight
            else ""
        )

        popup_html = f"""
        <div style='font-family:sans-serif;font-size:12px;min-width:220px;'>
          <b style='font-size:14px;'>{row.get('node_id','')}</b><br>
          <b>Country:</b>  {row.get('country','N/A')}<br>
          <b>Sector:</b>   {row.get('sector','N/A')}<br>
          <b>Commodity:</b>{row.get('commodity','N/A')}<br>
          <hr style='margin:4px 0;'>
          <b>Risk Tier:</b>
          <span style='color:{color};font-weight:bold;'>{tier}</span><br>
          <b>Disruption Prob:</b> {prob:.4f}<br>
          <b>Severity Pred:</b>   {row.get('severity_pred','N/A')} / 5.0<br>
          <b>Static Risk:</b>     {float(row.get('static_risk_score',0)):.4f}
          {insight_html}
        </div>
        """

        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=340),
            tooltip=f"{row.get('node_id','')} | {tier} | {prob:.3f}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                border:1px solid #ccc;font-family:sans-serif;font-size:12px;'>
      <b>Risk Tier</b><br>
      <span style='color:#d32f2f;'>●</span> Critical (&gt;0.70)<br>
      <span style='color:#f57c00;'>●</span> High (0.50–0.70)<br>
      <span style='color:#fbc02d;'>●</span> Medium (0.30–0.50)<br>
      <span style='color:#388e3c;'>●</span> Low (&lt;0.30)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=1100, height=580, returned_objects=[])

    # Top 10 table below map
    st.subheader("Top 10 Highest Risk Nodes")
    top10 = (
        scores
        .sort_values("disruption_prob", ascending=False)
        .head(10)[[
            "node_id", "country", "sector", "commodity",
            "disruption_prob", "severity_pred", "risk_tier"
        ]]
        .reset_index(drop=True)
    )
    top10.index += 1
    st.dataframe(top10, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Risk Trend Explorer
# ---------------------------------------------------------------------------

def page_trend_explorer() -> None:
    st.header("Risk Trend Explorer")

    scores = load_risk_scores()

    # Node selector — label as "Country — Sector — Node ID"
    scores["_label"] = (
        scores["country"].fillna("?") + " — " +
        scores["sector"].fillna("?")  + " — " +
        scores["node_id"].astype(str)
    )
    label_to_node = dict(zip(scores["_label"], scores["node_id"]))

    # Default to highest-risk node
    sorted_scores = scores.sort_values("disruption_prob", ascending=False)
    all_labels    = sorted(label_to_node.keys())
    default_label = sorted_scores["_label"].iloc[0] if not sorted_scores.empty else all_labels[0]

    selected_label = st.selectbox(
        "Select a supply chain node",
        options=all_labels,
        index=all_labels.index(default_label) if default_label in all_labels else 0,
    )
    node_id = label_to_node[selected_label]

    # Current score card
    node_row = scores[scores["node_id"] == node_id].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Disruption Prob",  f"{node_row['disruption_prob']:.4f}")
    col2.metric("Risk Tier",         node_row["risk_tier"])
    col3.metric("Severity Pred",    f"{node_row['severity_pred']:.1f} / 5.0")
    col4.metric("Static Risk Score", f"{float(node_row.get('static_risk_score', 0)):.4f}")

    # Load historical feature matrix for this node
    with st.spinner(f"Loading history for {node_id}..."):
        fm = load_feature_matrix_for_node(node_id)

    if fm.empty:
        st.warning(f"No historical data found in feature_matrix.csv for node {node_id}.")
    else:
        # Build a trend proxy: static_risk_score + normalized event activity
        # disruption_prob only exists for the latest snapshot (predict.py output)
        # so we approximate historical risk signal from feature components
        fm = fm.sort_values("snapshot_date")

        # Trend proxy = 0.5×static_risk + 0.3×severity_mean_14d + 0.2×mention_accel
        # All components already 0-1 normalized in feature_engineer.py
        fm["risk_signal"] = (
            0.50 * fm["static_risk_score"].fillna(0) +
            0.30 * fm["severity_mean_14d"].fillna(0) +
            0.20 * fm["mention_accel"].fillna(0)
        ).clip(0, 1)

        fig = go.Figure()

        # Risk signal line
        fig.add_trace(go.Scatter(
            x=fm["snapshot_date"],
            y=fm["risk_signal"],
            mode="lines",
            name="Risk Signal (proxy)",
            line=dict(color="#f57c00", width=2),
        ))

        # Event count bars (secondary)
        fig.add_trace(go.Bar(
            x=fm["snapshot_date"],
            y=fm["event_count_14d"],
            name="Events (14d)",
            marker_color="rgba(100,149,237,0.4)",
            yaxis="y2",
        ))

        # Current prob as horizontal reference line
        fig.add_hline(
            y=float(node_row["disruption_prob"]),
            line_dash="dash",
            line_color="#d32f2f",
            annotation_text=f"Current prob: {node_row['disruption_prob']:.4f}",
            annotation_position="top left",
        )

        fig.update_layout(
            title=f"Risk Trend — {node_id} ({node_row['country']} | {node_row['sector']})",
            xaxis_title="Date",
            yaxis=dict(title="Risk Signal (0–1)", range=[0, 1]),
            yaxis2=dict(
                title="Event Count (14d)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=420,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

        st.plotly_chart(fig, use_container_width=True)

        # Feature snapshot table
        with st.expander("Feature snapshot (latest values)"):
            latest = fm.iloc[-1].drop(["node_id", "snapshot_date"], errors="ignore")
            latest_df = pd.DataFrame({
                "Feature": latest.index,
                "Value":   latest.values,
            })
            st.dataframe(latest_df, use_container_width=True)

    # Recent events for this node
    st.subheader(f"Recent Events — {node_id}")
    events = load_events()
    node_events = events[
        events["matched_node_id"].astype(str) == str(node_id)
    ].copy()

    if node_events.empty:
        st.info("No events matched to this node in processed_events.csv.")
    else:
        node_events = node_events.sort_values("event_date", ascending=False)
        display_cols = [
            c for c in [
                "event_date", "event_category", "preliminary_severity",
                "GoldsteinScale", "NumMentions", "SOURCEURL"
            ] if c in node_events.columns
        ]
        st.dataframe(node_events[display_cols].head(20), use_container_width=True)

    # AI insight
    insight = str(node_row.get("insight_text", "")).strip()
    if insight and insight.lower() not in ("", "nan", "none"):
        st.subheader("AI Insight")
        st.info(insight)


# ---------------------------------------------------------------------------
# Page 3 — Event Feed
# ---------------------------------------------------------------------------

def page_event_feed() -> None:
    st.header("Live Event Feed")

    events = load_events()

    # Summary
    st.caption(
        f"Total events: {len(events)}  |  "
        f"Date range: {events['event_date'].min().date()} → "
        f"{events['event_date'].max().date()}"
    )

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        country_options = ["All"] + sorted(
            events["country"].dropna().unique().tolist()
        )
        selected_country = st.selectbox("Filter by country", country_options)

    with col2:
        category_options = ["All"] + sorted(
            events["event_category"].dropna().unique().tolist()
        )
        selected_category = st.selectbox("Filter by event category", category_options)

    # Apply filters
    filtered = events.copy()
    if selected_country != "All":
        filtered = filtered[filtered["country"] == selected_country]
    if selected_category != "All":
        filtered = filtered[filtered["event_category"] == selected_category]

    # Sort most recent first
    filtered = filtered.sort_values("event_date", ascending=False)

    st.caption(f"Showing {len(filtered)} events after filters.")

    # Build display DataFrame with clickable URLs
    display_cols = [
        c for c in [
            "event_date", "country", "event_category",
            "preliminary_severity", "matched_node_id",
            "matched_sector", "GoldsteinScale", "NumMentions",
            "SOURCEURL",
        ] if c in filtered.columns
    ]

    display_df = filtered[display_cols].copy().reset_index(drop=True)

    # Format event_date to date string
    if "event_date" in display_df.columns:
        display_df["event_date"] = display_df["event_date"].dt.strftime("%Y-%m-%d")

    # Round severity
    if "preliminary_severity" in display_df.columns:
        display_df["preliminary_severity"] = display_df["preliminary_severity"].round(3)

    # Truncate SOURCEURL for display
    if "SOURCEURL" in display_df.columns:
        display_df["SOURCEURL"] = display_df["SOURCEURL"].astype(str).apply(
            lambda url: url[:80] + "..." if len(url) > 80 else url
        )

    st.dataframe(
        display_df.head(200),
        use_container_width=True,
        height=500,
    )

    # Severity distribution chart
    st.subheader("Severity Distribution")
    if "preliminary_severity" in filtered.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=filtered["preliminary_severity"].dropna(),
            nbinsx=20,
            marker_color="#f57c00",
            opacity=0.8,
            name="Event Severity",
        ))
        fig.update_layout(
            xaxis_title="Preliminary Severity (0–1)",
            yaxis_title="Count",
            height=280,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

    # Category breakdown
    st.subheader("Events by Category")
    cat_counts = (
        filtered["event_category"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Category", "event_category": "Count"})
    )
    # Handle pandas version difference in value_counts column naming
    if "event_category" in cat_counts.columns and "count" in cat_counts.columns:
        cat_counts = cat_counts.rename(columns={"event_category": "Category", "count": "Count"})
    elif "event_category" in cat_counts.columns:
        cat_counts.columns = ["Category", "Count"]

    st.dataframe(cat_counts, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Supply Chain Risk Monitor",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.title("🌐 Supply Chain\nRisk Monitor")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        options=["🗺️ Global Risk Map", "📈 Risk Trend Explorer", "📋 Event Feed"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data sources: GDELT, EMIS, ISB India Data Portal, "
        "Fund for Peace FSI 2023, UN Comtrade"
    )
    st.sidebar.caption("ML: XGBoost | GenAI: Groq llama-3.3-70b")

    # Check required files exist before rendering
    missing = [
        p for p in [RISK_SCORES_PATH, EVENTS_PATH, NODES_ENRICHED_PATH]
        if not os.path.exists(p)
    ]
    if missing:
        st.error(
            "Required data files not found. Run the pipeline first:\n\n"
            "1. python src/processing/build_reference_data.py\n"
            "2. python src/processing/node_mapper.py\n"
            "3. python src/ingestion/gdelt_collector.py --mode historical\n"
            "4. python src/processing/event_processor.py\n"
            "5. python src/processing/feature_engineer.py\n"
            "6. python src/ml/train.py\n"
            "7. python src/ml/predict.py\n\n"
            f"Missing: {missing}"
        )
        return

    # Render selected page
    if page == "🗺️ Global Risk Map":
        page_risk_map()
    elif page == "📈 Risk Trend Explorer":
        page_trend_explorer()
    elif page == "📋 Event Feed":
        page_event_feed()


if __name__ == "__main__":
    main()