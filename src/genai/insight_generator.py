"""
src/genai/insight_generator.py
Supply Chain Risk Monitor — GenAI Insight Generator

For every node where disruption_prob > 0.70 (Critical tier) in
risk_scores.csv, generates a plain English 3-part insight paragraph
using the Anthropic Claude API.

Appends "insight_text" column to risk_scores.csv in place.

Run from project root:
  python src/genai/insight_generator.py

Inputs:
  data/processed/risk_scores.csv               ← filter to risk_tier == "Critical"
  data/processed/processed_events.csv          ← top 5 events per node
  data/processed/supply_chain_nodes_enriched.csv ← node context

Output:
  data/processed/risk_scores.csv               ← insight_text column appended
  Console log of each generated insight (truncated to 200 chars)

API: Anthropic Python SDK (NOT LangChain — minimal dependencies)
MODEL = "llama-3.3-70b-versatile"
Key: loaded from .env as ANTHROPIC_API_KEY
"""

import os
import time
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from groq import Groq

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

RISK_SCORES_PATH    = os.path.join(DATA_PROCESSED, "risk_scores.csv")
EVENTS_PATH         = os.path.join(DATA_PROCESSED, "processed_events.csv")
NODES_ENRICHED_PATH = os.path.join(DATA_PROCESSED, "supply_chain_nodes_enriched.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Only generate insights for Critical nodes
# Source: architecture spec — alert threshold = 0.70
CRITICAL_THRESHOLD = 0.70

# Number of most-severe events to inject per node
TOP_N_EVENTS = 5

# Event lookback window (days) relative to snapshot_date
EVENT_LOOKBACK_DAYS = 14

# API parameters
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 600
TEMP_FIRST  = 0.3   # first attempt
TEMP_RETRY  = 0.0   # retry attempt

# Rate limit buffer — 1 second between calls
SLEEP_BETWEEN_CALLS = 1


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------

def load_data():
    """
    Loads risk_scores, events, and node metadata.
    Deduplicates supply_chain_nodes_enriched on node_id (keep highest
    static_risk_score row — 130 known duplicates per node_mapper.py).
    """
    log.info("Loading data files...")

    # Risk scores
    if not os.path.exists(RISK_SCORES_PATH):
        raise FileNotFoundError(f"risk_scores.csv not found at {RISK_SCORES_PATH}. Run predict.py first.")
    scores = pd.read_csv(RISK_SCORES_PATH)
    log.info("  risk_scores.csv: %d rows", len(scores))

    # Events
    if not os.path.exists(EVENTS_PATH):
        raise FileNotFoundError(f"processed_events.csv not found at {EVENTS_PATH}. Run event_processor.py first.")
    events = pd.read_csv(EVENTS_PATH, low_memory=False)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    log.info("  processed_events.csv: %d rows", len(events))

    # Node metadata — dedup by keeping highest static_risk_score per node_id
    if not os.path.exists(NODES_ENRICHED_PATH):
        raise FileNotFoundError(f"supply_chain_nodes_enriched.csv not found at {NODES_ENRICHED_PATH}. Run node_mapper.py first.")
    nodes_raw = pd.read_csv(NODES_ENRICHED_PATH)
    nodes = (
        nodes_raw
        .sort_values("static_risk_score", ascending=False)
        .drop_duplicates(subset="node_id", keep="first")
        .reset_index(drop=True)
    )
    log.info("  supply_chain_nodes_enriched.csv: %d rows (%d after dedup)",
             len(nodes_raw), len(nodes))

    return scores, events, nodes


# ---------------------------------------------------------------------------
# Step 2 — Filter to Critical nodes
# ---------------------------------------------------------------------------

def get_critical_nodes(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows from risk_scores.csv where disruption_prob > CRITICAL_THRESHOLD.
    Handles both 'Critical' risk_tier label and raw probability check
    (belt and suspenders — in case tier labels shift).
    """
    critical = scores[scores["disruption_prob"] > CRITICAL_THRESHOLD].copy()
    log.info("Critical nodes (disruption_prob > %.2f): %d", CRITICAL_THRESHOLD, len(critical))
    return critical


# ---------------------------------------------------------------------------
# Step 3 — Pull top events for a node
# ---------------------------------------------------------------------------

def get_top_events_for_node(
    node_id: str,
    snapshot_date_str: str,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filters processed_events.csv to:
      - matched_node_id == node_id
      - event_date within last EVENT_LOOKBACK_DAYS of snapshot_date
    Sorts by preliminary_severity descending, returns top TOP_N_EVENTS rows.
    Returns empty DataFrame (not error) if no events match.
    """
    try:
        snapshot_date = pd.to_datetime(snapshot_date_str)
    except Exception:
        return pd.DataFrame()

    cutoff = snapshot_date - timedelta(days=EVENT_LOOKBACK_DAYS)

    node_events = events[
        (events["matched_node_id"].astype(str) == str(node_id)) &
        (events["event_date"] >= cutoff) &
        (events["event_date"] <= snapshot_date)
    ].copy()

    if node_events.empty:
        return pd.DataFrame()

    # Sort by severity and take top N
    if "preliminary_severity" in node_events.columns:
        node_events = node_events.sort_values("preliminary_severity", ascending=False)

    return node_events.head(TOP_N_EVENTS)


# ---------------------------------------------------------------------------
# Step 4 — Build prompt
# ---------------------------------------------------------------------------

def build_prompt(
    score_row: pd.Series,
    node_meta: pd.Series,
    top_events: pd.DataFrame,
) -> str:
    """
    Constructs the structured user prompt for the Anthropic API call.

    Includes:
      - Node details: country, sector, commodity, disruption_prob,
        severity_pred, static_risk_score
      - Top events (if any): event_date, event_category,
        preliminary_severity, GoldsteinScale, NumMentions, SOURCEURL
      - Explicit 3-part output instruction
    """

    # ── Node context block ────────────────────────────────────────────────
    node_section = f"""NODE DETAILS:
  Node ID:           {score_row.get('node_id', 'N/A')}
  Country:           {score_row.get('country', 'N/A')}
  Sector:            {score_row.get('sector', 'N/A')}
  Commodity:         {score_row.get('commodity', 'N/A')}
  Disruption Prob:   {score_row.get('disruption_prob', 'N/A'):.4f}
  Severity Pred:     {score_row.get('severity_pred', 'N/A'):.1f} / 5.0
  Static Risk Score: {score_row.get('static_risk_score', 'N/A'):.4f}
  Snapshot Date:     {score_row.get('snapshot_date', 'N/A')}"""

    # Add representative companies if available from node metadata
    if node_meta is not None and not node_meta.empty:
        rep_companies = node_meta.get("representative_companies", "")
        if pd.notna(rep_companies) and str(rep_companies).strip():
            node_section += f"\n  Key Companies:     {rep_companies}"

    # ── Events block ─────────────────────────────────────────────────────
    if top_events.empty:
        events_section = "RECENT EVENTS (last 14 days): None found for this node."
    else:
        event_lines = []
        for i, (_, ev) in enumerate(top_events.iterrows(), 1):
            date_str   = str(ev.get("event_date", ""))[:10]
            category   = ev.get("event_category", "Unknown")
            severity   = ev.get("preliminary_severity", "N/A")
            goldstein  = ev.get("GoldsteinScale", "N/A")
            mentions   = ev.get("NumMentions", "N/A")
            source_url = ev.get("SOURCEURL", "")

            # Truncate source URL to avoid prompt bloat
            if isinstance(source_url, str) and len(source_url) > 80:
                source_url = source_url[:80] + "..."

            line = (
                f"  Event {i}: [{date_str}] {category} | "
                f"Severity={severity} | Goldstein={goldstein} | "
                f"Mentions={mentions} | Source: {source_url}"
            )
            event_lines.append(line)

        events_section = "RECENT EVENTS (last 14 days, sorted by severity):\n" + "\n".join(event_lines)

    # ── Full prompt ───────────────────────────────────────────────────────
    prompt = f"""{node_section}

{events_section}

Based on the above data, provide a supply chain risk assessment in EXACTLY this format:

PART 1 — RISK EXPLANATION (2-3 sentences):
Explain in plain English why this supply chain node is at risk right now. Cite specific events listed above if available. Be concrete about geography and sector.

PART 2 — DOWNSTREAM IMPACT (2-3 bullet points):
List the 2-3 downstream industries or companies most likely to be affected and explain briefly why each is exposed.

PART 3 — CONFIDENCE RATING:
State High, Medium, or Low confidence in this risk assessment, followed by one sentence explaining why.

Do not include any text outside of these three parts."""

    return prompt


# ---------------------------------------------------------------------------
# Step 5 — Call Anthropic API with graceful degradation
# ---------------------------------------------------------------------------

def call_anthropic(
    client,
    prompt: str,
    node_id: str,
) -> str | None:
    """
    Calls the Anthropic API with first attempt at TEMP_FIRST.
    On any error or empty response, retries once at TEMP_RETRY.
    On second failure, logs error and returns None.
    Does NOT raise — always returns str or None.
    """

    system_prompt = (
        "You are a supply chain risk analyst. Be concise and precise. "
        "Always respond in exactly the format requested."
    )

    for attempt, temperature in enumerate([TEMP_FIRST, TEMP_RETRY], start=1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            # Extract text content
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            text = text.strip()

            # Basic validation — response should contain all 3 parts
            if not text:
                raise ValueError("Empty response from API")

            has_part1 = "PART 1" in text or "Risk Explanation" in text.lower()
            has_part2 = "PART 2" in text or "Downstream" in text.lower()
            has_part3 = "PART 3" in text or "Confidence" in text.lower()

            if not (has_part1 and has_part2 and has_part3):
                raise ValueError(
                    f"Response missing required parts. "
                    f"Part1={has_part1}, Part2={has_part2}, Part3={has_part3}"
                )

            log.info(
                "  [OK] Node %s (attempt %d): %s...",
                node_id, attempt, text[:200]
            )
            return text

        except Exception as e:
            log.warning(
                "  [API ERROR] Node %s attempt %d: status=%s msg=%s",
                node_id, attempt, e.status_code, str(e)[:100]
            )
        except Exception as e:
            log.warning(
                "  [CONNECTION ERROR] Node %s attempt %d: %s",
                node_id, attempt, str(e)[:100]
            )
        except Exception as e:
            log.warning(
                "  [RATE LIMIT] Node %s attempt %d — sleeping 30s: %s",
                node_id, attempt, str(e)[:100]
            )
            time.sleep(30)
        except Exception as e:
            log.warning(
                "  [ERROR] Node %s attempt %d: %s",
                node_id, attempt, str(e)[:150]
            )

        if attempt < 2:
            log.info("  Retrying node %s at temperature=0.0...", node_id)
            time.sleep(2)

    log.error("  [FAILED] Node %s: both attempts failed. insight_text=None.", node_id)
    return None


# ---------------------------------------------------------------------------
# Step 6 — Main pipeline
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Supply Chain Risk Monitor — insight_generator.py")
    log.info("=" * 60)

    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found in environment. "
            "Add it to your .env file: GROQ_API_KEY=gsk_..."
    )
    client = Groq(api_key=api_key)
    log.info("API key loaded from .env")

    # Initialise Anthropic client
    client = Groq(api_key=api_key)

    # Load all data
    scores, events, nodes = load_data()

    # Filter to Critical nodes only
    critical = get_critical_nodes(scores)

    if critical.empty:
        log.info("=" * 60)
        log.info("No Critical nodes found (disruption_prob > %.2f).", CRITICAL_THRESHOLD)
        log.info("Nothing to do. risk_scores.csv unchanged.")
        log.info("=" * 60)

        # Still ensure insight_text column exists (for dashboard compatibility)
        if "insight_text" not in scores.columns:
            scores["insight_text"] = None
            scores.to_csv(RISK_SCORES_PATH, index=False)
            log.info("Added empty insight_text column to risk_scores.csv.")
        return

    log.info("Generating insights for %d Critical node(s)...", len(critical))
    log.info("Model: %s | Max tokens: %d", MODEL, MAX_TOKENS)
    log.info("-" * 60)

    # Prepare insight_text column in scores DataFrame
    if "insight_text" not in scores.columns:
        scores["insight_text"] = None

    # Build node metadata lookup dict for fast access
    node_meta_lookup = nodes.set_index("node_id")

    # Process each Critical node
    success_count = 0
    fail_count = 0

    for idx, row in critical.iterrows():
        node_id      = str(row["node_id"])
        snapshot_str = str(row.get("snapshot_date", ""))

        log.info("Processing node %s (prob=%.4f)...", node_id, row["disruption_prob"])

        # Get node metadata
        node_meta = node_meta_lookup.loc[node_id] if node_id in node_meta_lookup.index else pd.Series()

        # Get top events for this node
        top_events = get_top_events_for_node(node_id, snapshot_str, events)
        log.info("  Events found in last %d days: %d", EVENT_LOOKBACK_DAYS, len(top_events))

        # Build prompt
        prompt = build_prompt(row, node_meta, top_events)

        # Call API
        insight_text = call_anthropic(client, prompt, node_id)

        # Write result back to scores DataFrame using the original index
        scores.loc[scores["node_id"].astype(str) == node_id, "insight_text"] = insight_text

        if insight_text is not None:
            success_count += 1
        else:
            fail_count += 1

        # Rate limiting — 1 second between calls
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Save updated risk_scores.csv
    scores.to_csv(RISK_SCORES_PATH, index=False)

    log.info("=" * 60)
    log.info("DONE")
    log.info("=" * 60)
    log.info("Insights generated: %d success, %d failed", success_count, fail_count)
    log.info("risk_scores.csv updated → %s", RISK_SCORES_PATH)

    # Print summary of generated insights
    if success_count > 0:
        log.info("\nGenerated insight previews:")
        generated = scores[scores["insight_text"].notna()]
        for _, row in generated.iterrows():
            preview = str(row["insight_text"])[:200]
            log.info(
                "  [%s] %s | %s | prob=%.4f\n    %s...",
                row["node_id"], row.get("country", "?"),
                row.get("sector", "?"), row["disruption_prob"],
                preview
            )


if __name__ == "__main__":
    main()