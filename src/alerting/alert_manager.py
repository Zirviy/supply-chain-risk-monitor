"""
src/alerting/alert_manager.py
Supply Chain Risk Monitor — Alert Manager

Reads risk_scores.csv hourly. Detects NEW crossings of disruption_prob > 0.70
(was not above 0.70 in previous check). Sends AWS SNS email for each new spike.

Spike detection logic:
  - Stores last-seen disruption_prob per node_id in alert_state.json
  - New spike = current_prob > 0.70 AND (last_prob <= 0.70 OR node not seen before)
  - alert_state.json updated after every check regardless of spike

Run from project root:
  python src/alerting/alert_manager.py

Inputs:
  data/processed/risk_scores.csv
  data/processed/alert_state.json   ← created automatically if not exists

Outputs:
  data/processed/alert_state.json   ← updated after every check
  logs/alert_manager.log            ← appended each run
  AWS SNS publish per new spike (if SNS_TOPIC_ARN set in .env)
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import boto3
import pandas as pd
import schedule
from dotenv import load_dotenv
from botocore.exceptions import BotoCoreError, ClientError

# ---------------------------------------------------------------------------
# Logging — dual output: console + file
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOGS_DIR, "alert_manager.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PROCESSED  = os.path.join(PROJECT_ROOT, "data", "processed")
RISK_SCORES_PATH  = os.path.join(DATA_PROCESSED, "risk_scores.csv")
ALERT_STATE_PATH  = os.path.join(DATA_PROCESSED, "alert_state.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Spike threshold — must match insight_generator.py CRITICAL_THRESHOLD
# Source: architecture spec
SPIKE_THRESHOLD = 0.70

# How often to check (minutes) — set to 60 for production
CHECK_INTERVAL_MINUTES = 60


# ---------------------------------------------------------------------------
# Step 1 — Load and save alert state
# ---------------------------------------------------------------------------

def load_alert_state() -> dict:
    """
    Loads last-seen disruption_prob per node_id from alert_state.json.
    Returns empty dict if file doesn't exist yet (first run).
    Format: { "node_id": last_disruption_prob, ... }
    """
    if not os.path.exists(ALERT_STATE_PATH):
        log.info("alert_state.json not found — creating fresh state (first run).")
        return {}

    try:
        with open(ALERT_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        log.info("Loaded alert state: %d nodes tracked.", len(state))
        return state
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not load alert_state.json (%s) — starting fresh.", e)
        return {}


def save_alert_state(state: dict) -> None:
    """Saves current disruption_prob per node to alert_state.json."""
    try:
        with open(ALERT_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log.info("Alert state saved: %d nodes tracked.", len(state))
    except OSError as e:
        log.error("Failed to save alert_state.json: %s", e)


# ---------------------------------------------------------------------------
# Step 2 — Detect new spikes
# ---------------------------------------------------------------------------

def detect_new_spikes(
    scores: pd.DataFrame,
    last_state: dict,
) -> list[dict]:
    """
    Compares current risk_scores against last known state.

    A 'new spike' occurs when:
      current disruption_prob > SPIKE_THRESHOLD
      AND last known prob for that node was <= SPIKE_THRESHOLD
          (or node was never seen before — first time above threshold)

    Returns list of dicts, one per new spike, with all alert fields.
    """
    new_spikes = []

    for _, row in scores.iterrows():
        node_id      = str(row["node_id"])
        current_prob = float(row["disruption_prob"])
        last_prob    = float(last_state.get(node_id, 0.0))

        is_currently_critical = current_prob > SPIKE_THRESHOLD
        was_not_critical      = last_prob <= SPIKE_THRESHOLD

        if is_currently_critical and was_not_critical:
            spike = {
                "node_id":        node_id,
                "country":        str(row.get("country",        "N/A")),
                "sector":         str(row.get("sector",         "N/A")),
                "commodity":      str(row.get("commodity",      "N/A")),
                "disruption_prob": current_prob,
                "severity_pred":  float(row.get("severity_pred", 0.0)),
                "risk_tier":      str(row.get("risk_tier",      "Critical")),
                "snapshot_date":  str(row.get("snapshot_date",  "")),
                "insight_text":   row.get("insight_text", None),
                "last_prob":      last_prob,
            }
            new_spikes.append(spike)
            log.info(
                "NEW SPIKE detected: %s | country=%s | sector=%s | "
                "prob=%.4f (was %.4f)",
                node_id, spike["country"], spike["sector"],
                current_prob, last_prob,
            )

    return new_spikes


# ---------------------------------------------------------------------------
# Step 3 — Build SNS email body
# ---------------------------------------------------------------------------

def build_email_subject(spike: dict) -> str:
    return (
        f"Supply Chain Alert: {spike['node_id']} — "
        f"{spike['risk_tier']} Risk | {spike['country']} {spike['sector']}"
    )


def build_email_body(spike: dict) -> str:
    """
    Constructs plain text email body for SNS.
    Includes all node fields + AI insight if available.
    """
    insight_section = (
        spike["insight_text"]
        if spike["insight_text"] and str(spike["insight_text"]).strip().lower() not in ("none", "nan", "")
        else "No AI insight generated for this node."
    )

    body = f"""
SUPPLY CHAIN RISK ALERT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*60}

NODE DETAILS
  Node ID:               {spike['node_id']}
  Country:               {spike['country']}
  Sector:                {spike['sector']}
  Commodity:             {spike['commodity']}
  Risk Tier:             {spike['risk_tier']}
  Snapshot Date:         {spike['snapshot_date']}

RISK SCORES
  Disruption Probability: {spike['disruption_prob']:.4f}
  Previous Probability:   {spike['last_prob']:.4f}
  Severity Prediction:    {spike['severity_pred']:.1f} / 5.0
  Alert Threshold:        {SPIKE_THRESHOLD:.2f}

AI INSIGHT
{insight_section}

{'='*60}
This alert was generated automatically by the Supply Chain Risk Monitor.
Node crossed the Critical threshold ({SPIKE_THRESHOLD}) for the first time.
""".strip()

    return body


# ---------------------------------------------------------------------------
# Step 4 — Send SNS notification
# ---------------------------------------------------------------------------

def send_sns_alert(
    spike: dict,
    sns_topic_arn: str,
    aws_region: str,
) -> bool:
    """
    Publishes an alert to AWS SNS.
    Returns True on success, False on any failure.
    Does NOT raise — caller continues regardless.

    SNS email subscription on the topic will deliver to subscribers.
    Set up: AWS Console → SNS → Topics → your topic → Create subscription
            Protocol: Email, Endpoint: your email address.
    """
    subject = build_email_subject(spike)
    body    = build_email_body(spike)

    # SNS subject max length = 100 chars
    if len(subject) > 100:
        subject = subject[:97] + "..."

    try:
        sns_client = boto3.client(
            "sns",
            region_name=aws_region,
        )

        response = sns_client.publish(
            TopicArn=sns_topic_arn,
            Subject=subject,
            Message=body,
        )

        message_id = response.get("MessageId", "unknown")
        log.info(
            "SNS alert sent for %s | MessageId: %s",
            spike["node_id"], message_id,
        )
        return True

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg  = e.response["Error"]["Message"]
        log.error(
            "SNS ClientError for %s: [%s] %s",
            spike["node_id"], error_code, error_msg,
        )
        return False

    except BotoCoreError as e:
        log.error("SNS BotoCoreError for %s: %s", spike["node_id"], str(e))
        return False

    except Exception as e:
        log.error("SNS unexpected error for %s: %s", spike["node_id"], str(e))
        return False


# ---------------------------------------------------------------------------
# Step 5 — Core check function (called hourly by scheduler)
# ---------------------------------------------------------------------------

def check_and_alert() -> None:
    """
    Main check loop:
      1. Load current risk scores
      2. Load last alert state
      3. Detect new spikes
      4. Send SNS for each spike (if configured)
      5. Update alert state
    """
    log.info("=" * 60)
    log.info("Running alert check at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    # Load environment each run (allows hot reload of .env changes)
    load_dotenv()
    sns_topic_arn = os.getenv("SNS_TOPIC_ARN", "").strip()
    aws_region    = os.getenv("AWS_REGION", "ap-south-1").strip()

    sns_enabled = bool(sns_topic_arn)
    if not sns_enabled:
        log.warning(
            "SNS_TOPIC_ARN not set in .env — alerts will be logged only, not sent."
        )

    # Load risk scores
    if not os.path.exists(RISK_SCORES_PATH):
        log.error(
            "risk_scores.csv not found at %s. Run predict.py first.",
            RISK_SCORES_PATH,
        )
        return

    try:
        scores = pd.read_csv(RISK_SCORES_PATH)
    except Exception as e:
        log.error("Failed to read risk_scores.csv: %s", e)
        return

    log.info("Loaded risk_scores.csv: %d nodes", len(scores))

    # Load last alert state
    last_state = load_alert_state()

    # Detect new spikes
    new_spikes = detect_new_spikes(scores, last_state)
    log.info("New spikes detected: %d", len(new_spikes))

    # Send alerts
    if new_spikes:
        for spike in new_spikes:
            log.info(
                "Alerting: %s | %s | %s | prob=%.4f",
                spike["node_id"], spike["country"],
                spike["sector"], spike["disruption_prob"],
            )

            if sns_enabled:
                success = send_sns_alert(spike, sns_topic_arn, aws_region)
                if not success:
                    log.warning(
                        "SNS send failed for %s — spike still recorded in state.",
                        spike["node_id"],
                    )
            else:
                # Log full alert body to console when SNS not configured
                log.info(
                    "ALERT (SNS disabled):\n%s",
                    build_email_body(spike),
                )
    else:
        log.info("No new spikes. No alerts sent.")

    # Update alert state with current probabilities for ALL nodes
    # (not just spikes — we need to track all nodes for future comparisons)
    new_state = {
        str(row["node_id"]): float(row["disruption_prob"])
        for _, row in scores.iterrows()
    }
    save_alert_state(new_state)

    # Summary
    above_threshold = (scores["disruption_prob"] > SPIKE_THRESHOLD).sum()
    log.info(
        "Check complete. Nodes above threshold: %d / %d",
        above_threshold, len(scores),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("Supply Chain Risk Monitor — alert_manager.py")
    log.info("=" * 60)
    log.info("Alert threshold:      disruption_prob > %.2f", SPIKE_THRESHOLD)
    log.info("Check interval:       every %d minutes", CHECK_INTERVAL_MINUTES)
    log.info("Alert state file:     %s", ALERT_STATE_PATH)
    log.info("Log file:             %s", LOG_PATH)
    log.info("=" * 60)

    # Run immediately on start
    check_and_alert()

    # Schedule hourly checks
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(check_and_alert)

    log.info(
        "Scheduler running. Next check in %d minutes. Press Ctrl+C to stop.",
        CHECK_INTERVAL_MINUTES,
    )

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()