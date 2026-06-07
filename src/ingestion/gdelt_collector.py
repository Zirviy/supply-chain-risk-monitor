"""
gdelt_collector.py  —  downloads GDELT 2.0 event files
GDELT updates every 15 minutes at:
  http://data.gdeltproject.org/gdeltv2/lastupdate.txt
  
Each update file is a tab-separated .CSV.zip with 61 columns, no header.
We download, unzip, filter for supply chain CAMEO codes, and save to
data/raw/gdelt/

Source: The GDELT Project — gdeltproject.org
        GDELT 2.0 Event Database documentation:
        gdeltproject.org/data/documentation/GDELT-Event_Codebook-V2.0.pdf
"""

import os
import requests
import zipfile
import io
import time
import schedule
import pandas as pd
from datetime import datetime
from loguru import logger

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))
RAW  = os.path.join(ROOT, "data", "raw", "gdelt")
LOG  = os.path.join(ROOT, "logs")
os.makedirs(RAW, exist_ok=True)
os.makedirs(LOG, exist_ok=True)

logger.add(os.path.join(LOG, "gdelt_collector.log"), rotation="1 day", retention="7 days")

GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# CAMEO prefixes to keep — same set as event_processor.py
CAMEO_KEEP = {"14", "141", "142", "143", "144", "145",
              "17", "171", "172", "173", "174", "175",
              "18", "180", "181", "182", "183",
              "19", "190", "193", "195", "196",
              "20", "201", "202", "0233", "0234", "0251"}

# GDELT column index for EventCode is 26 (0-indexed)
EVENTCODE_COL_IDX = 26


def get_latest_gdelt_url() -> str | None:
    """
    Fetches the GDELT lastupdate.txt and extracts the export (events) file URL.
    lastupdate.txt has 3 lines:
      Line 1: export file (events)
      Line 2: mentions file
      Line 3: gkg file (Global Knowledge Graph)
    We only want line 1.
    """
    try:
        resp = requests.get(GDELT_LASTUPDATE_URL, timeout=15)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        # Each line format: "<hash>  <size>  <url>"
        export_line = lines[0]
        url = export_line.split()[-1]
        logger.info(f"Latest GDELT URL: {url}")
        return url
    except Exception as e:
        logger.error(f"Failed to fetch lastupdate.txt: {e}")
        return None


def already_downloaded(url: str) -> bool:
    """Check if this file was already downloaded (by filename)."""
    filename = url.split("/")[-1].replace(".zip", "")
    return os.path.exists(os.path.join(RAW, filename))


def download_and_filter(url: str) -> int:
    """
    Downloads a GDELT zip, filters for supply chain CAMEO codes,
    saves the filtered CSV to data/raw/gdelt/.
    Returns number of rows saved.
    """
    filename = url.split("/")[-1].replace(".zip", "")
    out_path = os.path.join(RAW, filename)

    try:
        logger.info(f"Downloading: {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(
                    f, sep="\t", header=None, dtype=str,
                    on_bad_lines="skip", low_memory=False,
                )

        # Filter by CAMEO code (column index 26) before saving
        # This keeps file sizes small — typically 5-15% of raw rows
        if len(df.columns) > EVENTCODE_COL_IDX:
            event_codes = df.iloc[:, EVENTCODE_COL_IDX].astype(str)
            mask = event_codes.apply(
                lambda code: any(
                    code == c or code.startswith(c)
                    for c in CAMEO_KEEP
                )
            )
            df_filtered = df[mask]
        else:
            df_filtered = df

        df_filtered.to_csv(out_path, sep="\t", index=False, header=False)
        logger.success(
            f"Saved {len(df_filtered)} rows (from {len(df)} raw) → {filename}"
        )
        return len(df_filtered)

    except Exception as e:
        logger.error(f"Failed to download/process {url}: {e}")
        return 0


def collect_once():
    """Single collection run — called by scheduler or directly."""
    logger.info("=== GDELT collection run ===")
    url = get_latest_gdelt_url()
    if url is None:
        return

    if already_downloaded(url):
        logger.info("Already downloaded, skipping.")
        return

    rows = download_and_filter(url)
    logger.info(f"Collection complete. {rows} supply-chain events saved.")


def collect_historical(days: int = 7):
    """
    Downloads the last N days of GDELT data for historical backfill.
    GDELT historical files follow the pattern:
      http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip
    Files exist every 15 minutes. We pull one per hour to limit volume.
    """
    from datetime import timedelta

    logger.info(f"Starting historical backfill: last {days} days")
    base_url = "http://data.gdeltproject.org/gdeltv2/"
    now = datetime.utcnow()
    total_rows = 0

    for day_offset in range(days):
        target_date = now - timedelta(days=day_offset)
        # One file per hour (00 minutes only) to limit download size
        for hour in range(0, 24, 6):
            timestamp = target_date.strftime(f"%Y%m%d{hour:02d}0000")
            url = f"{base_url}{timestamp}.export.CSV.zip"

            if already_downloaded(url):
                continue

            rows = download_and_filter(url)
            total_rows += rows
            time.sleep(2)  # Be polite to GDELT servers

    logger.info(f"Historical backfill complete. Total rows: {total_rows}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GDELT Collector")
    parser.add_argument(
        "--mode",
        choices=["once", "schedule", "historical"],
        default="once",
        help="once: single run | schedule: every 15 min | historical: last 7 days",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Days of history to backfill (only used with --mode historical)",
    )
    args = parser.parse_args()

    if args.mode == "once":
        print("Running single GDELT collection...")
        collect_once()

    elif args.mode == "historical":
        print(f"Backfilling {args.days} days of GDELT history...")
        collect_historical(days=args.days)

    elif args.mode == "schedule":
        print("Starting scheduled GDELT collector (every 15 minutes)...")
        print("Press Ctrl+C to stop.")
        collect_once()  # Run immediately on start
        schedule.every(15).minutes.do(collect_once)
        while True:
            schedule.run_pending()
            time.sleep(30)