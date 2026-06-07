"""
build_trade_data.py  —  corrected for actual ISB file structure
Actual columns: id, date, country_name, alpha_3_code, country_code,
region, region_code, sub_region, sub_region_code, hs_code, commodity,
unit, value_qt, value_rs, value_dl
Files are .csv split by direction and region.
"""
import pandas as pd
import numpy as np
import os
import glob

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))
ISB  = os.path.join(ROOT, "data", "reference", "isb")
OUT  = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

HS_SECTOR_MAP = {
    "27": "Energy",          "26": "Metals & Mining",
    "72": "Metals & Mining", "73": "Metals & Mining",
    "74": "Metals & Mining", "76": "Metals & Mining",
    "28": "Chemicals",       "29": "Chemicals",
    "30": "Pharma",          "38": "Chemicals",
    "39": "Chemicals",       "40": "Chemicals",
    "52": "Textiles",        "54": "Textiles",
    "55": "Textiles",        "61": "Textiles",
    "62": "Textiles",        "84": "Manufacturing",
    "85": "Electronics",     "87": "Automotive",
    "10": "Agri",            "12": "Agri",
    "15": "Agri",            "17": "Agri",
}

HIGH_RISK_COUNTRIES = {
    "China": 0.90, "Taiwan": 0.80, "Russia": 0.90,
    "Ukraine": 0.80, "Pakistan": 0.70, "Bangladesh": 0.50,
    "Vietnam": 0.40, "Indonesia": 0.35, "Malaysia": 0.35,
    "Saudi Arabia": 0.55, "Iran": 0.85, "Myanmar": 0.75,
}


def hs_to_sector(hs_code) -> str:
    if pd.isna(hs_code):
        return "Other"
    try:
        prefix = str(int(float(str(hs_code).strip())))[:2].zfill(2)
        return HS_SECTOR_MAP.get(prefix, "Other")
    except:
        return "Other"


def read_isb_file(path: str) -> pd.DataFrame:
    filename = os.path.basename(path)
    print(f"  Reading: {filename}  ({os.path.getsize(path)//1024} KB)")

    df = pd.read_csv(path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    rename = {
        "country_name": "country", "value_dl": "value_usd",
        "value_rs": "value_inr",   "value_qt": "quantity",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    fname = filename.lower()
    df["direction"] = "Import" if "import" in fname else "Export" if "export" in fname else "Unknown"
    df["source_file"] = filename

    if "date" in df.columns:
        df["year"] = pd.to_datetime(
            df["date"], format="%d-%m-%Y", errors="coerce"
        ).dt.year

    for col in ["value_usd", "value_inr", "quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].str.replace(",", "", regex=False), errors="coerce"
            )

    df["sector"] = df["hs_code"].apply(hs_to_sector) if "hs_code" in df.columns else "Other"

    if "country" in df.columns:
        df = df.dropna(subset=["country"])
        df = df[df["country"].str.strip() != ""]

    print(f"    Rows: {len(df)}")
    return df


def load_all_isb() -> pd.DataFrame:
    files = glob.glob(os.path.join(ISB, "*.csv"))
    if not files:
        print(f"[WARN] No csv files in {ISB}")
        return pd.DataFrame()

    frames = []
    for path in sorted(files):
        try:
            frames.append(read_isb_file(path))
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(path)}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  ISB total rows: {len(combined)}")
    return combined


def build_vulnerability(trade: pd.DataFrame) -> pd.DataFrame:
    imports = trade[trade["direction"] == "Import"].copy()
    agg = (
        imports.groupby(["sector", "country"], as_index=False)["value_usd"]
        .sum().rename(columns={"value_usd": "total_import_usd"})
    )
    sector_total = agg.groupby("sector")["total_import_usd"].sum().rename("sector_total")
    agg = agg.merge(sector_total, on="sector")
    agg["import_share_pct"] = (agg["total_import_usd"] / agg["sector_total"] * 100).round(2)
    agg["country_risk"] = agg["country"].map(HIGH_RISK_COUNTRIES).fillna(0.15)
    agg["vulnerability_score"] = ((agg["import_share_pct"] / 100) * agg["country_risk"]).round(4)
    return agg.sort_values(["sector", "vulnerability_score"], ascending=[True, False])


def build_sector_summary(vuln: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector, grp in vuln.groupby("sector"):
        top_idx = grp["import_share_pct"].idxmax()
        rows.append({
            "sector": sector,
            "total_import_usd": grp["total_import_usd"].sum(),
            "top_source_country": grp.loc[top_idx, "country"],
            "top_source_share_pct": grp["import_share_pct"].max().round(2),
            "hhi": round((grp["import_share_pct"]**2).sum() / 10000, 4),
            "vulnerability_score": round(grp["vulnerability_score"].sum(), 4),
            "n_countries": len(grp),
        })
    return pd.DataFrame(rows).sort_values("vulnerability_score", ascending=False)


if __name__ == "__main__":
    print("="*60)
    print("STEP 1: Loading ISB files")
    print("="*60)
    trade = load_all_isb()

    if not trade.empty:
        vuln = build_vulnerability(trade)
        summary = build_sector_summary(vuln)
        print(summary.to_string(index=False))

        trade.to_csv(os.path.join(OUT, "india_trade_raw.csv"), index=False)
        vuln.to_csv(os.path.join(OUT, "india_trade_vulnerability.csv"), index=False)
        summary.to_csv(os.path.join(OUT, "sector_import_dependency.csv"), index=False)
        print("\nDONE — files saved to data/processed/")