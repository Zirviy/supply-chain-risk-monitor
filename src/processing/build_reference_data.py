"""
build_reference_data.py  —  corrected for actual EMIS file structure
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))
REF  = os.path.join(ROOT, "data", "reference")
OUT  = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

EMIS_SECTOR_MAP = {
    "emis_automotive_india.xlsx":    "Automotive",
    "emis_chemicals_india.xlsx":     "Chemicals",
    "emis_energy_india.xlsx":        "Energy",
    "emis_logistics_india.xlsx":     "Logistics",
    "emis_manufacturing_india.xlsx": "Manufacturing",
    "emis_metals_india.xlsx":        "Metals & Mining",
    "emis_pharma_india.xlsx":        "Pharma",
    "emis_textile_india.xlsx":       "Textiles",
}
EMIS_SECTOR_MAP["emis_energy_india_xlsx.xlsx"] = "Energy"

NUMERIC_COLS = [
    "Total Operating Revenue", "Operating Profit",
    "Net Profit/Loss for the Period", "Total Assets",
    "Cash and Cash Equivalents", "Total Equity",
    "Retained Earnings", "Total Liabilities",
    "Return on Assets (ROA) (%)", "Net Profit Margin (%)",
    "Altman Z-Score", "Debt / Assets (%)", "Debt / Equity (%)",
    "Operating Profit Margin (%)", "Inventory Turnover (x)",
    "Days Inventory Outstanding (Ending)", "Cash Conversion Cycle (Ending)",
    "Export Proportion (%)", "Interest Coverage Ratio (x)",
    "Number of Employees",
]


def find_header_row(path: str) -> int:
    for skip in range(15):
        try:
            df = pd.read_excel(path, header=skip, nrows=1, dtype=str)
            cols = [str(c).strip() for c in df.columns]
            if "Company" in cols or "Num" in cols or "EMIS ID" in cols:
                return skip
        except Exception:
            continue
    return 7


def clean_emis_file(path: str, sector: str) -> pd.DataFrame:
    print(f"  Reading: {os.path.basename(path)}")
    header_row = find_header_row(path)
    print(f"    Header found at row: {header_row}")

    df = pd.read_excel(path, header=header_row, dtype=str)
    df.columns = df.columns.str.strip()

    if "Company" in df.columns:
        df = df.dropna(subset=["Company"])
        df = df[df["Company"].str.strip().str.len() > 0]
        df = df[df["Company"] != "Company"]

    df["sector"] = sector

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("x", "", regex=False)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "N/A": np.nan, "-": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Fiscal Year" in df.columns:
        df["Fiscal Year"] = (
            df["Fiscal Year"].astype(str).str.extract(r"(\d{4})")[0]
        )
        df["Fiscal Year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce")

    if "Company" in df.columns:
        df["company_clean"] = (
            df["Company"].str.strip().str.upper()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"[^\w\s]", "", regex=True)
        )

    if "EMIS ID" in df.columns:
        df = df.drop_duplicates(subset=["EMIS ID"], keep="first")

    print(f"    Rows after clean: {len(df)}")
    return df


def load_all_emis() -> pd.DataFrame:
    emis_dir = os.path.join(REF, "emis")
    frames = []
    for filename, sector in EMIS_SECTOR_MAP.items():
        path = os.path.join(emis_dir, filename)
        if not os.path.exists(path):
            continue
        df = clean_emis_file(path, sector)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No EMIS files found in {emis_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["_fill"] = combined[
        ["Total Operating Revenue", "Total Assets", "Net Profit/Loss for the Period"]
    ].notna().sum(axis=1)

    combined = (
        combined
        .sort_values("_fill", ascending=False)
        .drop_duplicates(subset=["company_clean", "Fiscal Year"], keep="first")
        .drop(columns=["_fill"])
    )

    print(f"\n  EMIS total rows: {len(combined)}")
    print(f"  Unique companies: {combined['company_clean'].nunique()}")
    return combined


if __name__ == "__main__":
    print("="*60)
    print("Loading EMIS files")
    print("="*60)
    emis = load_all_emis()
    out_path = os.path.join(OUT, "supply_chain_nodes.csv")
    emis.to_csv(out_path, index=False)
    print(f"\nDONE → {out_path}")
    print(f"Total rows: {len(emis)}")
    print(emis["sector"].value_counts().to_string())
    print(f"Altman Z-Score available: {emis['Altman Z-Score'].notna().sum()} rows")