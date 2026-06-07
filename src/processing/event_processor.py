"""
event_processor.py  —  cleans, enriches, and scores raw GDELT events
Maps each event to a supply chain node from supply_chain_nodes_enriched.csv
Outputs: data/processed/processed_events.csv

GDELT columns used:
  GLOBALEVENTID, SQLDATE, Actor1CountryCode, Actor2CountryCode,
  EventCode, GoldsteinScale, NumMentions, NumArticles, AvgTone,
  ActionGeo_CountryCode, ActionGeo_FullName, ActionGeo_Lat, ActionGeo_Long,
  SOURCEURL

─────────────────────────────────────────────────────────────────────────────
COLUMN LAYOUT NOTE  (diagnosed 2026-06-07 from actual downloaded files)
─────────────────────────────────────────────────────────────────────────────
Your GDELT collector downloads files that have 61 columns, not the canonical
58.  The three extra columns appear to be additional geo-feature columns
inserted in the Actor1/Actor2 blocks.  More importantly:

  • Col  7  → Actor1CountryCode  BUT  contains ISO3 codes (e.g. "ISR", "USA")
              NOT the FIPS-10-4 two-letter codes the original code expected.
  • Col 50  → appears to be ActionGeo_FeatureID (numeric strings / short codes)
              NOT ActionGeo_FullName.
  • Col 51  → appears to be ActionGeo_Type (integers 1-5)
              NOT ActionGeo_CountryCode.

Actual ActionGeo block in the 61-col variant (inferred from values):
  col 52 = ActionGeo_FullName   (e.g. "Gaza, Israel")
  col 53 = ActionGeo_CountryCode  (FIPS 2-letter, e.g. "IS")
  col 54 = ActionGeo_ADM1Code
  col 55 = ActionGeo_Lat
  col 56 = ActionGeo_Long
  col 57 = ActionGeo_FeatureID
  col 58 = DATEADDED
  col 59 = SOURCEURL
  (col 60 = extra trailing field — ignored)

Resolution strategy (in order):
  1. Try FIPS code from ActionGeo_CountryCode (col 53 in 61-col files)
  2. Try ISO3 code from Actor1CountryCode (col 7) via ISO3→country dict
  3. Try ISO3 code from Actor2CountryCode (col 17) via ISO3→country dict
  4. Parse the last token of ActionGeo_FullName (col 52) as a country name
  5. Fall back to "Unknown"
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import os
import hashlib
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))
OUT  = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

# ── CAMEO event codes relevant to supply chain disruption
# Source: GDELT CAMEO Codebook — gdeltproject.org/data/documentation/CAMEO.Manual.1.1b3.pdf
CAMEO_SUPPLY_CHAIN = {
    "14":   "Protest/Strike",
    "141":  "Protest/Strike",
    "142":  "Protest/Strike",
    "143":  "Protest/Strike",
    "144":  "Protest/Strike",
    "145":  "Protest/Strike",
    "17":   "Coerce/Sanction",
    "171":  "Coerce/Sanction",
    "172":  "Coerce/Sanction",
    "173":  "Coerce/Sanction",
    "174":  "Coerce/Sanction",
    "175":  "Coerce/Sanction",
    "18":   "Armed Assault",
    "180":  "Armed Assault",
    "181":  "Armed Assault",
    "182":  "Armed Assault",
    "183":  "Armed Assault",
    "19":   "Armed Conflict",
    "190":  "Armed Conflict",
    "193":  "Armed Conflict",
    "195":  "Armed Conflict",
    "196":  "Armed Conflict",
    "20":   "Mass Violence",
    "201":  "Mass Violence",
    "202":  "Mass Violence",
    "0233": "Natural Disaster",
    "0234": "Natural Disaster",
    "0251": "Economic Crisis",
}

CAMEO_SEVERITY_WEIGHT = {
    "Mass Violence":     1.00,
    "Armed Conflict":    0.90,
    "Armed Assault":     0.85,
    "Coerce/Sanction":   0.70,
    "Natural Disaster":  0.75,
    "Economic Crisis":   0.60,
    "Protest/Strike":    0.50,
}

# ─────────────────────────────────────────────
# COUNTRY CODE LOOKUP TABLES
# ─────────────────────────────────────────────

# FIPS 10-4 two-letter → country name
# Used for ActionGeo_CountryCode (col 53 in 61-col files)
FIPS_TO_COUNTRY = {
    "AF": "Afghanistan",    "AG": "Algeria",        "AE": "UAE",
    "AR": "Argentina",      "AS": "Australia",      "AU": "Austria",
    "BA": "Bahrain",        "BE": "Belgium",        "BG": "Bangladesh",
    "BM": "Myanmar",        "BO": "Bolivia",        "BR": "Brazil",
    "BU": "Bulgaria",       "CA": "Canada",         "CE": "Sri Lanka",
    "CG": "Congo",          "CH": "China",          "CI": "Chile",
    "CO": "Colombia",       "CU": "Cuba",           "DA": "Denmark",
    "EC": "Ecuador",        "EG": "Egypt",          "EI": "Ireland",
    "ET": "Ethiopia",       "EZ": "Czech Republic", "FI": "Finland",
    "FR": "France",         "GM": "Germany",        "GR": "Greece",
    "GT": "Guatemala",      "HK": "Hong Kong",      "HU": "Hungary",
    "ID": "Indonesia",      "IN": "India",          "IR": "Iran",
    "IS": "Israel",         "IT": "Italy",          "IZ": "Iraq",
    "JA": "Japan",          "JO": "Jordan",         "KS": "South Korea",
    "KU": "Kuwait",         "LE": "Lebanon",        "LI": "Libya",
    "LY": "Libya",          "MO": "Morocco",        "MX": "Mexico",
    "MY": "Malaysia",       "MZ": "Mozambique",     "NI": "Nigeria",
    "NL": "Netherlands",    "NO": "Norway",         "OM": "Oman",
    "PA": "Panama",         "PE": "Peru",           "PK": "Pakistan",
    "PL": "Poland",         "PO": "Portugal",       "QA": "Qatar",
    "RO": "Romania",        "RP": "Philippines",    "RS": "Russia",
    "SA": "Saudi Arabia",   "SF": "South Africa",   "SN": "Singapore",
    "SO": "Somalia",        "SP": "Spain",          "SU": "Sudan",
    "SW": "Sweden",         "SY": "Syria",          "SZ": "Switzerland",
    "TH": "Thailand",       "TS": "Tunisia",        "TT": "Trinidad and Tobago",
    "TU": "Turkey",         "TW": "Taiwan",         "UK": "United Kingdom",
    "UP": "Ukraine",        "US": "United States",  "UY": "Uruguay",
    "VE": "Venezuela",      "VM": "Vietnam",        "YE": "Yemen",
    "YM": "Yemen",          "ZI": "Zimbabwe",       "ZO": "South Africa",
}

# ISO 3166-1 alpha-3 three-letter → country name
# Used for Actor1CountryCode / Actor2CountryCode in the 61-col GDELT variant
ISO3_TO_COUNTRY = {
    "AFG": "Afghanistan",   "DZA": "Algeria",       "ARE": "UAE",
    "ARG": "Argentina",     "ARM": "Armenia",       "AUS": "Australia",
    "AUT": "Austria",       "AZE": "Azerbaijan",    "BHR": "Bahrain",
    "BGD": "Bangladesh",    "BLR": "Belarus",       "BEL": "Belgium",
    "BOL": "Bolivia",       "BIH": "Bosnia",        "BRA": "Brazil",
    "BGR": "Bulgaria",      "MMR": "Myanmar",       "KHM": "Cambodia",
    "CMR": "Cameroon",      "CAN": "Canada",        "CHL": "Chile",
    "CHN": "China",         "COL": "Colombia",      "COD": "Congo",
    "CRI": "Costa Rica",    "HRV": "Croatia",       "CUB": "Cuba",
    "CZE": "Czech Republic","DNK": "Denmark",       "DOM": "Dominican Republic",
    "ECU": "Ecuador",       "EGY": "Egypt",         "SLV": "El Salvador",
    "ETH": "Ethiopia",      "FIN": "Finland",       "FRA": "France",
    "GEO": "Georgia",       "DEU": "Germany",       "GHA": "Ghana",
    "GRC": "Greece",        "GTM": "Guatemala",     "HTI": "Haiti",
    "HND": "Honduras",      "HKG": "Hong Kong",     "HUN": "Hungary",
    "IND": "India",         "IDN": "Indonesia",     "IRN": "Iran",
    "IRQ": "Iraq",          "IRL": "Ireland",       "ISR": "Israel",
    "ITA": "Italy",         "JAM": "Jamaica",       "JPN": "Japan",
    "JOR": "Jordan",        "KAZ": "Kazakhstan",    "KEN": "Kenya",
    "PRK": "North Korea",   "KOR": "South Korea",   "KWT": "Kuwait",
    "KGZ": "Kyrgyzstan",    "LAO": "Laos",          "LBN": "Lebanon",
    "LBY": "Libya",         "LTU": "Lithuania",     "MYS": "Malaysia",
    "MLI": "Mali",          "MEX": "Mexico",        "MDA": "Moldova",
    "MAR": "Morocco",       "MOZ": "Mozambique",    "NAM": "Namibia",
    "NPL": "Nepal",         "NLD": "Netherlands",   "NZL": "New Zealand",
    "NIC": "Nicaragua",     "NER": "Niger",         "NGA": "Nigeria",
    "NOR": "Norway",        "OMN": "Oman",          "PAK": "Pakistan",
    "PAN": "Panama",        "PRY": "Paraguay",      "PER": "Peru",
    "PHL": "Philippines",   "POL": "Poland",        "PRT": "Portugal",
    "QAT": "Qatar",         "ROU": "Romania",       "RUS": "Russia",
    "RWA": "Rwanda",        "SAU": "Saudi Arabia",  "SEN": "Senegal",
    "SRB": "Serbia",        "SGP": "Singapore",     "SVK": "Slovakia",
    "SOM": "Somalia",       "ZAF": "South Africa",  "SSD": "South Sudan",
    "ESP": "Spain",         "LKA": "Sri Lanka",     "SDN": "Sudan",
    "SWE": "Sweden",        "CHE": "Switzerland",   "SYR": "Syria",
    "TWN": "Taiwan",        "TJK": "Tajikistan",    "TZA": "Tanzania",
    "THA": "Thailand",      "TLS": "Timor-Leste",   "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia",       "TUR": "Turkey",        "TKM": "Turkmenistan",
    "UGA": "Uganda",        "UKR": "Ukraine",       "GBR": "United Kingdom",
    "USA": "United States", "URY": "Uruguay",       "UZB": "Uzbekistan",
    "VEN": "Venezuela",     "VNM": "Vietnam",       "YEM": "Yemen",
    "ZMB": "Zambia",        "ZWE": "Zimbabwe",
}

# Country name normalizer — handles variations that appear in ActionGeo_FullName
# and need to match node keys from supply_chain_nodes_enriched.csv
COUNTRY_NAME_ALIASES = {
    "United States of America": "United States",
    "US":                        "United States",
    "UK":                        "United Kingdom",
    "Great Britain":             "United Kingdom",
    "England":                   "United Kingdom",
    "Korea":                     "South Korea",
    "Republic of Korea":         "South Korea",
    "Democratic Republic of the Congo": "Congo",
    "DR Congo":                  "Congo",
    "UAE":                       "UAE",
    "United Arab Emirates":      "UAE",
    "Russia":                    "Russia",
    "Russian Federation":        "Russia",
    "Iran":                      "Iran",
    "Islamic Republic of Iran":  "Iran",
    "Vietnam":                   "Vietnam",
    "Viet Nam":                  "Vietnam",
    "Myanmar":                   "Myanmar",
    "Burma":                     "Myanmar",
    "Sri Lanka":                 "Sri Lanka",
    "Ceylon":                    "Sri Lanka",
    "Taiwan":                    "Taiwan",
    "Taiwan, Province of China": "Taiwan",
}


# ─────────────────────────────────────────────
# GDELT COLUMN SCHEMA — 61-col variant
# ─────────────────────────────────────────────
# The standard GDELT 2.0 spec has 58 columns.
# The files downloaded by your gdelt_collector.py have 61 columns.
# Diagnostic (2026-06-07) confirmed:
#   col 7  = Actor1CountryCode  (ISO3, e.g. "ISR")
#   col 17 = Actor2CountryCode  (ISO3)
#   col 26 = EventCode          (CAMEO)
#   col 50 = ActionGeo_FeatureID or similar  ← NOT FullName
#   col 51 = ActionGeo_Type  (int 1-5)       ← NOT CountryCode
#   col 52 = ActionGeo_FullName  (e.g. "Gaza, Israel")
#   col 53 = ActionGeo_CountryCode  (FIPS 2-letter, e.g. "IS")
#   col 55 = ActionGeo_Lat
#   col 56 = ActionGeo_Long
#   col 59 = SOURCEURL
#
# We define BOTH schemas and auto-detect which applies at load time.

GDELT_COLS_58 = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",          # 0-4
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",   # 5-8
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",          # 9-11
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",                   # 12-14
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",   # 15-18
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",          # 19-21
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",                   # 22-24
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",              # 25-28
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",                # 29-32
    "NumArticles", "AvgTone",                                                  # 33-34
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",           # 35-37
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long",                  # 38-40
    "Actor1Geo_FeatureID",                                                     # 41
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",           # 42-44
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long",                  # 45-47
    "Actor2Geo_FeatureID",                                                     # 48
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",           # 49-51
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long",                  # 52-54
    "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL",                          # 55-57
]

# 61-col variant: 3 extra columns inserted, shifting the ActionGeo block.
# Based on diagnostic: col 50=FeatureID-ish, 51=Type(int), 52=FullName, 53=CountryCode.
# Best fit is an extra column in each geo sub-block (Actor1, Actor2, ActionGeo).
GDELT_COLS_61 = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",          # 0-4
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",   # 5-8
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",          # 9-11
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",                   # 12-14
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",   # 15-18
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",          # 19-21
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",                   # 22-24
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",              # 25-28
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",                # 29-32
    "NumArticles", "AvgTone",                                                  # 33-34
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",           # 35-37
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long",                  # 38-40
    "Actor1Geo_FeatureID", "Actor1Geo_Extra",                                  # 41-42  ← extra col
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",           # 43-45
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long",                  # 46-48
    "Actor2Geo_FeatureID", "Actor2Geo_Extra",                                  # 49-50  ← extra col (=col50 = Feature-ID-like values seen)
    "ActionGeo_Type",                                                           # 51     ← confirmed int 1-5
    "ActionGeo_FullName",                                                       # 52     ← place names go here
    "ActionGeo_CountryCode",                                                    # 53     ← FIPS codes go here
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long",                  # 54-56
    "ActionGeo_FeatureID",                                                     # 57     ← extra col
    "DATEADDED", "SOURCEURL",                                                  # 58-59
    "Extra60",                                                                 # 60
]


def load_gdelt_csv(path: str) -> pd.DataFrame:
    """
    Loads a GDELT export CSV, assigns correct column names based on actual
    column count, and coerces numeric columns.
    """
    df = pd.read_csv(
        path, sep="\t", header=None, dtype=str,
        on_bad_lines="skip", low_memory=False,
    )
    n = len(df.columns)

    if n == 58:
        schema = GDELT_COLS_58
    elif n == 61:
        schema = GDELT_COLS_61
    else:
        # Best-effort: use whichever schema is closer, pad/truncate
        schema = GDELT_COLS_61 if n > 58 else GDELT_COLS_58
        print(f"    [WARN] Unexpected column count {n}, using closest schema ({len(schema)} cols)")

    # Assign names — handle mismatch gracefully
    if n <= len(schema):
        df.columns = schema[:n]
    else:
        df.columns = schema + [f"col_{i}" for i in range(len(schema), n)]

    # Coerce numeric columns
    for col in ["GoldsteinScale", "NumMentions", "NumArticles", "AvgTone",
                "ActionGeo_Lat", "ActionGeo_Long"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────

def make_event_hash(row) -> str:
    """
    Dedup key: EventCode + ActionGeo_CountryCode + date (day precision).
    Same event covered by multiple outlets on same day → one record.
    """
    key = (
        f"{row.get('EventCode', '')}_"
        f"{row.get('ActionGeo_CountryCode', '')}_"
        f"{str(row.get('SQLDATE', ''))[:8]}"
    )
    return hashlib.md5(key.encode()).hexdigest()


# ─────────────────────────────────────────────
# CAMEO FILTER
# ─────────────────────────────────────────────

def filter_supply_chain_events(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose EventCode prefix matches tracked CAMEO codes."""
    if "EventCode" not in df.columns:
        return df

    def is_relevant(code):
        if pd.isna(code):
            return False
        code = str(code).strip()
        for cameo in CAMEO_SUPPLY_CHAIN:
            if code == cameo or code.startswith(cameo):
                return True
        return False

    return df[df["EventCode"].apply(is_relevant)].copy()


# ─────────────────────────────────────────────
# COUNTRY RESOLUTION
# ─────────────────────────────────────────────

def _normalize_country_name(raw: str) -> str:
    """Apply alias table to canonicalize country names."""
    raw = raw.strip()
    return COUNTRY_NAME_ALIASES.get(raw, raw)


def _country_from_fips(code: str) -> str:
    """Look up FIPS 10-4 two-letter code."""
    code = str(code).strip().upper()
    if len(code) == 2 and code.isalpha():
        return FIPS_TO_COUNTRY.get(code, "")
    return ""


def _country_from_iso3(code: str) -> str:
    """Look up ISO 3166-1 alpha-3 three-letter code."""
    code = str(code).strip().upper()
    if len(code) == 3 and code.isalpha():
        return ISO3_TO_COUNTRY.get(code, "")
    return ""


def _country_from_fullname(full_name: str) -> str:
    """
    Extract country from ActionGeo_FullName which is typically
    'City, Region, Country' or 'Region, Country'.
    Takes the last comma-separated token and checks against known names.
    """
    if not full_name or full_name.lower() in ("nan", "none", ""):
        return ""
    parts = [p.strip() for p in full_name.split(",")]
    # Try from right to left — rightmost part is usually country
    for part in reversed(parts):
        part = part.strip()
        if not part:
            continue
        # Direct match
        if part in FIPS_TO_COUNTRY.values():
            return _normalize_country_name(part)
        # Alias match
        normalized = _normalize_country_name(part)
        if normalized in FIPS_TO_COUNTRY.values():
            return normalized
        # Try FIPS lookup (sometimes full_name contains a code not a name)
        fips_result = _country_from_fips(part)
        if fips_result:
            return fips_result
        iso3_result = _country_from_iso3(part)
        if iso3_result:
            return iso3_result
    return ""


def resolve_country(row: dict) -> str:
    """
    Multi-strategy country resolution (in priority order):

    1. FIPS code from ActionGeo_CountryCode
       (col 53 in 61-col files, col 51 in 58-col files — auto-handled by schema)
    2. ISO3 code from Actor1CountryCode (col 7 in both schemas)
    3. ISO3 code from Actor2CountryCode (col 17 in both schemas)
    4. Parse ActionGeo_FullName last token
    5. "Unknown"
    """
    # 1. ActionGeo_CountryCode — expect FIPS 2-letter
    fips = str(row.get("ActionGeo_CountryCode", "")).strip()
    if fips and fips.lower() not in ("nan", "none", ""):
        result = _country_from_fips(fips)
        if result:
            return result
        # Some GDELT variants put ISO3 here too — try that
        result = _country_from_iso3(fips)
        if result:
            return result

    # 2. Actor1CountryCode — ISO3 in the 61-col variant
    a1 = str(row.get("Actor1CountryCode", "")).strip()
    if a1 and a1.lower() not in ("nan", "none", ""):
        result = _country_from_iso3(a1)
        if result:
            return result
        result = _country_from_fips(a1)
        if result:
            return result

    # 3. Actor2CountryCode
    a2 = str(row.get("Actor2CountryCode", "")).strip()
    if a2 and a2.lower() not in ("nan", "none", ""):
        result = _country_from_iso3(a2)
        if result:
            return result
        result = _country_from_fips(a2)
        if result:
            return result

    # 4. ActionGeo_FullName text parsing
    full_name = str(row.get("ActionGeo_FullName", ""))
    result = _country_from_fullname(full_name)
    if result:
        return result

    return "Unknown"


# ─────────────────────────────────────────────
# ENRICHMENT
# ─────────────────────────────────────────────

def enrich_event(df: pd.DataFrame) -> pd.DataFrame:
    """Add country name, CAMEO category, and parsed event date."""
    df["country"] = df.apply(resolve_country, axis=1)

    def get_category(code):
        code = str(code).strip()
        for cameo, label in CAMEO_SUPPLY_CHAIN.items():
            if code == cameo or code.startswith(cameo):
                return label
        return "Other"

    df["event_category"] = df["EventCode"].apply(get_category)

    if "SQLDATE" in df.columns:
        df["event_date"] = pd.to_datetime(
            df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce"
        )

    return df


# ─────────────────────────────────────────────
# SEVERITY SCORING
# ─────────────────────────────────────────────

def score_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes preliminary_severity_score (0–1) per event.

    Formula:
      severity = 0.40 × goldstein_norm
               + 0.35 × cameo_weight
               + 0.25 × mention_norm

    goldstein_norm = (10 − GoldsteinScale) / 20
      GoldsteinScale: −10 (most destabilizing) to +10 (most stabilizing).
      Inverted so 1.0 = maximum disruption risk.
      Source: Goldstein (1992), Journal of Conflict Resolution.

    mention_norm = log1p(NumMentions) / log1p(1000)
      Log scale, 1000-mention ceiling. Captures media intensity
      without letting viral events dominate the score.
    """
    gs = df["GoldsteinScale"].fillna(0).clip(-10, 10)
    df["goldstein_norm"] = ((10 - gs) / 20).round(4)

    df["cameo_weight"] = df["event_category"].map(CAMEO_SEVERITY_WEIGHT).fillna(0.3)

    mentions = df["NumMentions"].fillna(1).clip(1, None)
    df["mention_norm"] = (np.log1p(mentions) / np.log1p(1000)).round(4)

    df["preliminary_severity"] = (
        0.40 * df["goldstein_norm"] +
        0.35 * df["cameo_weight"] +
        0.25 * df["mention_norm"]
    ).round(4)

    return df


# ─────────────────────────────────────────────
# NODE MATCHER
# ─────────────────────────────────────────────

def load_nodes() -> pd.DataFrame:
    path = os.path.join(OUT, "supply_chain_nodes_enriched.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "supply_chain_nodes_enriched.csv not found. "
            "Run node_mapper.py first."
        )
    return pd.read_csv(path)


def match_events_to_nodes(
    events: pd.DataFrame, nodes: pd.DataFrame
) -> pd.DataFrame:
    """
    Matches each event to supply chain nodes by country name.
    One event may match multiple nodes (one per sector in that country).
    Unmatched events are kept with matched_node_id = None.
    """
    # Build {country_name: [{node_id, sector}, ...]} lookup
    trade_nodes = nodes[nodes["node_type"] == "trade"]
    country_nodes: dict = (
        trade_nodes
        .groupby("country")[["node_id", "sector"]]
        .apply(lambda x: x.to_dict("records"))
        .to_dict()
    )

    matched_rows = []
    for _, event in events.iterrows():
        country = event.get("country", "Unknown")
        node_matches = country_nodes.get(country, [])

        if not node_matches:
            row = event.to_dict()
            row["matched_node_id"] = None
            row["matched_sector"]  = None
            matched_rows.append(row)
        else:
            for nm in node_matches:
                row = event.to_dict()
                row["matched_node_id"] = nm["node_id"]
                row["matched_sector"]  = nm["sector"]
                matched_rows.append(row)

    return pd.DataFrame(matched_rows)


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def process_gdelt_file(path: str, nodes: pd.DataFrame) -> pd.DataFrame:
    """Full processing pipeline for one GDELT export file."""
    print(f"  Processing: {os.path.basename(path)}")

    df = load_gdelt_csv(path)
    print(f"    Raw rows: {len(df)} | Cols: {len(df.columns)}")

    df = filter_supply_chain_events(df)
    print(f"    After CAMEO filter: {len(df)}")

    if df.empty:
        return pd.DataFrame()

    df = enrich_event(df)

    # Quick resolution quality report
    resolved   = (df["country"] != "Unknown").sum()
    unresolved = (df["country"] == "Unknown").sum()
    print(f"    Country resolved: {resolved} | Unknown: {unresolved}")

    df = score_event(df)

    df["event_hash"] = df.apply(make_event_hash, axis=1)
    df = df.drop_duplicates(subset=["event_hash"])
    print(f"    After dedup: {len(df)}")

    df = match_events_to_nodes(df, nodes)
    matched = df["matched_node_id"].notna().sum()
    print(f"    Matched to nodes: {matched} / {len(df)}")

    return df


def process_all(gdelt_dir: str = None) -> pd.DataFrame:
    """
    Process all GDELT *.CSV / *.csv files in a directory.
    Falls back to synthetic test mode if no files found.
    """
    nodes = load_nodes()

    if gdelt_dir is None:
        gdelt_dir = os.path.join(ROOT, "data", "raw", "gdelt")

    import glob
    files = list({
        os.path.normcase(f): f
        for f in (
            glob.glob(os.path.join(gdelt_dir, "*.CSV")) +
            glob.glob(os.path.join(gdelt_dir, "*.csv"))
        )
    }.values())

    if not files:
        print(f"[INFO] No GDELT files found in {gdelt_dir}")
        print("       Run gdelt_collector.py first to download live data.")
        print("       Running in test mode with synthetic sample...")
        return _test_mode(nodes)

    all_frames = []
    for f in sorted(files):
        result = process_gdelt_file(f, nodes)
        if not result.empty:
            all_frames.append(result)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["event_hash"])
    return combined


def _test_mode(nodes: pd.DataFrame) -> pd.DataFrame:
    """
    20 synthetic events (mix of ISO3 and FIPS codes) to validate the
    full pipeline before live data is available.
    """
    print("  Generating synthetic test events...")
    test_events = [
        # ISO3 actor codes (as seen in 61-col files)
        {"EventCode": "145", "Actor1CountryCode": "CHN", "Actor2CountryCode": "",
         "ActionGeo_CountryCode": "CH", "ActionGeo_FullName": "Shanghai, China",
         "GoldsteinScale": -7.0, "NumMentions": 450, "NumArticles": 120, "AvgTone": -4.2,
         "SQLDATE": "20260601", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T001"},
        {"EventCode": "172", "Actor1CountryCode": "USA", "Actor2CountryCode": "RUS",
         "ActionGeo_CountryCode": "RS", "ActionGeo_FullName": "Moscow, Russia",
         "GoldsteinScale": -8.0, "NumMentions": 800, "NumArticles": 200, "AvgTone": -6.1,
         "SQLDATE": "20260601", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T002"},
        {"EventCode": "180", "Actor1CountryCode": "IRN", "Actor2CountryCode": "",
         "ActionGeo_CountryCode": "IR", "ActionGeo_FullName": "Tehran, Iran",
         "GoldsteinScale": -9.0, "NumMentions": 600, "NumArticles": 180, "AvgTone": -7.5,
         "SQLDATE": "20260602", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T003"},
        {"EventCode": "0233", "Actor1CountryCode": "IND", "Actor2CountryCode": "",
         "ActionGeo_CountryCode": "IN", "ActionGeo_FullName": "Chennai, India",
         "GoldsteinScale": -5.0, "NumMentions": 200, "NumArticles": 60, "AvgTone": -3.0,
         "SQLDATE": "20260603", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T004"},
        {"EventCode": "193", "Actor1CountryCode": "YEM", "Actor2CountryCode": "",
         "ActionGeo_CountryCode": "YM", "ActionGeo_FullName": "Aden, Yemen",
         "GoldsteinScale": -10.0, "NumMentions": 950, "NumArticles": 300, "AvgTone": -8.0,
         "SQLDATE": "20260604", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T005"},
        {"EventCode": "174", "Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
         "ActionGeo_CountryCode": "TW", "ActionGeo_FullName": "Taipei, Taiwan",
         "GoldsteinScale": -7.5, "NumMentions": 1200, "NumArticles": 350, "AvgTone": -5.8,
         "SQLDATE": "20260605", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T006"},
        {"EventCode": "145", "Actor1CountryCode": "VNM", "Actor2CountryCode": "",
         "ActionGeo_CountryCode": "VM", "ActionGeo_FullName": "Ho Chi Minh City, Vietnam",
         "GoldsteinScale": -4.0, "NumMentions": 150, "NumArticles": 40, "AvgTone": -2.5,
         "SQLDATE": "20260605", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T007"},
        {"EventCode": "0251", "Actor1CountryCode": "PAK", "Actor2CountryCode": "",
         "ActionGeo_CountryCode": "PK", "ActionGeo_FullName": "Karachi, Pakistan",
         "GoldsteinScale": -6.0, "NumMentions": 300, "NumArticles": 90, "AvgTone": -4.8,
         "SQLDATE": "20260606", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T008"},
        # Test FullName-only resolution (no valid country code)
        {"EventCode": "18", "Actor1CountryCode": "", "Actor2CountryCode": "",
         "ActionGeo_CountryCode": "", "ActionGeo_FullName": "Port Klang, Malaysia",
         "GoldsteinScale": -8.5, "NumMentions": 500, "NumArticles": 150, "AvgTone": -6.0,
         "SQLDATE": "20260607", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T009"},
        {"EventCode": "195", "Actor1CountryCode": "NGA", "Actor2CountryCode": "",
         "ActionGeo_CountryCode": "NI", "ActionGeo_FullName": "Lagos, Nigeria",
         "GoldsteinScale": -9.0, "NumMentions": 400, "NumArticles": 120, "AvgTone": -7.0,
         "SQLDATE": "20260607", "SOURCEURL": "test://synthetic", "GLOBALEVENTID": "T010"},
    ]

    df = pd.DataFrame(test_events)
    df = enrich_event(df)
    df = score_event(df)
    df["event_hash"] = df.apply(make_event_hash, axis=1)
    df = match_events_to_nodes(df, nodes)
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Event Processor")
    print("=" * 60)

    result = process_all()

    if result.empty:
        print("No events processed.")
    else:
        out_path = os.path.join(OUT, "processed_events.csv")
        result.to_csv(out_path, index=False)
        print(f"\nDONE → {out_path}")
        print(f"Total event-node rows: {len(result)}")

        # Resolution quality summary
        unknown = (result["country"] == "Unknown").sum()
        resolved = (result["country"] != "Unknown").sum()
        print(f"\nCountry resolution: {resolved} resolved | {unknown} unknown "
              f"({100*resolved/max(len(result),1):.1f}% hit rate)")

        matched = result["matched_node_id"].notna().sum()
        print(f"Node matching:       {matched} matched | "
              f"{len(result)-matched} unmatched "
              f"({100*matched/max(len(result),1):.1f}% match rate)")

        print(f"\nSeverity distribution:")
        print(result["preliminary_severity"].describe().round(4).to_string())

        print(f"\nTop 10 highest severity events:")
        cols = ["event_date", "country", "event_category",
                "preliminary_severity", "matched_sector", "SOURCEURL"]
        cols = [c for c in cols if c in result.columns]
        print(result.nlargest(10, "preliminary_severity")[cols].to_string(index=False))

        print(f"\nCountry distribution (top 15):")
        print(result["country"].value_counts().head(15).to_string())