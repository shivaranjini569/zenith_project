#!/usr/bin/env python3
"""
src/merge_ndvi.py

- Loads your India crop production dataset (adjust path if needed)
- Loads processed NDVI aggregation (ndvi_state_year.csv or ndvi_year.csv)
- Normalizes year and state names and merges NDVI -> produces india_prod_with_ndvi.csv
"""

from pathlib import Path
import pandas as pd
import re

ROOT = Path.cwd()
PROD_PATH = ROOT / "data" / "raw" / "India Agriculture Crop Production.csv"
NDVI_STATE_PATH = ROOT / "data" / "processed" / "ndvi_state_year.csv"
NDVI_YEAR_PATH = ROOT / "data" / "processed" / "ndvi_year.csv"
OUT_PATH = ROOT / "data" / "processed" / "india_prod_with_ndvi.csv"

if not PROD_PATH.exists():
    print("Cannot find India production CSV at", PROD_PATH)
    print("Please update PROD_PATH in this script to the correct file.")
    raise SystemExit(1)

prod = pd.read_csv(PROD_PATH, low_memory=False)
print("Loaded production data:", prod.shape)
print("Columns:", prod.columns.tolist())

# try to detect year column inside production dataset
year_col = None
for c in prod.columns:
    if c.lower() in ("year", "year_ending", "season", "date"):
        year_col = c
        break
# fallback: find col that contains pattern like '2001-02'
if year_col is None:
    for c in prod.columns:
        if prod[c].astype(str).str.contains(r"\d{4}-\d{2}", na=False).any():
            year_col = c
            break

if year_col is None:
    print("Could not detect a year column automatically. Please inspect and set year_col manually.")
    raise SystemExit(1)

print("Using year column:", year_col)
# normalize to integer start year (e.g., 2001-02 -> 2001)
def extract_year(x):
    if pd.isna(x):
        return None
    s = str(x)
    m = re.search(r"(\d{4})", s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except:
        return None

prod["year_int"] = prod[year_col].apply(extract_year)
if prod["year_int"].isna().all():
    print("year_int is all NaN — please fix year column parsing.")
    raise SystemExit(1)

# detect state column
state_col = None
for c in prod.columns:
    if c.lower() in ("state", "state_name", "region", "area"):
        state_col = c
        break
if state_col is None:
    print("Could not detect State column automatically. Columns:", prod.columns.tolist())
    raise SystemExit(1)

prod["State_clean"] = prod[state_col].astype(str).str.strip()

# load NDVI
if NDVI_STATE_PATH.exists():
    ndvi = pd.read_csv(NDVI_STATE_PATH)
    print("Loaded ndvi_state_year.csv with", len(ndvi), "rows")
    # normalize
    ndvi["State"] = ndvi["State"].astype(str).str.strip()
    merged = prod.merge(ndvi, left_on=["State_clean", "year_int"], right_on=["State", "year"], how="left")
else:
    if NDVI_YEAR_PATH.exists():
        ndvi_year = pd.read_csv(NDVI_YEAR_PATH)
        print("Loaded ndvi_year.csv with", len(ndvi_year), "rows")
        merged = prod.merge(ndvi_year, left_on="year_int", right_on="year", how="left")
    else:
        print("No NDVI aggregation file found. Run src/ingest_ndvi.py first.")
        raise SystemExit(1)

print("Merged shape:", merged.shape)
print("NDVI missing after merge:", merged["mean_ndvi"].isna().sum(), "of", merged.shape[0])

# Save result
merged.to_csv(OUT_PATH, index=False)
print("Saved merged file to", OUT_PATH)

# print top missing states for inspection
missing = merged[merged["mean_ndvi"].isna()]
if not missing.empty:
    print("Sample missing (first 10):")
    print(missing[["State_clean", "year_int"]].drop_duplicates().head(10))
    print("\nIf state names differ between datasets, prepare a mapping dict and re-run merge. Example mapping:")
    # show example mapping skeleton
    unique_prod_states = prod["State_clean"].unique()[:20].tolist()
    print("Unique production states (sample):", unique_prod_states)

print("Done.")
