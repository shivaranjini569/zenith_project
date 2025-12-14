#!/usr/bin/env python3
"""
Ingest Kevin Mathews' India Drought NDVI dataset (2000–2023)
Outputs a clean State × Year NDVI table:
    data/processed/ndvi_kevin_state_year.csv
"""

from pathlib import Path
import pandas as pd
import sys

ROOT = Path.cwd()

IN_DIR = ROOT / "data" / "external" / "india_drought"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_ndvi_file():
    """Find any CSV containing NDVI in Kevin's dataset."""
    if not IN_DIR.exists():
        print("ERROR: Folder does not exist:", IN_DIR)
        sys.exit(1)

    all_files = list(IN_DIR.rglob("*.csv"))
    if not all_files:
        print("ERROR: No CSVs found in", IN_DIR)
        sys.exit(1)

    # look for file names containing NDVI
    candidates = [f for f in all_files if "ndvi" in f.name.lower()]

    if candidates:
        print("Found NDVI files:", [c.name for c in candidates])
        return candidates[0]

    # else search for files that have NDVI column
    for f in all_files:
        try:
            df = pd.read_csv(f, nrows=5)
            cols = [c.lower() for c in df.columns]
            if "ndvi" in cols or "ndvi_mean" in cols:
                print("Found NDVI column inside:", f.name)
                return f
        except:
            continue

    print("ERROR: Could not find any NDVI CSV in Kevin dataset.")
    print("Files scanned:", [f.name for f in all_files])
    sys.exit(1)

def main():
    ndvi_file = find_ndvi_file()
    print("Using NDVI file:", ndvi_file)

    try:
        df = pd.read_csv(ndvi_file, low_memory=False)
    except Exception as e:
        print("ERROR reading NDVI file:", e)
        sys.exit(1)

    print("Columns found:", df.columns.tolist())

    # Detect columns
    def detect(col_names, candidates):
        for c in col_names:
            if c.lower() in candidates:
                return c
        for c in col_names:
            for cand in candidates:
                if cand in c.lower():
                    return c
        return None

    state_col = detect(df.columns, ["state", "region", "state_name"])
    year_col  = detect(df.columns, ["year", "year_int", "yr"])
    ndvi_col  = detect(df.columns, ["ndvi", "ndvi_mean", "mean_ndvi"])

    if not state_col or not year_col or not ndvi_col:
        print("ERROR: Could not detect necessary NDVI columns automatically.")
        print("Detected State:", state_col)
        print("Detected Year :", year_col)
        print("Detected NDVI :", ndvi_col)
        sys.exit(1)

    print("Using columns → State:", state_col, "Year:", year_col, "NDVI:", ndvi_col)

    # Clean data
    df[state_col] = df[state_col].astype(str).str.strip()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[ndvi_col] = pd.to_numeric(df[ndvi_col], errors="coerce")

    df = df.dropna(subset=[state_col, year_col, ndvi_col])

    # Aggregation: State × Year
    agg = (
        df.groupby([state_col, year_col])[ndvi_col]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
    )

    agg = agg.rename(
        columns={
            state_col: "State",
            year_col: "Year",
            "mean": "mean_ndvi",
            "min": "min_ndvi",
            "max": "max_ndvi",
            "count": "observations",
        }
    )

    out_path = OUT_DIR / "ndvi_kevin_state_year.csv"
    agg.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("Rows:", len(agg))
    print("\nSample:")
    print(agg.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
