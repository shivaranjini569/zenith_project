#!/usr/bin/env python3
# src/ingest_ndvi.py (improved)
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import math

ROOT = Path.cwd()
NDVI_DIR = ROOT / "data" / "external" / "ndvi"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config
CSV_CHUNK_SIZE = 200_000  # read csv in chunks if file large
RAW_SAVE_LIMIT = 2_000_000  # maximum rows to write to ndvi_raw.csv (prevents OOM in accidental huge concat)
POSSIBLE_DATE_NAMES = ["date", "acq_date", "acquisition_date", "timestamp", "time", "datetime", "obs_date"]
POSSIBLE_NDVI_NAMES = ["ndvi", "ndvi_mean", "ndvi_mean_value", "ndvi_value", "ndvi_med", "value", "mean_ndvi", "NDVI_mean", "NDVI"]
POSSIBLE_LAT = ["lat","latitude","y"]
POSSIBLE_LON = ["lon","longitude","long","x"]
POSSIBLE_STATE = ["state","region","admin1","province","state_name","region_name","district","zone"]

def try_read_any_file(path: Path):
    """Read a csv/xlsx; if csv and big, return iterator of chunks, else df"""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        size = path.stat().st_size
        if size > 50_000_000:  # >50MB -> use chunking to avoid memory blow
            return pd.read_csv(path, chunksize=CSV_CHUNK_SIZE, low_memory=False)
        else:
            return pd.read_csv(path, low_memory=False)
    elif suffix in (".xls", ".xlsx"):
        # For Excel, try first sheet
        return pd.read_excel(path, sheet_name=0, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {path}")

def detect_column(cols_lower, candidates):
    for cand in candidates:
        if cand in cols_lower:
            return cand  # return the lowercase name found
    # try substring match
    for c in cols_lower:
        for cand in candidates:
            if cand in c:
                return c
    return None

def secure_colname_match(df, candidates):
    """Return actual column name from df if any candidate matches (case-insensitive)."""
    cols = df.columns.tolist()
    cols_lower = [c.lower() for c in cols]
    for cand in candidates:
        if cand.lower() in cols_lower:
            idx = cols_lower.index(cand.lower())
            return cols[idx]
    # substring & fuzzy fallback
    for i,c in enumerate(cols_lower):
        for cand in candidates:
            if cand.lower() in c:
                return cols[i]
    return None

def map_latlon_to_state_if_possible(df_points, lon_col, lat_col, shapefile_path=None):
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except Exception as e:
        print("geopandas not available:", e)
        return None
    shp = Path(shapefile_path) if shapefile_path else Path("data/external/india_states.geojson")
    if not shp.exists():
        print("Shapefile not found at", shp)
        return None
    states = gpd.read_file(str(shp))
    if states.crs is None:
        states = states.set_crs("EPSG:4326")
    # create GeoDataFrame from points
    pts = df_points.copy()
    pts = pts.dropna(subset=[lon_col, lat_col])
    if pts.empty:
        print("No valid lat/lon points to map.")
        return None
    try:
        gdf_pts = gpd.GeoDataFrame(pts, geometry=[Point(xy) for xy in zip(pts[lon_col], pts[lat_col])], crs="EPSG:4326")
        joined = gpd.sjoin(gdf_pts, states, how="left", predicate="within")
    except Exception as e:
        print("geopandas spatial join failed:", e)
        return None
    # try to find best name column in states
    for col in ["st_nm","STATE_NAME","NAME_1","state_name","STATE","name","ADM1_NAME"]:
        if col in joined.columns:
            joined = joined.rename(columns={col:"State_from_shapefile"})
            break
    if "State_from_shapefile" not in joined.columns:
        # fallback: pick any column containing 'name'
        for c in joined.columns:
            if 'name' in c.lower():
                joined = joined.rename(columns={c:"State_from_shapefile"})
                break
    return pd.DataFrame(joined.drop(columns=["geometry","index_right"], errors="ignore"))

def append_or_accumulate(agg_df, chunk):
    if agg_df is None:
        return chunk
    return pd.concat([agg_df, chunk], ignore_index=True)

def main():
    if not NDVI_DIR.exists():
        print("No ndvi folder at", NDVI_DIR, " — place your ndvi files there.")
        sys.exit(1)

    files = sorted(list(NDVI_DIR.rglob("*.*")))
    files = [f for f in files if f.suffix.lower() in (".csv", ".xls", ".xlsx")]
    if not files:
        print("No CSV/XLSX files found in", NDVI_DIR)
        sys.exit(1)

    print("Found files:", [f.name for f in files])

    # We'll build a small list of frames or aggregated stats without loading everything at once.
    raw_parts = []
    total_rows = 0
    # Track if we found a state column anywhere
    found_state = False
    found_latlon = False

    # We'll accumulate per-file region/year aggregates (if possible) to then merge
    perfile_aggregates = []

    for f in files:
        print("Processing", f.name)
        try:
            obj = try_read_any_file(f)
        except Exception as e:
            print("Failed to open", f, ":", e)
            continue

        # if obj is a TextFileReader (chunks)
        if hasattr(obj, "__iter__") and not isinstance(obj, pd.DataFrame):
            # chunked processing for large csv
            agg_this_file = None
            row_count = 0
            for chunk in obj:
                row_count += len(chunk)
                # detect columns in this chunk
                ndvi_col = secure_colname_match(chunk, POSSIBLE_NDVI_NAMES)
                date_col = secure_colname_match(chunk, POSSIBLE_DATE_NAMES)
                state_col = secure_colname_match(chunk, POSSIBLE_STATE)
                lat_c = secure_colname_match(chunk, POSSIBLE_LAT)
                lon_c = secure_colname_match(chunk, POSSIBLE_LON)

                # if ndvi missing in this chunk skip
                if ndvi_col is None:
                    continue

                # ensure numeric
                chunk[ndvi_col] = pd.to_numeric(chunk[ndvi_col], errors="coerce")
                # parse date to year if possible
                if date_col:
                    chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
                    chunk["year"] = chunk[date_col].dt.year
                else:
                    chunk["year"] = pd.NA

                # prefer state col if available
                if state_col:
                    found_state = True
                    chunk[state_col] = chunk[state_col].astype(str).str.strip()
                    grp = chunk.dropna(subset=["year", ndvi_col]).groupby([state_col, "year"])[ndvi_col].agg(["mean","min","max","count"]).reset_index()
                    grp = grp.rename(columns={state_col:"region","mean":"mean_ndvi","min":"min_ndvi","max":"max_ndvi","count":"observations"})
                    agg_this_file = append_or_accumulate(agg_this_file, grp)
                elif lat_c and lon_c:
                    found_latlon = True
                    # aggregate by rounded lat/lon + year
                    chunk = chunk.dropna(subset=[lat_c, lon_c, ndvi_col])
                    chunk["lat_round"] = chunk[lat_c].round(3)
                    chunk["lon_round"] = chunk[lon_c].round(3)
                    grp = chunk.groupby(["lat_round","lon_round","year"])[ndvi_col].agg(["mean","count"]).reset_index()
                    grp = grp.rename(columns={"mean":"mean_ndvi","count":"observations"})
                    agg_this_file = append_or_accumulate(agg_this_file, grp)
                else:
                    # year-only aggregation
                    grp = chunk.dropna(subset=["year", ndvi_col]).groupby(["year"])[ndvi_col].agg(["mean","count"]).reset_index()
                    grp = grp.rename(columns={"mean":"mean_ndvi","count":"observations"})
                    agg_this_file = append_or_accumulate(agg_this_file, grp)

                # keep a small sample of raw rows (limit total)
                if len(raw_parts) == 0:
                    sample = chunk.head(1000)
                else:
                    sample = chunk.head(200)
                raw_parts.append(sample)
            print(f"Read (chunked) ~{row_count} rows from {f.name}")
            if agg_this_file is not None:
                perfile_aggregates.append((f.name, agg_this_file))
        else:
            # obj is DataFrame (small file)
            df = obj
            total_rows += len(df)
            # store sample raw
            raw_parts.append(df.head(2000))
            # detect columns
            ndvi_col = secure_colname_match(df, POSSIBLE_NDVI_NAMES)
            date_col = secure_colname_match(df, POSSIBLE_DATE_NAMES)
            state_col = secure_colname_match(df, POSSIBLE_STATE)
            lat_c = secure_colname_match(df, POSSIBLE_LAT)
            lon_c = secure_colname_match(df, POSSIBLE_LON)

            if ndvi_col is None:
                print("Skipping", f.name, "- no NDVI-like column found.")
                continue

            df[ndvi_col] = pd.to_numeric(df[ndvi_col], errors="coerce")
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df["year"] = df[date_col].dt.year
            else:
                df["year"] = pd.NA

            if state_col:
                found_state = True
                df[state_col] = df[state_col].astype(str).str.strip()
                grp = df.dropna(subset=["year", ndvi_col]).groupby([state_col, "year"])[ndvi_col].agg(["mean","min","max","count"]).reset_index()
                grp = grp.rename(columns={state_col:"region","mean":"mean_ndvi","min":"min_ndvi","max":"max_ndvi","count":"observations"})
                perfile_aggregates.append((f.name, grp))
            elif lat_c and lon_c:
                found_latlon = True
                df = df.dropna(subset=[lat_c, lon_c, ndvi_col])
                df["lat_round"] = df[lat_c].round(3)
                df["lon_round"] = df[lon_c].round(3)
                grp = df.groupby(["lat_round","lon_round","year"])[ndvi_col].agg(["mean","count"]).reset_index()
                grp = grp.rename(columns={"mean":"mean_ndvi","count":"observations"})
                perfile_aggregates.append((f.name, grp))
            else:
                grp = df.dropna(subset=["year", ndvi_col]).groupby(["year"])[ndvi_col].agg(["mean","count"]).reset_index()
                grp = grp.rename(columns={"mean":"mean_ndvi","count":"observations"})
                perfile_aggregates.append((f.name, grp))

    # Save a combined raw sample (avoid writing enormous raw file)
    if raw_parts:
        raw_concat = pd.concat(raw_parts, ignore_index=True)
        if len(raw_concat) > RAW_SAVE_LIMIT:
            raw_concat = raw_concat.head(RAW_SAVE_LIMIT)
            print(f"Raw sample truncated to {RAW_SAVE_LIMIT} rows to avoid huge file.")
        raw_concat.to_csv(OUT_DIR / "ndvi_raw.csv", index=False)
        print("Saved raw sample to", OUT_DIR / "ndvi_raw.csv")
    else:
        print("No raw rows captured (no NDVI columns found in files?)")

    # Merge per-file aggregates into a unified table
    if not perfile_aggregates:
        print("No per-file aggregates created — there was no year/ndvi info found.")
        sys.exit(1)

    # unify logic: if any aggregate has 'region' column use region-based merging
    use_region = any('region' in df.columns for _, df in perfile_aggregates)
    if use_region:
        agg_frames = []
        for name, df in perfile_aggregates:
            if 'region' in df.columns:
                agg_frames.append(df[['region','year','mean_ndvi','min_ndvi','max_ndvi','observations']])
        combined = pd.concat(agg_frames, ignore_index=True)
        # group again to ensure duplicates handled
        combined = combined.groupby(['region','year']).agg({'mean_ndvi':'mean','min_ndvi':'min','max_ndvi':'max','observations':'sum'}).reset_index()
        combined = combined.rename(columns={'region':'region'})
        out_path = OUT_DIR / "ndvi_region_year.csv"
        combined.to_csv(out_path, index=False)
        print("Saved region-year NDVI to", out_path, "with rows:", len(combined))
        return

    # if no region but lat/lon grid aggregates exist
    latlon_present = any('lat_round' in df.columns for _, df in perfile_aggregates)
    if latlon_present:
        agg_frames = []
        for name, df in perfile_aggregates:
            if 'lat_round' in df.columns:
                agg_frames.append(df)
        combined = pd.concat(agg_frames, ignore_index=True)
        combined = combined.groupby(['lat_round','lon_round','year']).agg({'mean_ndvi':'mean','observations':'sum'}).reset_index()
        out_path = OUT_DIR / "ndvi_year_grid.csv"
        combined.to_csv(out_path, index=False)
        print("Saved lat/lon grid-year NDVI to", out_path, "with rows:", len(combined))
        return

    # else fallback: year-only aggregation
    agg_frames = []
    for name, df in perfile_aggregates:
        if 'year' in df.columns:
            agg_frames.append(df[['year','mean_ndvi','observations']])
    combined = pd.concat(agg_frames, ignore_index=True)
    combined = combined.groupby(['year']).agg({'mean_ndvi':'mean','observations':'sum'}).reset_index()
    out_path = OUT_DIR / "ndvi_year.csv"
    combined.to_csv(out_path, index=False)
    print("Saved year-only NDVI to", out_path, "with rows:", len(combined))

if __name__ == "__main__":
    main()
