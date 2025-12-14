import requests
import pandas as pd
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CENTROIDS = ROOT / "data" / "processed" / "tn_district_centroids.csv"
OUT = ROOT / "data" / "processed" / "tn_soil_features.csv"

print("Loading district centroids...")
df = pd.read_csv(CENTROIDS)

# ---- AUTO-DETECT COLUMN NAMES (NO ASSUMPTIONS) ----
cols = {c.lower(): c for c in df.columns}

district_col = cols.get("district") or cols.get("dtname") or cols.get("name")
lat_col = cols.get("lat") or cols.get("latitude")
lon_col = cols.get("lon") or cols.get("longitude")

if not all([district_col, lat_col, lon_col]):
    raise ValueError(f"Could not detect columns. Found: {df.columns.tolist()}")

print("Using columns:", district_col, lat_col, lon_col)

# Soil properties (ISRIC)
PROPERTIES = ["clay", "sand", "silt", "soc", "bdod", "cec", "phh2o"]
BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

rows = []

for _, r in df.iterrows():
    district = r[district_col]
    lat = r[lat_col]
    lon = r[lon_col]

    print(f"Fetching soil for {district}...")

    params = {
        "lat": lat,
        "lon": lon,
        "property": ",".join(PROPERTIES)
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    record = {"district": district}

    for p in PROPERTIES:
        try:
            record[p] = data["properties"][p]["mean"]
        except Exception:
            record[p] = None

    rows.append(record)
    time.sleep(1)  # polite delay

out_df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT, index=False)

print("✅ Soil features saved to:", OUT)
print(out_df.head())
