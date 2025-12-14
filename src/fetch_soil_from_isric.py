import requests
import pandas as pd
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CENTROIDS = ROOT / "data" / "processed" / "tn_district_centroids.csv"
OUT = ROOT / "data" / "processed" / "tn_soil_features.csv"

print("Loading district centroids...")
df = pd.read_csv(CENTROIDS)

# ✅ FIXED column names (based on your actual CSV)
district_col = "District"
lat_col = "lat"
lon_col = "lon"

print("Using columns:", district_col, lat_col, lon_col)

# SoilGrids properties (stable + meaningful)
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

    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"⚠️ Failed for {district}: {e}")
        continue

    record = {"District": district}

    for p in PROPERTIES:
        record[p] = data.get("properties", {}).get(p, {}).get("mean")

    rows.append(record)
    time.sleep(1)  # polite API usage

out_df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT, index=False)

print("✅ Soil features saved to:", OUT)
print(out_df.head())
