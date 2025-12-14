import requests
import pandas as pd
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CENTROIDS = ROOT / "data" / "processed" / "tn_district_centroids.csv"
OUT = ROOT / "data" / "processed" / "tn_soil_features.csv"

print("Loading district centroids...")
df = pd.read_csv(CENTROIDS)

# Soil properties we want (topsoil equivalent)
PROPERTIES = [
    "clay", "sand", "silt",
    "soc", "bdod", "cec", "phh2o"
]

BASE_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

rows = []

for _, r in df.iterrows():
    district = r["district"]
    lat = r["lat"]
    lon = r["lon"]

    print(f"Fetching soil for {district}...")

    params = {
        "lat": lat,
        "lon": lon,
        "property": ",".join(PROPERTIES)
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    record = {"district": district}

    for p in PROPERTIES:
        try:
            # depth-averaged value (most stable)
            record[p] = data["properties"][p]["mean"]
        except:
            record[p] = None

    rows.append(record)

    # polite delay to avoid rate limits
    time.sleep(1)

out_df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT, index=False)

print("✅ Saved soil features to:", OUT)
print(out_df.head())
