# src/soilgrids_fetch.py
# Query ISRIC SoilGrids REST for a set of soil properties at district centroids.
# Produces: data/processed/tn_soil_features.csv

import requests
import pandas as pd
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CENT = ROOT / "data" / "processed" / "tn_district_centroids.csv"
OUT = ROOT / "data" / "processed" / "tn_soil_features.csv"

centroids = pd.read_csv(CENT)
OUT.parent.mkdir(parents=True, exist_ok=True)

properties = ["clay","silt","sand","soc","bdod","cec","phh2o"]  # adjust if you want more
prop_str = ",".join(properties)

rows = []
base = "https://rest.isric.org/soilgrids/v2.0/properties/query"

for _, r in centroids.iterrows():
    district = r["District"]
    lat = r["lat"]
    lon = r["lon"]
    if pd.isna(lat) or pd.isna(lon):
        print("Skipping (no coords):", district)
        rows.append({"District": district, **{p: None for p in properties}})
        continue

    # build request
    params = {
        "lat": float(lat),
        "lon": float(lon),
        "property": prop_str
        # optionally you can request &depth=0-5 etc but ISRIC returns depths; we'll pick top layer below
    }

    success = False
    for attempt in range(1, 5):
        try:
            resp = requests.get(base, params=params, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"Status {resp.status_code}")
            data = resp.json()
            soil = {"District": district}
            # data['properties'][prop]['depths'] is usually a list of depth bands; pick first available mean value
            for p in properties:
                try:
                    vals = data["properties"][p]["values"]
                    # values often include depths array; handle variations
                    if isinstance(vals, list) and len(vals) > 0:
                        # find first numeric mean
                        found = None
                        for v in vals:
                            if isinstance(v, dict) and "value" in v:
                                found = v["value"]
                                break
                            if isinstance(v, dict) and "mean" in v:
                                found = v["mean"]
                                break
                        soil[p] = found
                    else:
                        # fallback: try depths field
                        depths = data["properties"][p].get("depths")
                        if depths and isinstance(depths, list) and len(depths) > 0:
                            soil[p] = depths[0].get("values", {}).get("mean", None)
                        else:
                            soil[p] = None
                except Exception:
                    soil[p] = None
            rows.append(soil)
            print("OK:", district)
            success = True
            break
        except Exception as e:
            print(f"Attempt {attempt} failed for {district}: {e}")
            time.sleep(2 * attempt)
    if not success:
        print("Failed all attempts:", district)
        rows.append({"District": district, **{p: None for p in properties}})
    time.sleep(1)  # be polite

pd.DataFrame(rows).to_csv(OUT, index=False)
print("Saved soil features to:", OUT)
