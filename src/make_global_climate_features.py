import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NDVI = ROOT / "data/external/ndvi/NDVI Prediction.csv"
OUT = ROOT / "data/processed/global_climate_features.csv"

df = pd.read_csv(NDVI)

summary = {
    "NDVI_mean": df["NDVI"].mean(),
    "NDVI_std": df["NDVI"].std(),
    "LST_mean": df["LST"].mean(),
    "PRECIP_mean": df["PRECIPITATION"].mean(),
    "ETo_mean": df["ETo"].mean()
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUT, index=False)

print("Saved:", OUT)
print(summary_df)
