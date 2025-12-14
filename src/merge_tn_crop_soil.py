import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CROP_FILE = ROOT / "data" / "processed" / "india_prod_tn.csv"
SOIL_FILE = ROOT / "data" / "external" / "tn_soil_types.csv"
OUT_FILE  = ROOT / "data" / "processed" / "tn_crop_with_soil.csv"

print("Loading Tamil Nadu crop data...")
crop_df = pd.read_csv(CROP_FILE)

print("Loading Tamil Nadu soil types...")
soil_df = pd.read_csv(SOIL_FILE)

# Normalize district names (VERY IMPORTANT)
crop_df["District"] = crop_df["District"].str.upper().str.strip()
soil_df["District"] = soil_df["District"].str.upper().str.strip()

print("Merging crop + soil data...")
merged_df = crop_df.merge(
    soil_df,
    on="District",
    how="left"
)

# Check missing soil values
missing = merged_df["Soil_Type"].isna().sum()
print(f"Missing soil entries: {missing}")

# Save output
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
merged_df.to_csv(OUT_FILE, index=False)

print("✅ Saved merged dataset to:")
print(OUT_FILE)

print("\nSample rows:")
print(merged_df[["District", "Crop", "Soil_Type"]].head())
print("\nShape:", merged_df.shape)