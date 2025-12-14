import pandas as pd

print("Loading crop + soil data...")
crop = pd.read_csv("data/processed/tn_crop_with_soil.csv")

print("Loading NDVI features...")
ndvi = pd.read_csv("data/processed/tn_ndvi_features.csv")

# ✅ FIX YEAR FORMAT
crop["Year"] = crop["Year"].astype(str).str[:4].astype(int)
ndvi["Year"] = ndvi["Year"].astype(int)

print("Merging crop + soil + NDVI...")
merged = crop.merge(ndvi, on="Year", how="left")

# Save final ML dataset
merged.to_csv("data/processed/tn_final_ml_dataset.csv", index=False)

print("✅ Final ML dataset saved")
print("Shape:", merged.shape)
print(merged.head())
