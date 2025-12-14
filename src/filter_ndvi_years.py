import pandas as pd

df = pd.read_csv("data/processed/tn_final_ml_dataset.csv")

print("Before filtering:", df.shape)

# Keep only years where NDVI exists
df = df[df["ndvi_mean"].notna()]

print("After filtering:", df.shape)
print("Years used:", sorted(df["Year"].unique()))

df.to_csv("data/processed/tn_ml_ndvi_only.csv", index=False)

print("✅ NDVI-aligned ML dataset saved")
print(df.head())