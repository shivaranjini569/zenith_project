import pandas as pd

ndvi = pd.read_csv("data/processed/tn_ndvi_clean.csv")

kharif = [6,7,8,9,10]
rabi = [10,11,12,1,2,3]

features = []

for year, g in ndvi.groupby("Year"):
    row = {
        "Year": year,
        "ndvi_mean": g["NDVI"].mean(),
        "ndvi_max": g["NDVI"].max(),
        "ndvi_std": g["NDVI"].std(),
        "ndvi_kharif_mean": g[g["Month"].isin(kharif)]["NDVI"].mean(),
        "ndvi_rabi_mean": g[g["Month"].isin(rabi)]["NDVI"].mean(),
    }
    features.append(row)

df = pd.DataFrame(features)
df.to_csv("data/processed/tn_ndvi_features.csv", index=False)

print("✅ NDVI features created")
print(df)