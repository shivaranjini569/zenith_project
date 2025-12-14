import pandas as pd

ndvi = pd.read_csv("data/external/ndvi_monthly_Tamil_Nadu_2018_2023.csv")

# Keep only useful columns
ndvi = ndvi[["year", "month", "mean_ndvi"]]

# Rename for clarity
ndvi = ndvi.rename(columns={
    "year": "Year",
    "month": "Month",
    "mean_ndvi": "NDVI"
})

# Ensure numeric
ndvi["Year"] = ndvi["Year"].astype(int)
ndvi["Month"] = ndvi["Month"].astype(int)
ndvi["NDVI"] = ndvi["NDVI"].astype(float)

ndvi.to_csv("data/processed/tn_ndvi_clean.csv", index=False)

print("✅ Clean NDVI saved")
print(ndvi.head())

