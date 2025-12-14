import pandas as pd
from pathlib import Path

NDVI = Path("data/external/ndvi/NDVI Prediction.csv")

print("Reading:", NDVI)
df = pd.read_csv(NDVI)
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nPreview:")
print(df.head())
