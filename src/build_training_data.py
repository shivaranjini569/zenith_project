import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CORE = ROOT / "data" / "raw" / "data_core.csv"
CLIMATE = ROOT / "data" / "processed" / "global_climate_features.csv"
OUT = ROOT / "data" / "processed" / "training_with_climate.csv"

# load datasets
core = pd.read_csv(DATA_CORE)
climate = pd.read_csv(CLIMATE)

# Add climate values as fixed features to every row
for col in climate.columns:
    core[col] = climate[col].iloc[0]

core.to_csv(OUT, index=False)
print("Saved merged training dataset ->", OUT)
print(core.head())
print("Shape:", core.shape)
