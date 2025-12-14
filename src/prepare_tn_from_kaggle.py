import pandas as pd
from pathlib import Path

# ROOT = zenith_project folder
ROOT = Path(__file__).resolve().parents[1]

# Correct Kaggle file name (from your folder)
KAG = ROOT / "data" / "external" / "india_prod" / "India Agriculture Crop Production.csv"

# Output file
OUT = ROOT / "data" / "processed" / "india_prod_tn.csv"

print("Loading:", KAG)

df = pd.read_csv(KAG, low_memory=False)
print("Total rows loaded:", df.shape)

# Normalize State column
df["State_clean"] = df["State"].astype(str).str.strip().str.lower()

# Filter only Tamil Nadu
tn = df[df["State_clean"] == "tamil nadu"]

print("Tamil Nadu rows:", tn.shape)

# Save output
OUT.parent.mkdir(parents=True, exist_ok=True)
tn.to_csv(OUT, index=False)
print("Saved Tamil Nadu dataset to:", OUT)
