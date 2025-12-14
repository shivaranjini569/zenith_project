import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "processed" / "tn_crop_with_soil.csv"

df = pd.read_csv(FILE)

print("Before fix:")
print(df[df["Soil_Type"].isna()]["District"].value_counts())

# ---- FIX ALIASES ----
alias_map = {
    "TUTICORIN": "THOOTHUKUDI"
}

df["District"] = df["District"].replace(alias_map)

# Reload soil types
soil = pd.read_csv(ROOT / "data" / "external" / "tn_soil_types.csv")

# Merge again (safe re-merge)
df = df.drop(columns=["Soil_Type"]).merge(
    soil,
    on="District",
    how="left"
)

print("\nAfter fix:")
print(df[df["Soil_Type"].isna()]["District"].value_counts())

df.to_csv(FILE, index=False)
print("\n✅ Alias fixed and file saved:", FILE)
