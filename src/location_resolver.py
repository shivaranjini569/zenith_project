# location_resolver.py
import pandas as pd

VILLAGE_MAP_PATH = "data/village_to_district.csv"

try:
    village_df = pd.read_csv(VILLAGE_MAP_PATH)
    village_df["village_norm"] = village_df["village"].str.lower().str.strip()
except:
    village_df = pd.DataFrame()

def resolve_location(place_name: str):
    """
    Resolves village/town name to nearest known district.
    Returns: (district, resolution_type)
    """
    name = place_name.lower().strip()

    # 1️⃣ Exact village match
    match = village_df[village_df["village_norm"] == name]
    if not match.empty:
        return match.iloc[0]["district"].lower(), "VILLAGE_MATCH"

    # 2️⃣ If user already gave district
    return name, "DISTRICT_ASSUMED"
    # 3️⃣ Fallback