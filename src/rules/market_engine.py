import pandas as pd

MARKET_DATA_PATH = "data/market_reference.csv"

# Fallback mapping by agro-climatic zone
ZONE_REFERENCE_MARKETS = {
    "WEST": "Erode",
    "DELTA": "Thanjavur",
    "SOUTH": "Madurai",
    "NE": "Chennai",
    "DRY": "Salem"
}

def get_market_info(crop: str, zone: str):
    """
    Returns a nearby high-volume reference market.
    Never returns NO_DATA.
    """

    df = pd.read_csv(MARKET_DATA_PATH)
    crop = crop.lower()

    # 1️⃣ Try crop-specific market
    crop_match = df[df["crop"].str.lower() == crop]
    if not crop_match.empty:
        row = crop_match.iloc[0]
        return {
            "status": "OK",
            "market": row["market"],
            "trend": row["trend"],
            "note": (
                "Reference market based on crop trade volume. "
                "Shown for awareness only, not local pricing."
            )
        }

    # 2️⃣ Zone-based fallback market
    reference_market = ZONE_REFERENCE_MARKETS.get(zone, "State Market")

    return {
        "status": "OK",
        "market": reference_market,
        "trend": "Stable",
        "note": (
            f"Reference market inferred using {zone} agro-climatic zone. "
            "Used only to show general trend, not local price."
        )
    }
