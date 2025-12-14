import joblib
import pandas as pd
import numpy as np
from catboost import Pool

from soil_health import estimate_soil_health
from agro_zones import get_zone
from rules.fertilizer_engine import recommend_fertilizer
from rules.market_engine import get_market_info
from soil_behavior import infer_soil_behavior
from location_resolver import resolve_location

# ==================================================
# LOAD MODELS & DATA (ONCE)
# ==================================================
MODEL_PATH = "models/catboost_tn_top3.joblib"
SCHEMA_PATH = "models/feature_schema_catboost.joblib"
DATA_PATH = "data/processed/tn_ml_ndvi_only.csv"

model = joblib.load(MODEL_PATH)
schema = joblib.load(SCHEMA_PATH)

features = schema["features"]
cat_features = schema["cat_features"]

data = pd.read_csv(DATA_PATH)
data["District_norm"] = data["District"].str.strip().str.lower()

# ==================================================
# HELPERS
# ==================================================
def infer_season():
    m = pd.Timestamp.now().month
    if m in [6,7,8,9]:
        return "Kharif"
    elif m in [10,11,12,1]:
        return "Rabi"
    return "Summer"

def confidence_band_relative(p1, p2):
    if p1 - p2 >= 0.25:
        return "HIGH"
    elif p1 - p2 >= 0.12:
        return "MEDIUM"
    return "LOW"

def diversify_ranking(crops, probs, zone):
    """
    Keeps ML honest but avoids monoculture dominance
    """
    if probs[0] - probs[1] >= 0.15:
        return crops, probs

    zone_priority = {
        "DELTA": ["paddy", "rice"],
        "DRY": ["millet", "groundnut"],
        "SOUTH": ["millet", "pulse"],
        "NE": ["paddy", "groundnut"],
        "WEST": ["cotton", "maize"]
    }.get(zone, [])

    ranked = list(zip(crops, probs))
    ranked.sort(key=lambda x: (x[0].lower() not in zone_priority, -x[1]))

    return [c for c,_ in ranked], [p for _,p in ranked]

# ==================================================
# MAIN PREDICTION FUNCTION
# ==================================================
def predict_crop(farmer_input: dict):

    # --------------------------
    # INPUT NORMALIZATION
    # --------------------------
    input_place = farmer_input["District"]
    district, location_mode = resolve_location(input_place)

    season = infer_season()

    district_rows = data[data["District_norm"] == district]
    fallback_level = "DISTRICT"

    if district_rows.empty:
        fallback_level = "NEAREST_DISTRICT"
        district_rows = data.copy()

    # --------------------------
    # NDVI SELECTION
    # --------------------------
    ndvi_col = (
        "ndvi_kharif_mean" if season == "Kharif"
        else "ndvi_rabi_mean" if season == "Rabi"
        else "ndvi_mean"
    )
    ndvi_value = float(district_rows[ndvi_col].mean())

    # --------------------------
    # FEATURE VECTOR
    # --------------------------
    row = {}
    for f in features:
        if f in cat_features:
            row[f] = (
                district_rows[f].mode().iloc[0]
                if f in district_rows.columns and not district_rows[f].dropna().empty
                else ""
            )
        else:
            row[f] = (
                float(district_rows[f].mean())
                if f in district_rows.columns and not district_rows[f].dropna().empty
                else 0.0
            )

    row["District"] = farmer_input["District"]
    row["Season"] = season
    row[ndvi_col] = ndvi_value

    X = pd.DataFrame([row])
    pool = Pool(X, cat_features=cat_features)

    # --------------------------
    # ML INFERENCE
    # --------------------------
    probs = model.predict_proba(pool)[0]
    classes = model.classes_

    idx = np.argsort(probs)[::-1][:3]
    top3_crops = classes[idx].tolist()
    top3_probs = probs[idx].tolist()

    # --------------------------
    # AGRO-CLIMATIC INTELLIGENCE
    # --------------------------
    zone = get_zone(district)
    top3_crops, top3_probs = diversify_ranking(top3_crops, top3_probs, zone)

    # --------------------------
    # CONFIDENCE & SAFE MODE
    # --------------------------
    top1_conf = confidence_band_relative(top3_probs[0], top3_probs[1])
    safe_mode = bool(top1_conf == "LOW" or ndvi_value < 0.28)

    # --------------------------
    # SOIL INTELLIGENCE (NO SOIL TYPE ASSUMED)
    # --------------------------
    soil_health = estimate_soil_health(district_rows)
    soil_behavior = infer_soil_behavior(
        soil_health=soil_health,
        ndvi=ndvi_value,
        zone=zone
    )

    # --------------------------
    # FERTILIZER (RULE-BASED, SAFE)
    # --------------------------
    fertilizer = recommend_fertilizer(
        crop=top3_crops[0],
        soil_behavior=soil_behavior
    )

    # --------------------------
    # MARKET AWARENESS (ZONE-SPECIFIC)
    # --------------------------
    market = get_market_info(
        crop=top3_crops[0],
        zone=zone
    )

    # --------------------------
    # TRUST LOGIC
    # --------------------------
    if fallback_level == "DISTRICT":
        trust, radius = "MEDIUM", 30
    else:
        trust, radius = "LOW", 60

    # --------------------------
    # FINAL OUTPUT (SYSTEM CONTRACT)
    # --------------------------
    return {
        "top3_crops": top3_crops,
        "top3_probs": [round(p, 3) for p in top3_probs],
        "top1_confidence": top1_conf,
        "safe_mode": safe_mode,

        "soil_health": soil_health,
        "soil_behavior": soil_behavior,

        "fertilizer_guidance": fertilizer,
        "market_awareness": market,

        "fallback_level": fallback_level,
        "season": season,
        "ndvi_value": round(ndvi_value, 3),
        "agro_climatic_zone": zone,

        "data_trust_level": {
            "source": fallback_level,
            "trust": trust,
            "radius_km": radius
        },

        "decision_reasoning": {
            "ml_role": "Primary crop suitability ranking",
            "zone_role": f"Risk-aware adjustment using {zone} agro-climatic zone",
            "soil_role": "Soil behavior inferred from nutrients and vegetation",
            "fertilizer_role": "Conservative agronomy rules (not ML)",
            "market_role": "Awareness only, no price prediction",
            "fallback_role": f"{fallback_level} data used to avoid false precision"
        },
        "location_resolution": {
            "input": input_place,
            "resolved_district": district,
            "method": location_mode
        }
    }
