import joblib
import pandas as pd
import numpy as np
from catboost import Pool

# ==================================================
# 1. Load model & schema
# ==================================================
model = joblib.load("models/catboost_tn_top3.joblib")
feature_schema = joblib.load("models/feature_schema_catboost.joblib")

features = feature_schema["features"]
cat_features = feature_schema["cat_features"]

# ==================================================
# 2. Load dataset
# ==================================================
data = pd.read_csv("data/processed/tn_ml_ndvi_only.csv")

# ==================================================
# 3. Farmer minimal input (simulate NFC / voice)
# ==================================================
farmer_input = {
    "District": "ranipet"   # TRY any district safely
}

# ==================================================
# 4. Helper functions
# ==================================================
def normalize_text(x):
    return str(x).strip().lower()

def infer_season():
    month = pd.Timestamp.now().month
    if month in [6, 7, 8, 9]:
        return "Kharif"
    elif month in [10, 11, 12, 1]:
        return "Rabi"
    else:
        return "Summer"

def confidence_band_relative(p1, p2):
    margin = p1 - p2
    if margin >= 0.25:
        return "HIGH"
    elif margin >= 0.12:
        return "MEDIUM"
    else:
        return "LOW"

# ==================================================
# 5. Normalize district & infer season
# ==================================================
data["District_norm"] = data["District"].apply(normalize_text)
farmer_district = normalize_text(farmer_input["District"])
season = infer_season()

district_rows = data[data["District_norm"] == farmer_district]

# ==================================================
# 6. Fallback handling (CRITICAL)
# ==================================================
fallback_level = "DISTRICT"

if district_rows.empty:
    fallback_level = "STATE"
    print(
        f"\n⚠️ District '{farmer_input['District']}' not found in dataset."
        "\nUsing Tamil Nadu state-level statistics."
    )
    if "State_clean" in data.columns:
        district_rows = data[data["State_clean"].str.lower() == "tamil nadu"]

if district_rows.empty:
    fallback_level = "GLOBAL"
    print("\n⚠️ State-level data unavailable. Using global dataset averages.")
    district_rows = data.copy()

# ==================================================
# 7. Select NDVI column (season-aware)
# ==================================================
if season == "Kharif":
    ndvi_col = "ndvi_kharif_mean"
elif season == "Rabi":
    ndvi_col = "ndvi_rabi_mean"
else:
    ndvi_col = "ndvi_mean"

if ndvi_col in district_rows.columns and not district_rows[ndvi_col].dropna().empty:
    ndvi_value = float(district_rows[ndvi_col].mean())
else:
    ndvi_value = float(data[ndvi_col].mean())

# ==================================================
# 8. Build FULL feature row (SAFE + CONTEXTUAL)
# ==================================================
row = {}

for f in features:
    if f in cat_features:
        if f in district_rows.columns and not district_rows[f].dropna().empty:
            row[f] = district_rows[f].mode().iloc[0]
        else:
            row[f] = ""
    else:
        if f in district_rows.columns and not district_rows[f].dropna().empty:
            row[f] = float(district_rows[f].mean())
        else:
            row[f] = 0.0

# Override with system-derived values
row["District"] = farmer_input["District"]
row["Season"] = season
row[ndvi_col] = ndvi_value

X_sample = pd.DataFrame([row])

# ==================================================
# 9. Create CatBoost Pool
# ==================================================
pool = Pool(X_sample, cat_features=cat_features)

# ==================================================
# 10. Predict
# ==================================================
probs = model.predict_proba(pool)[0]
classes = model.classes_

top3_idx = np.argsort(probs)[::-1][:3]
top3_crops = classes[top3_idx]
top3_probs = probs[top3_idx]

# ==================================================
# 11. District affinity (gentle re-ranking)
# ==================================================
DISTRICT_AFFINITY = {
    "coimbatore": ["maize", "cotton"],
    "erode": ["turmeric", "groundnut"],
    "madurai": ["millet", "pulse"],
    "vellore": ["groundnut", "vegetable"],
    "karaikal": ["paddy", "rice"]
}

affinity = DISTRICT_AFFINITY.get(farmer_district, [])

adjusted_scores = []
for crop, prob in zip(top3_crops, top3_probs):
    if any(a in crop.lower() for a in affinity):
        adjusted_scores.append(prob + 0.05)
    else:
        adjusted_scores.append(prob)

sorted_idx = np.argsort(adjusted_scores)[::-1]
top3_crops = top3_crops[sorted_idx]
top3_probs = top3_probs[sorted_idx]

# ==================================================
# 12. Relative confidence
# ==================================================
top3_conf = [
    confidence_band_relative(top3_probs[0], top3_probs[1]),
    "LOW",
    "LOW"
]

# ==================================================
# 13. Smart SAFE MODE
# ==================================================
safe_mode = False
if top3_conf[0] == "LOW" and ndvi_value < 0.30:
    safe_mode = True

# ==================================================
# 14. XAI (global importance)
# ==================================================
importances = model.get_feature_importance()
fi = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

top_features = fi.head(3)["feature"].tolist()

# ==================================================
# 15. OUTPUT
# ==================================================
def explain_prediction():
    """
    Returns XAI explanation as structured data
    """

    explanation = {
        "top_factors": top_features,
        "safe_mode": safe_mode,
        "fallback_level": fallback_level,
        "system_note": (
            f"Decision used {fallback_level}-level agronomic statistics, "
            "satellite vegetation trends, seasonal patterns, and "
            "district affinity logic."
        )
    }

    return explanation

# print("\n🌾 Crop Suitability Recommendations (Top-3):\n")

# for i in range(3):
#     print(f"{i+1}️⃣ {top3_crops[i]} | Confidence: {top3_conf[i]}")

# print("\n🔍 Key Factors Influencing This Recommendation:")
# for f in top_features:
#     print(f"- {f}")

# if safe_mode:
#     print(
#         "\n⚠️ SAFE MODE ACTIVATED\n"
#         "Vegetation or historical signals show uncertainty.\n"
#         "Low-risk or traditional crops are advised.\n"
#         "Consult local agricultural officers before major crop changes."
#     )

# print(
#     f"\nℹ️ System Note:\n"
#     f"Decision used {fallback_level}-level agronomic statistics, "
#     "satellite-derived vegetation trends, seasonal patterns, and "
#     "district agronomic affinity. Relative confidence and Safe Mode "
#     "prevent risky recommendations."
# )

print("\n✅ Explanation generated successfully")


#['ARIYALUR', 'CHENGALPATTU', 'CHENNAI', 'COIMBATORE', 'CUDDALORE', 'DHARMAPURI', 'DINDIGUL', 
#'ERODE', 'KALLAKURICHI', 'KANCHIPURAM', 'KARAIKAL',
#'KANNIYAKUMARI', 'KARUR', 'KRISHNAGIRI', 'MADURAI', 'NAGAPATTINAM', 'NAMAKKAL', 'PERAMBALUR', 'PUDUKKOTTAI', 'RAMANATHAPURAM', 'RANIPET']