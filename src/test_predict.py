from predict import predict_crop
from explain import explain_prediction

# ==================================================
# INPUT (Village / Area name as farmer would say)
# ==================================================
place_name = "ranipet"   # try: "thelungapatti", "karaikal", "salem"

result = predict_crop({"District": place_name})

# ==================================================
# HEADER
# ==================================================
print("\n" + "=" * 72)
print("🌾 SMART CROP ADVISORY REPORT")
print("=" * 72)

# ==================================================
# LOCATION & CONTEXT
# ==================================================
print("\n📍 LOCATION & CONTEXT")
print("-" * 72)

loc = result.get("location_resolution", {})
print(f"Village / Area Provided : {place_name.title()}")
print(f"Resolved District       : {loc.get('resolved_district', place_name).title()}")
print(f"Resolution Method       : {loc.get('method', 'DIRECT_INPUT')}")

print(f"Season                  : {result.get('season', 'N/A')}")
print(f"Agro-Climatic Zone       : {result.get('agro_climatic_zone', 'N/A')}")
print(f"Vegetation Index (NDVI)  : {result.get('ndvi_value', 'N/A')}")

trust = result.get("data_trust_level", {})
print(
    f"Data Confidence          : {trust.get('trust', 'N/A')} "
    f"(within ~{trust.get('radius_km', 'N/A')} km)"
)

# ==================================================
# CROP SUITABILITY
# ==================================================
print("\n🌱 CROP SUITABILITY (TOP 3)")
print("-" * 72)

crops = result.get("top3_crops", [])
scores = result.get("top3_probs", [])

for i, (crop, score) in enumerate(zip(crops, scores), start=1):
    print(f"{i}. {crop:<12} | Suitability Score : {score}")

print(f"\nOverall Confidence : {result.get('top1_confidence', 'N/A')}")
print(f"Safe Mode Enabled  : {'YES' if result.get('safe_mode') else 'NO'}")

# ==================================================
# SOIL INTELLIGENCE
# ==================================================
soil = result.get("soil_health", {})

print("\n🧪 SOIL HEALTH (INFERRED, NOT ASSUMED)")
print("-" * 72)
print(f"Nitrogen   : {soil.get('nitrogen', 'N/A')}")
print(f"Phosphorus : {soil.get('phosphorus', 'N/A')}")
print(f"Potassium  : {soil.get('potassium', 'N/A')}")
print(f"Overall    : {soil.get('overall', 'N/A')}")
print(f"Soil Behavior Pattern : {result.get('soil_behavior', 'BALANCED')}")

print(
    "⚠️ Note: Soil health is inferred using nearby historical soil statistics\n"
    "        and vegetation signals, not assumed or manually entered."
)

# ==================================================
# FERTILIZER GUIDANCE
# ==================================================
fert = result.get("fertilizer_guidance", {})

print("\n🌾 FERTILIZER GUIDANCE (RULE-BASED, OPTIONAL)")
print("-" * 72)

if fert.get("status") == "OK":
    print(f"Suggested Fertilizer : {fert.get('fertilizer')}")
    print(f"Suggested Quantity   : {fert.get('rate_kg_acre')} kg / acre")
    print(f"Reason               : {fert.get('logic')}")
    print("⚠️ Note               : Guidance only. Not a mandatory instruction.")
else:
    print("No aggressive fertilizer recommendation generated.")
    print("System prefers conservative nutrient management under uncertainty.")

# ==================================================
# MARKET AWARENESS
# ==================================================
market = result.get("market_awareness", {})

print("\n📈 MARKET AWARENESS (INFORMATION ONLY)")
print("-" * 72)

print(
    "⚠️ IMPORTANT NOTE FOR FARMERS:\n"
    "The market shown below is NOT your village market.\n"
    "It is a nearby high-volume reference market inferred\n"
    "using crop trade patterns and agro-climatic similarity.\n"
    "This is for awareness only, not a selling recommendation."
)

print(f"\nReference Market : {market.get('market', 'Regional Reference Market')}")
print(f"Market Trend     : {market.get('trend', 'Stable')}")
print(
    f"Explanation      : {market.get('note', 'Shown only to understand general trends.')}"
)

# ==================================================
# EXPLAINABLE AI (XAI)
# ==================================================
print("\n🔍 WHY THIS WAS RECOMMENDED (EXPLAINABLE AI)")
print("-" * 72)

for i, explanation in enumerate(explain_prediction(result), start=1):
    print(f"{i}. {explanation}")

# ==================================================
# FOOTER
# ==================================================
print("\n" + "=" * 72)
print("✅ END OF ADVISORY REPORT")
print("=" * 72)



# from predict import predict_crop
# from explain import explain_prediction

# # --------------------------
# # Run prediction
# # --------------------------
# district_name = "ranipet"
# result = predict_crop({"District": district_name})

# # ==================================================
# # HEADER
# # ==================================================
# print("\n" + "=" * 70)
# print("🌾 SMART CROP ADVISORY REPORT")
# print("=" * 70)

# # ==================================================
# # LOCATION & CONTEXT
# # ==================================================
# print("\n📍 LOCATION & CONTEXT")
# print("-" * 70)
# print(f"Village / Area Provided : {district_name.title()}")
# print(f"Season                  : {result.get('season', 'N/A')}")
# print(f"Agro-Climatic Zone       : {result.get('agro_climatic_zone', 'N/A')}")
# print(f"Vegetation Index (NDVI)  : {result.get('ndvi_value', 'N/A')}")

# trust = result.get("data_trust_level", {})
# print(
#     f"Data Confidence          : {trust.get('trust', 'N/A')} "
#     f"(within ~{trust.get('radius_km', 'N/A')} km)"
# )

# # ==================================================
# # CROP SUITABILITY
# # ==================================================
# print("\n🌱 CROP SUITABILITY (TOP 3)")
# print("-" * 70)

# for i, (crop, score) in enumerate(
#     zip(result.get("top3_crops", []), result.get("top3_probs", [])), start=1
# ):
#     print(f"{i}. {crop:<12} | Suitability Score : {score}")

# print(f"\nOverall Confidence : {result.get('top1_confidence', 'N/A')}")
# print(f"Safe Mode Enabled  : {'YES' if result.get('safe_mode') else 'NO'}")

# # ==================================================
# # SOIL INTELLIGENCE
# # ==================================================
# soil = result.get("soil_health", {})

# print("\n🧪 SOIL HEALTH (INFERRED, NOT ASSUMED)")
# print("-" * 70)
# print(f"Nitrogen   : {soil.get('nitrogen', 'N/A')}")
# print(f"Phosphorus : {soil.get('phosphorus', 'N/A')}")
# print(f"Potassium  : {soil.get('potassium', 'N/A')}")
# print(f"Overall    : {soil.get('overall', 'N/A')}")
# print(f"Soil Behavior Pattern : {result.get('soil_behavior', 'BALANCED')}")

# # ==================================================
# # FERTILIZER GUIDANCE
# # ==================================================
# fert = result.get("fertilizer_guidance", {})

# print("\n🌾 FERTILIZER GUIDANCE (RULE-BASED, OPTIONAL)")
# print("-" * 70)

# if fert.get("status") == "OK":
#     print(f"Suggested Fertilizer : {fert.get('fertilizer')}")
#     print(f"Suggested Quantity   : {fert.get('rate_kg_acre')} kg / acre")
#     print(f"Reason               : {fert.get('logic')}")
#     print("⚠️ Note               : This is guidance only, not a mandatory instruction.")
# else:
#     print("No aggressive fertilizer recommendation generated.")
#     print("System favors conservative nutrient management under uncertainty.")

# # ==================================================
# # MARKET AWARENESS
# # ==================================================
# market = result.get("market_awareness", {})

# print("\n📈 MARKET AWARENESS (INFORMATION ONLY)")
# print("-" * 70)
# print("⚠️ IMPORTANT NOTE FOR FARMERS:")
# print(
#     "The market shown below is NOT your village market.\n"
#     "It is a nearby high-volume reference market inferred\n"
#     "using crop trade patterns and agro-climatic similarity."
# )

# print(f"\nReference Market : {market.get('market', 'Regional Market')}")
# print(f"Market Trend     : {market.get('trend', 'Stable')}")
# print(
#     f"Explanation      : {market.get('note', 'Shown for awareness only, not pricing decisions.')}"
# )

# # ==================================================
# # EXPLANATION (XAI)
# # ==================================================
# print("\n🔍 WHY THIS WAS RECOMMENDED (EXPLAINABLE AI)")
# print("-" * 70)

# for i, e in enumerate(explain_prediction(result), start=1):
#     print(f"{i}. {e}")

# # ==================================================
# # FOOTER
# # ==================================================
# print("\n" + "=" * 70)
# print("✅ END OF ADVISORY REPORT")
# print("=" * 70)
