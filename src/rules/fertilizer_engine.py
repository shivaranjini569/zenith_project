def recommend_fertilizer(crop, soil_behavior):
    """
    Conservative, rule-based fertilizer guidance
    """

    if soil_behavior == "MOISTURE_STRESSED":
        return {
            "status": "OK",
            "fertilizer": "Urea",
            "rate_kg_acre": 20,
            "logic": "Lower dose due to moisture stress"
        }

    if soil_behavior == "LOW_RETENTION":
        return {
            "status": "OK",
            "fertilizer": "DAP",
            "rate_kg_acre": 25,
            "logic": "Phosphorus support for weak retention soils"
        }

    if soil_behavior == "RESPONSIVE_BUT_DEPLETING":
        return {
            "status": "OK",
            "fertilizer": "Urea",
            "rate_kg_acre": 30,
            "logic": "Soil shows response but nutrients depleting"
        }

    return {
        "status": "OK",
        "fertilizer": "Urea",
        "rate_kg_acre": 25,
        "logic": "Maintenance dose only"
    }
