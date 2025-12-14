def infer_soil_behavior(soil_health: dict, ndvi: float, zone: str):
    """
    Infers soil behavior (not soil type)
    """

    n = soil_health.get("nitrogen", "MEDIUM")
    p = soil_health.get("phosphorus", "MEDIUM")

    # Strong vegetation but low nutrients → responsive soil
    if ndvi > 0.45 and n == "LOW":
        return "RESPONSIVE_BUT_DEPLETING"

    # Low vegetation + medium nutrients → moisture or retention issue
    if ndvi < 0.30 and n == "MEDIUM":
        return "MOISTURE_STRESSED"

    # Dry zone behavior
    if zone == "DRY":
        return "LOW_RETENTION"

    # Delta zone behavior
    if zone == "DELTA":
        return "HIGH_RETENTION"

    return "BALANCED"
