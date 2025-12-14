def band_value(x, low, high):
    """
    Converts numeric value into LOW / MEDIUM / HIGH
    """
    if x < low:
        return "LOW"
    elif x < high:
        return "MEDIUM"
    else:
        return "HIGH"


def estimate_soil_health(district_rows):
    """
    Uses nearby soil statistics (safe approximation)
    """

    # Default values if columns missing
    nitrogen = district_rows["Nitrogen"].mean() if "Nitrogen" in district_rows.columns else 40
    phosphorus = district_rows["Phosphorus"].mean() if "Phosphorus" in district_rows.columns else 30
    potassium = district_rows["Potassium"].mean() if "Potassium" in district_rows.columns else 30

    soil_health = {
        "nitrogen": band_value(nitrogen, 30, 60),
        "phosphorus": band_value(phosphorus, 20, 50),
        "potassium": band_value(potassium, 20, 50)
    }

    # Overall soil health
    if "LOW" in soil_health.values():
        soil_health["overall"] = "POOR"
    elif "MEDIUM" in soil_health.values():
        soil_health["overall"] = "MODERATE"
    else:
        soil_health["overall"] = "GOOD"

    return soil_health

def estimated_nitrogen_value(soil_health: dict):
    """
    Converts soil health band into safe numeric nitrogen estimate
    (used ONLY for fertilizer rule matching)
    """

    band = soil_health.get("nitrogen", "MEDIUM")

    if band == "LOW":
        return 25
    elif band == "MEDIUM":
        return 30
    else:  # HIGH
        return 40
