def explain_prediction(output: dict):
    """
    Converts system output into human-readable explanation.
    No ML, no rules — explanation only.
    """

    exp = []

    # Confidence explanation
    if output["top1_confidence"] == "HIGH":
        exp.append(
            "The top crop has a strong advantage over alternatives, indicating stable suitability."
        )
    elif output["top1_confidence"] == "MEDIUM":
        exp.append(
            "Multiple crops show similar suitability; the ranking reflects lower agronomic risk."
        )
    else:
        exp.append(
            "Crop suitability is uncertain; conservative crop choices are advised."
        )

    # NDVI explanation
    ndvi = output["ndvi_value"]
    if ndvi < 0.3:
        exp.append(
            "Vegetation health is currently low, which may indicate crop stress or low productivity."
        )
    else:
        exp.append(
            "Vegetation health is moderate, supporting common seasonal crops."
        )

    # Soil explanation
    exp.append(
        f"Soil condition is inferred as {output['soil_health']['overall'].lower()}, "
        "based on regional nutrient patterns."
    )

    # Zone explanation
    exp.append(
        f"The location falls under the {output['agro_climatic_zone']} agro-climatic zone, "
        "which influences crop suitability."
    )

    # Trust explanation
    trust = output["data_trust_level"]
    exp.append(
        f"The recommendation uses {trust['source'].lower()}-level data "
        f"within approximately {trust['radius_km']} km to avoid false precision."
    )

    # Safe mode
    if output["safe_mode"]:
        exp.append(
            "Safe Mode is enabled due to uncertainty, avoiding risky recommendations."
        )

    return exp
