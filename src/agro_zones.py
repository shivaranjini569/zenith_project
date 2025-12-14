# agro_zones.py
# Scientifically aligned with TNAU / ICAR agro-climatic zones

DISTRICT_TO_ZONE = {
    # Cauvery Delta Zone
    "thanjavur": "DELTA",
    "thiruvarur": "DELTA",
    "nagapattinam": "DELTA",
    "cuddalore": "DELTA",

    # Western Zone
    "coimbatore": "WEST",
    "erode": "WEST",
    "salem": "WEST",
    "tiruppur": "WEST",
    "namakkal": "WEST",

    # Southern Zone
    "madurai": "SOUTH",
    "virudhunagar": "SOUTH",
    "thoothukudi": "SOUTH",
    "ramanathapuram": "SOUTH",
    "kanniyakumari": "SOUTH",
    "dindigul": "SOUTH",

    # North Eastern Zone
    "vellore": "NE",
    "ranipet": "NE",
    "tiruvallur": "NE",
    "kanchipuram": "NE",
    "chengalpattu": "NE",
    "chennai": "NE",

    # Dry Zone
    "dharmapuri": "DRY",
    "krishnagiri": "DRY",
    "karur": "DRY",
    "ariyalur": "DRY",
    "perambalur": "DRY",
    "pudukkottai": "DRY",
    "kallakurichi": "DRY"
}

ZONE_CROP_TENDENCY = {
    "DELTA": ["paddy", "rice"],
    "WEST": ["maize", "cotton"],
    "SOUTH": ["millet", "pulse"],
    "NE": ["paddy", "groundnut"],
    "DRY": ["millet", "groundnut"]
}


def get_zone(district: str):
    if not district:
        return None
    return DISTRICT_TO_ZONE.get(district.strip().lower())


def get_zone_bias_crops(zone: str):
    return ZONE_CROP_TENDENCY.get(zone, [])
