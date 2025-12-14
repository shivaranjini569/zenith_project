from rules.fertilizer_engine import fertilizer_rules
from rules.market_engine import get_market_info

fertilizer_info = fertilizer_rules(
    crop="Paddy",
    soil="Clayey",
    nitrogen=32
)

market_info = get_market_info("Paddy")
print("Fertilizer Recommendation:", fertilizer_info)