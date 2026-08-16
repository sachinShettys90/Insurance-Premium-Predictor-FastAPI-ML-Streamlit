"""
Feature engineering for the insurance premium category model.

Transforms raw input fields (age, weight, height, income_lpa, smoker, city, occupation)
into the engineered features the model was trained on
(bmi, age_group, lifestyle_risk, city_tier, income_lpa, occupation).

Import compute_features() into your FastAPI app and call it on the raw
request payload before passing the result to model.predict().
"""

TIER_1_CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
TIER_2_CITIES = [
    "Lucknow", "Jaipur", "Chandigarh", "Indore", "Bhopal", "Nagpur",
    "Surat", "Ahmedabad", "Coimbatore", "Kochi", "Patna", "Vadodara",
]
# Any other city not listed above falls into tier 3.


def get_city_tier(city: str) -> int:
    if city in TIER_1_CITIES:
        return 1
    elif city in TIER_2_CITIES:
        return 2
    return 3


def get_age_group(age: int) -> str:
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    return "senior"


def get_bmi(weight: float, height: float) -> float:
    return round(weight / (height ** 2), 2)


def get_lifestyle_risk(smoker: bool, bmi: float) -> str:
    if smoker and bmi > 30:
        return "high"
    elif smoker or bmi > 27:
        return "medium"
    return "low"


def compute_features(age: int, weight: float, height: float, income_lpa: float,
                      smoker: bool, city: str, occupation: str) -> dict:
    """Takes raw patient/customer fields, returns the engineered feature dict
    in the exact shape the model expects."""
    bmi = get_bmi(weight, height)
    return {
        "bmi": bmi,
        "age_group": get_age_group(age),
        "lifestyle_risk": get_lifestyle_risk(smoker, bmi),
        "city_tier": get_city_tier(city),
        "income_lpa": income_lpa,
        "occupation": occupation,
    }
