"""
Generates a synthetic 'raw' dataset matching the original data schema
(age, weight, height, income_lpa, smoker, city, occupation), then derives
a plausible insurance_premium_category label using a rule-based risk score
with random noise, simulating how a real insurer's premium tiers might
correlate with age, BMI, smoking status, and income.
"""

import numpy as np
import pandas as pd
from MLModel.feature_engineering import get_bmi

np.random.seed(42)

N = 3000

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune",
    "Lucknow", "Jaipur", "Chandigarh", "Indore", "Bhopal", "Nagpur",
    "Surat", "Ahmedabad", "Coimbatore", "Kochi", "Patna", "Vadodara",
    "Ranchi", "Guwahati", "Shimla", "Mysore", "Nashik",
]

OCCUPATIONS = [
    "retired", "unemployed", "business_owner", "government_job",
    "private_job", "freelancer", "student",
]

age = np.random.randint(18, 75, size=N)
height = np.round(np.random.normal(1.65, 0.09, size=N).clip(1.4, 2.0), 2)
weight = np.round(np.random.normal(70, 15, size=N).clip(40, 140), 1)
income_lpa = np.round(np.abs(np.random.normal(8, 6, size=N)).clip(0.5, 60), 2)
smoker = np.random.choice([True, False], size=N, p=[0.25, 0.75])
city = np.random.choice(CITIES, size=N)
occupation = np.random.choice(OCCUPATIONS, size=N)

df = pd.DataFrame({
    "age": age,
    "weight": weight,
    "height": height,
    "income_lpa": income_lpa,
    "smoker": smoker,
    "city": city,
    "occupation": occupation,
})

# ---------------------------------------------------------------------------
# Rule-based label generation: higher health/lifestyle risk -> higher premium
# category. Adds noise so the relationship isn't perfectly deterministic,
# which keeps the model from trivially memorizing a hard rule.
# ---------------------------------------------------------------------------
bmi = df.apply(lambda r: get_bmi(r["weight"], r["height"]), axis=1)

risk_score = np.zeros(N)
risk_score += df["smoker"].astype(int) * 3
risk_score += np.where(bmi > 30, 2, np.where(bmi > 25, 1, 0))
risk_score += np.where(df["age"] > 60, 2, np.where(df["age"] > 45, 1, 0))
risk_score += df["occupation"].isin(["unemployed",
                                    "freelancer"]).astype(int) * 1
risk_score += np.random.normal(0, 0.8, size=N)  # noise

category = np.where(risk_score >= 5, "High", np.where(
    risk_score >= 2.5, "Medium", "Low"))

df["insurance_premium_category"] = category

df.to_csv("raw_data.csv", index=False)
print(df.head())
print("\nLabel distribution:")
print(df["insurance_premium_category"].value_counts())
