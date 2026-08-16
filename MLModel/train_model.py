import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from feature_engineering import compute_features

# ---------------------------------------------------------------------------
# 1. Load raw data and apply the same feature engineering the API will use
# ---------------------------------------------------------------------------
raw_df = pd.read_csv("raw_data.csv")

engineered_rows = raw_df.apply(
    lambda r: compute_features(
        age=r["age"], weight=r["weight"], height=r["height"],
        income_lpa=r["income_lpa"], smoker=r["smoker"],
        city=r["city"], occupation=r["occupation"],
    ),
    axis=1,
    result_type="expand",
)

df = engineered_rows.copy()
df["insurance_premium_category"] = raw_df["insurance_premium_category"]

print("Transformed data sample:")
print(df.head())

# ---------------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------------
X = df.drop(columns=["insurance_premium_category"])
y = df["insurance_premium_category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Preprocessing + model pipeline
#    bmi, income_lpa, city_tier -> numeric, passed through as-is
#    age_group, lifestyle_risk, occupation -> one-hot encoded
# ---------------------------------------------------------------------------
categorical_features = ["age_group", "lifestyle_risk", "occupation"]
numeric_features = ["bmi", "income_lpa", "city_tier"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
], remainder="passthrough")

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
    )),
])

model_pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
y_pred = model_pipeline.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 5. Save the full pipeline (preprocessing + model together) as model.pkl
#    so the API only needs to call compute_features() then .predict()
# ---------------------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("\nSaved model.pkl")
