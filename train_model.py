"""Train a baseline model only on real, independently verified labeled field readings."""
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "data", "sugarcane_readings.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(CSV):
    raise SystemExit("Dataset not found. Run app.py and save field readings first.")

df = pd.read_csv(CSV)
required = ["temperature_c", "humidity_percent", "soil_moisture_percent", "brix", "crop_age_days", "variety", "maturity_label"]
df = df.dropna(subset=["maturity_label"])
df = df[df["maturity_label"].isin(["Immature", "Semi-Mature", "Mature"])]
if len(df) < 30:
    raise SystemExit(f"Only {len(df)} verified labeled rows found. Add more real labeled data before training (30 is only a minimal software guardrail, not scientific validation).")
if df["maturity_label"].nunique() < 2 or df["maturity_label"].value_counts().min() < 2:
    raise SystemExit("Need at least two maturity classes and at least two examples in each class.")

features = ["temperature_c", "humidity_percent", "soil_moisture_percent", "brix", "crop_age_days", "variety"]
X = df[features]
y = df["maturity_label"]
numeric = ["temperature_c", "humidity_percent", "soil_moisture_percent", "brix", "crop_age_days"]
categorical = ["variety"]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Holdout accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, zero_division=0))
joblib.dump(model, os.path.join(MODEL_DIR, "maturity_model.joblib"))
print("Saved model to models/maturity_model.joblib")
