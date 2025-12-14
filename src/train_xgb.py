# src/train_xgb.py
# Fixed: encode string class labels to integers (LabelEncoder) to avoid XGBoost class mismatch.
# Compatible with different sklearn versions (handles sparse_output argument).
# Usage: python src/train_xgb.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "training_with_climate.csv"
MODEL_OUT = ROOT / "models" / "xgb_climate_model.joblib"
ENC_OUT = ROOT / "models" / "label_encoder.joblib"
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
print("Loaded:", DATA, "Rows:", df.shape[0])

# target (raw strings)
if "Crop Type" not in df.columns:
    raise SystemExit("Expected 'Crop Type' column in training data")

y_raw = df["Crop Type"].astype(str).values
X = df.drop(columns=["Crop Type"])

# encode labels to integers (robust for XGBoost & CV)
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, ENC_OUT)
print("Saved label encoder ->", ENC_OUT)
print("Found classes:", list(le.classes_))

# train split (use encoded y for stratify)
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Holdout:", X_hold.shape)

# identify numeric & categorical
numeric = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical = [c for c in X_train.columns if c not in numeric]

print("Numeric features:", len(numeric), "Categorical features:", len(categorical))

# Build a OneHotEncoder compatible with older and newer sklearn
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore")

# Preprocessing pipelines
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", make_ohe())
])

pre = ColumnTransformer([
    ("num", num_pipeline, numeric),
    ("cat", cat_pipeline, categorical)
], remainder="drop")

# XGBoost classifier
xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False
)

pipe = Pipeline([
    ("pre", pre),
    ("xgb", xgb)
])

# Hyperparameter search space
param_dist = {
    "xgb__n_estimators": [150, 300, 500],
    "xgb__max_depth": [6, 10, 16],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample": [0.7, 0.9, 1.0],
    "xgb__colsample_bytree": [0.7, 1.0]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    error_score="raise"  # fail fast so we can see useful traceback if something breaks
)

print("Starting RandomizedSearchCV...")
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("Best CV F1:", search.best_score_)

# Save the best estimator (the full pipeline)
best_est = search.best_estimator_
joblib.dump(best_est, MODEL_OUT)
print("Saved model ->", MODEL_OUT)

# Evaluate on holdout and print human-readable classification report (decode labels)
preds = best_est.predict(X_hold)
print("Holdout f1_macro:", f1_score(y_hold, preds, average="macro"))

# inverse-transform labels for readable report
le = joblib.load(ENC_OUT)
y_hold_decoded = le.inverse_transform(y_hold)
preds_decoded = le.inverse_transform(preds)
print("Classification report (holdout):")
print(classification_report(y_hold_decoded, preds_decoded, zero_division=0))
