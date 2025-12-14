import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "tn_crop_with_soil.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA)

# -------------------------------
# FIX YEAR COLUMN
# -------------------------------
df["Year"] = (
    df["Year"]
    .astype(str)
    .str.extract(r"(\d{4})")[0]
    .astype(int)
)

# Drop missing soil
df = df.dropna(subset=["Soil_Type"])

# -------------------------------
# FEATURES & TARGET
# -------------------------------
FEATURES = ["District", "Soil_Type", "Season", "Area", "Year"]
TARGET = "Crop"

X = df[FEATURES]
y_raw = df[TARGET]

# -------------------------------
# 🔑 LABEL ENCODING (CRITICAL FIX)
# -------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Save crop label mapping (AWS NEEDS THIS)
joblib.dump(label_encoder, MODEL_DIR / "crop_label_encoder.joblib")

cat_cols = ["District", "Soil_Type", "Season"]
num_cols = ["Area", "Year"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training model...")
pipeline.fit(X_train, y_train)

print("Evaluating...")
y_pred = pipeline.predict(X_test)
f1 = f1_score(y_test, y_pred, average="macro")
print("✅ Macro F1:", round(f1, 4))

# -------------------------------
# SAVE MODEL & SCHEMA
# -------------------------------
joblib.dump(pipeline, MODEL_DIR / "tn_crop_xgb.joblib")

schema = {
    "features": FEATURES,
    "categorical": cat_cols,
    "numerical": num_cols,
    "target": TARGET
}
joblib.dump(schema, MODEL_DIR / "feature_schema.joblib")

print("✅ Model saved successfully")
