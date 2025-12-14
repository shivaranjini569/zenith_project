import pandas as pd
import joblib
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "tn_crop_with_soil.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA)

# ---------------------------
# Fix Year column
# ---------------------------
df["Year"] = (
    df["Year"]
    .astype(str)
    .str.extract(r"(\d{4})")[0]
    .astype(int)
)

df = df.dropna(subset=["Soil_Type"])

# ---------------------------
# Features & target
# ---------------------------
FEATURES = ["District", "Season", "Soil_Type", "Area", "Year"]
TARGET = "Crop"

X = df[FEATURES]
y = df[TARGET]

# Identify categorical features by index
cat_features = [
    X.columns.get_loc("District"),
    X.columns.get_loc("Season"),
    X.columns.get_loc("Soil_Type")
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training CatBoost model...")
model = CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.05,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    random_seed=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test)
)

print("Evaluating...")
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average="macro")
print("✅ Macro F1:", round(f1, 4))

# Save model + schema
model.save_model(MODEL_DIR / "tn_crop_catboost.cbm")

schema = {
    "features": FEATURES,
    "categorical": ["District", "Season", "Soil_Type"],
    "numerical": ["Area", "Year"],
    "target": TARGET
}
joblib.dump(schema, MODEL_DIR / "feature_schema_catboost.joblib")

print("✅ CatBoost model saved successfully")
