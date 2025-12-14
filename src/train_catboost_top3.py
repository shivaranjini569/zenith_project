import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

# ----------------------------
# 1. Load dataset
# ----------------------------
print("Loading NDVI-aligned dataset...")
df = pd.read_csv("data/processed/tn_ml_ndvi_only.csv")

# Target
TARGET = "Crop"

# Drop columns not used for prediction
drop_cols = [
    "State",
    "Crop",   # target
]

X = df.drop(columns=drop_cols)
y = df[TARGET]

# ----------------------------
# 2. Identify categorical features
# ----------------------------
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
print("Categorical features:", cat_features)

# ----------------------------
# 3. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 4. Train CatBoost
# ----------------------------
model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.1,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    verbose=100,
    random_seed=42
)

print("Training CatBoost model...")
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# ----------------------------
# 5. Evaluate Top-1 & Top-3
# ----------------------------
probs = model.predict_proba(X_test)
classes = model.classes_

# Top-1
top1_preds = classes[np.argmax(probs, axis=1)]
top1_f1 = f1_score(y_test, top1_preds, average="macro")

# Top-3 accuracy
top3_correct = 0
for i, true_crop in enumerate(y_test.values):
    top3 = classes[np.argsort(probs[i])[-3:]]
    if true_crop in top3:
        top3_correct += 1

top3_acc = top3_correct / len(y_test)

print(f"\n✅ Macro F1 (Top-1): {top1_f1:.4f}")
print(f"✅ Top-3 Accuracy: {top3_acc:.4f}")

# ----------------------------
# 6. Save model
# ----------------------------
joblib.dump(model, "models/catboost_tn_top3.joblib")
print("✅ Model saved to models/catboost_tn_top3.joblib")

# ----------------------------
# 7. Save feature schema (IMPORTANT FOR XAI)
# ----------------------------
feature_schema = {
    "features": X.columns.tolist(),
    "cat_features": cat_features
}

joblib.dump(feature_schema, "models/feature_schema_catboost.joblib")
print("✅ Feature schema saved")

