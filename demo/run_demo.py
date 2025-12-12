# demo/run_demo.py  (replace your current file with this)
import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import sys

# ensure we can import prepare_features from src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import prepare_features

MODEL = Path("models/crop_rf_search.joblib")
SAMPLE = Path("demo/sample_input.csv")

# load model search object
search = joblib.load(MODEL)
best = search.best_estimator_

# read demo sample
sample = pd.read_csv(SAMPLE).iloc[0:1].copy()

# 1) run the same feature-prep used during training
sample = prepare_features(sample)

# 2) inspect what the fitted preprocessor expects as input column names
pre = best.named_steps["pre"]
# ColumnTransformer / transformers usually store the names they were fit on
expected_cols = None
if hasattr(pre, "feature_names_in_"):
    expected_cols = list(pre.feature_names_in_)
else:
    # fallback: try to read X columns from the pipeline's first step attributes
    try:
        expected_cols = list(pre.transformers_[0][2]) + list(pre.transformers_[1][2])
    except Exception:
        expected_cols = list(sample.columns)

print("Pipeline expects columns:", expected_cols)
print("Sample columns before alignment:", list(sample.columns))

# 3) automatic fix for obvious typo: Temperature vs Temparature
if "Temparature" in expected_cols and "Temperature" in sample.columns:
    sample = sample.rename(columns={"Temperature": "Temparature"})
    print("Renamed 'Temperature' -> 'Temparature' for compatibility.")

# 4) ensure derived nutrient ratio features exist (created by prepare_features)
for col in ["N_to_P", "N_to_K", "P_to_K"]:
    if col not in sample.columns:
        sample[col] = 0.0

# 5) make sure all expected columns exist in the sample;
#    if missing, add them with zeros (safe fallback)
for c in expected_cols:
    if c not in sample.columns:
        sample[c] = 0

# 6) keep only the expected columns in the same order
sample = sample[expected_cols]

print("Sample columns after alignment:", list(sample.columns))

# 7) predict
pred = best.predict(sample)[0]
print("\nPredicted crop:", pred)

# 8) top-3 probabilities if supported
try:
    probs = best.predict_proba(sample)[0]
    # the classifier's classes:
    rf = best.named_steps['rf']
    classes = rf.classes_
    top_idx = np.argsort(probs)[-3:][::-1]
    print("\nTop predictions:")
    for i in top_idx:
        print(f"  {classes[i]} ({probs[i]:.3f})")
except Exception as e:
    print("Probabilities not available or error:", e)
