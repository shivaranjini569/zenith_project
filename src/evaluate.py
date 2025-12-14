# src/evaluate.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# ensure we can import prepare_features from src
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "src"))
from features import prepare_features

MODEL = Path("models/crop_rf_search.joblib")
DATA_CORE = Path("data/raw/data_core.csv")
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_holdout_and_model():
    search = joblib.load(MODEL)
    best = search.best_estimator_

    # load and prepare features exactly as training
    df = pd.read_csv(DATA_CORE)
    df = prepare_features(df)

    # align column names (Temperature typo) - same logic as demo
    if "Temparature" in best.named_steps["pre"].feature_names_in_ and "Temperature" in df.columns:
        # rename if needed (but training used Temparature so ensure df matches)
        df = df.rename(columns={"Temperature": "Temparature"})

    if "Crop Type" not in df.columns:
        raise ValueError("expected 'Crop Type' in data_core.csv after prepare_features")

    X = df.drop(columns=["Crop Type"])
    y = df["Crop Type"].astype(str)

    # reproduce same split used in training
    from sklearn.model_selection import train_test_split
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Ensure X_hold contains all columns expected by the pipeline preprocessor.
    pre = best.named_steps["pre"]
    expected_cols = list(pre.feature_names_in_)
    # if feature_names_in_ missing, fall back to transformer's defined lists
    if not expected_cols:
        try:
            expected_cols = list(pre.transformers_[0][2]) + list(pre.transformers_[1][2])
        except Exception:
            expected_cols = list(X.columns)

    # Add any missing expected columns with safe defaults
    for c in expected_cols:
        if c not in X_hold.columns:
            if c in X_hold.select_dtypes(include=["number"]).columns:
                X_hold[c] = 0
            else:
                X_hold[c] = "missing"

    # Reorder X_hold to expected order
    X_hold = X_hold[expected_cols]

    return search, best, X_hold, y_hold

def evaluate():
    search, best, X_hold, y_hold = load_holdout_and_model()
    preds = best.predict(X_hold)
    print("CLASSIFICATION REPORT (holdout):")
    print(classification_report(y_hold, preds, zero_division=0))

    # confusion matrix (normalized)
    labels = best.named_steps['rf'].classes_
    cm = confusion_matrix(y_hold, preds, labels=labels, normalize='true')
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(cm_df, annot=False, cmap="Blues", ax=ax)
    ax.set_title("Normalized Confusion Matrix (holdout)")
    plt.tight_layout()
    fig_path = OUT_DIR / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=150)
    print("Saved confusion matrix to", fig_path)

    # save a small CSV with report metrics per class
    report = classification_report(y_hold, preds, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_path = OUT_DIR / "classification_report_holdout.csv"
    report_df.to_csv(report_path)
    print("Saved classification report to", report_path)

if __name__ == "__main__":
    evaluate()
