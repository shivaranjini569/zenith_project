# src/train_classifier.py

import pandas as pd
from pathlib import Path
import sys
import os

# --------- FIX: ensure src/ is on the import path ----------
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

# imports after path fix
from features import prepare_features
from model_pipeline import train_rf_classifier, save_search
from sklearn.model_selection import train_test_split

# paths
DATA_CORE = CURRENT_DIR.parent / "data" / "raw" / "data_core.csv"
MODEL_OUT = CURRENT_DIR.parent / "models" / "crop_rf_search.joblib"


def prepare_data_for_training():
    df = pd.read_csv(DATA_CORE)
    df = prepare_features(df)

    if "Crop Type" not in df.columns:
        raise ValueError("Expected 'Crop Type' in data_core.csv")

    X = df.drop(columns=["Crop Type"])
    y = df["Crop Type"].astype(str)

    return X, y


def main():
    # ensure models/ directory exists
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    X, y = prepare_data_for_training()

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    print("Training rows:", X_train.shape[0], "Holdout rows:", X_hold.shape[0])

    search = train_rf_classifier(X_train, y_train, n_iter=12)

    print("Best params:", search.best_params_)
    print("Best CV score (f1_macro):", search.best_score_)

    save_search(search, MODEL_OUT)
    print("Saved model search object to", MODEL_OUT)


if __name__ == "__main__":
    main()
 