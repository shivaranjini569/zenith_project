# src/model_pipeline.py
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

RANDOM_STATE = 42

# helper to create OneHotEncoder in a version-compatible way
def make_onehot_encoder():
    # try the newest API first (sparse_output), fall back to older (sparse)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            # last fallback: default OHE (may produce sparse matrix)
            return OneHotEncoder(handle_unknown="ignore")

def build_preprocessor(numeric_features, categorical_features):
    """ColumnTransformer: numeric -> median+scale, categorical -> most_frequent+onehot"""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_onehot_encoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    return preprocessor

def train_rf_classifier(X, y, n_iter=12):
    """Train RandomForest inside an sklearn Pipeline with RandomizedSearchCV."""
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

    pre = build_preprocessor(numeric, categorical)

    pipe = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    param_dist = {
        "rf__n_estimators": [100, 200, 400],
        "rf__max_depth": [None, 10, 20, 40],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__class_weight": [None, "balanced"]
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X, y)
    return search

def save_search(search_obj, path):
    joblib.dump(search_obj, path)

def load_search(path):
    return joblib.load(path)
def predict_crop(search_obj, X):
    return search_obj.predict(X)