# src/features.py
import pandas as pd
import numpy as np

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces from string columns and normalize basic types."""
    # ensure we operate on a copy
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        # convert to str first to avoid Categorical pitfalls, then strip
        df[col] = df[col].astype(str).str.strip()
    return df

def add_nutrient_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features — N/P, N/K, P/K ratios."""
    if all(col in df.columns for col in ["Nitrogen", "Phosphorous", "Potassium"]):
        # convert to numeric safely
        df["Nitrogen"] = pd.to_numeric(df["Nitrogen"], errors="coerce")
        df["Phosphorous"] = pd.to_numeric(df["Phosphorous"], errors="coerce")
        df["Potassium"] = pd.to_numeric(df["Potassium"], errors="coerce")
        df["N_to_P"] = df["Nitrogen"] / (df["Phosphorous"] + 1e-6)
        df["N_to_K"] = df["Nitrogen"] / (df["Potassium"] + 1e-6)
        df["P_to_K"] = df["Phosphorous"] / (df["Potassium"] + 1e-6)
    return df

def add_temperature_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Bucketize temperature — helps tree models generalize."""
    if "Temperature" in df.columns or "Temparature" in df.columns:
        # accept both spellings by normalizing to Temperature
        if "Temparature" in df.columns and "Temperature" not in df.columns:
            df = df.rename(columns={"Temparature": "Temperature"})
        df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
        bins = [-50, 15, 20, 25, 30, 40, 60]
        labels = ["very_cold", "cold", "mild", "warm", "hot", "very_hot"]
        df["temp_bucket"] = pd.cut(df["Temperature"], bins=bins, labels=labels)
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline of light cleaning + domain features. Safe for single-row input."""
    df = df.copy()
    df = clean_basic(df)
    df = add_nutrient_ratios(df)
    df = add_temperature_bins(df)

    # fill numeric columns with 0 and categorical/text with "missing"
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(0)

    # for categorical/text columns, fill with string placeholder
    for c in cat_cols:
        # convert to str (this avoids Categorical setitem issues)
        df[c] = df[c].astype(str).fillna("missing").replace("nan", "missing")

    return df
