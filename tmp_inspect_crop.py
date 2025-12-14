import pandas as pd
import sys

path = r"data/raw/India Agriculture Crop Production.csv"

try:
    df = pd.read_csv(path, low_memory=False)
except FileNotFoundError:
    print("ERROR: file not found:", path)
    sys.exit(1)
except Exception as e:
    print("ERROR reading CSV:", e)
    sys.exit(1)

# find columns that look like year columns
year_cols = [c for c in df.columns if "year" in c.lower() or "yr"==c.lower()]

print("Year-like columns found:", year_cols)

if year_cols:
    col = year_cols[0]
    print(f"\nUsing column: {col}")
    vals = df[col].dropna().astype(str).unique()[:50].tolist()
    print("Sample unique values (up to 50):")
    print(vals)
else:
    # fallback: try to detect patterns like '2001-02' or single 4-digit years anywhere
    possible = []
    for c in df.columns:
        s = df[c].astype(str)
        if s.str.contains(r'\d{4}-\d{2}', regex=True).any() or s.str.contains(r'^\d{4}$', regex=True).any():
            possible.append(c)
    if possible:
        print("Columns with year-like values detected:", possible)
        c = possible[0]
        print(f"\nShowing sample values from column: {c}")
        print(df[c].dropna().astype(str).unique()[:50].tolist())
    else:
        print("\nCould not auto-detect a year column. Showing first 10 rows for inspection:")
        print(df.head(10).to_string(index=False))
