import pandas as pd
from pathlib import Path
import sys

root = Path('data/external/india_drought')
if not root.exists():
    print('ERROR: folder not found:', root)
    sys.exit(1)

files = sorted([p for p in root.rglob('*') if p.suffix.lower() in ('.csv','.xls','.xlsx')])
print('Files found:', [p.name for p in files])

keywords = ['ndvi','vhi','evi','vci','lai','veget','vegetation','ndwi','gpp','savi','biomass']
report = []

for p in files:
    print('\\n===', p.name, '===') 
    try:
        if p.suffix.lower() in ('.xls','.xlsx'):
            df = pd.read_excel(p, sheet_name=0, nrows=5, engine='openpyxl')
        else:
            df = pd.read_csv(p, nrows=5, low_memory=False)
    except Exception as e:
        print('  ERROR reading file:', e)
        continue
    cols = df.columns.tolist()
    print('  columns:', cols)
    sample = df.head(5)
    # show sample values for columns that match keywords
    found = []
    for c in cols:
        lower = c.lower()
        for kw in keywords:
            if kw in lower:
                found.append(c)
                break
    if found:
        print('  ** vegetation-like columns found:', found, '**')
        print('  sample rows for those columns:')
        print(sample[found].to_string(index=False))
    else:
        print('  no obvious vegetation columns found in header.')
    # also print first row (for context)
    print('  sample row (first):')
    try:
        print(sample.iloc[0].to_dict())
    except:
        print('   (could not print sample row)')
