import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / 'Topic classification of historical political programs' / 'data'

cze = pd.read_excel(DATA_DIR / 'Habsburg_Cze_issues.xlsx')
ger = pd.read_excel(DATA_DIR / 'Habsburg_Ger_issues.xlsx')

# --- merge ---
cze = cze.drop(columns=['Farbe'], errors='ignore')
corpus = pd.concat([ger, cze], axis=0, ignore_index=True)

# fill numeric NaNs
numeric_cols = corpus.select_dtypes(include='number').columns
corpus[numeric_cols] = corpus[numeric_cols].fillna(0)

# --- check ---
print("GER columns:", len(ger.columns))
print("CZE columns:", len(cze.columns))
print("Merged columns:", len(corpus.columns))
print("Merged rows:", len(corpus))

# --- save ---
corpus.to_csv(DATA_DIR / 'corpus_ger_cze.csv', index=False)




