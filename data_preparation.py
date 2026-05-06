import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

cze = pd.read_excel(DATA_DIR / 'Habsburg_Cze_issues.xlsx')
ger = pd.read_excel(DATA_DIR / 'Habsburg_Ger_issues.xlsx')

# Merge German and Czech datasets
corpus = pd.concat([ger, cze], axis=0, ignore_index=True)

numeric_cols = corpus.select_dtypes(include='number').columns
corpus[numeric_cols] = corpus[numeric_cols].fillna(0)

print("GER columns:", len(ger.columns))
print("CZE columns:", len(cze.columns))
print("Merged columns:", len(corpus.columns))
print("Merged rows:", len(corpus))

corpus.to_csv(DATA_DIR / 'corpus_ger_cze.csv', index=False)




