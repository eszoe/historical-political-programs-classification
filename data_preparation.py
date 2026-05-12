import pandas as pd
from pathlib import Path


BASE_DIR = Path.cwd()
DATA_DIR = Path('data')

# 1) Merge German and Czech data
cze = pd.read_excel(DATA_DIR / 'HMD_Cze_issues.xlsx')
ger = pd.read_excel(DATA_DIR / 'HMD_Ger_issues.xlsx')

corpus = pd.concat([ger, cze], axis=0, ignore_index=True)

numeric_cols = corpus.select_dtypes(include='number').columns
corpus[numeric_cols] = corpus[numeric_cols].fillna(0)

# 2) Merge with data on document type, year, party


corpus.to_csv(DATA_DIR / 'corpus_ger_cze.csv', index=False)
