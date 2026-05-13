import pandas as pd
from pathlib import Path


BASE_DIR = Path.cwd()
DATA_DIR = Path('data')

# 1) Merge German and Czech data
cze_issues = pd.read_excel(DATA_DIR / 'HMD_Cze_issues.xlsx', engine='openpyxl')
ger_issues = pd.read_excel(DATA_DIR / 'HMD_Ger_issues.xlsx', engine='openpyxl')
ger_doc = pd.read_excel(DATA_DIR / 'HMD_Ger_doc.xlsx', engine='openpyxl')
cze_doc = pd.read_excel(DATA_DIR / 'HMD_Cze_doc.xlsx', engine='openpyxl')
jew_doc = pd.read_excel(DATA_DIR / 'HMD_Jew_doc.xlsx', engine='openpyxl')

hmd_issues = pd.concat([ger_issues, cze_issues], axis=0, ignore_index=True)

hmd_issues['doc_title'] = hmd_issues['Dokumentname']


# 2) Merge with data on document type, year, party
corpus = (
    hmd_issues
    .merge(ger_doc, on="doc_title", how="inner")
    .merge(cze_doc, on="doc_title", how="inner")
    .merge(jew_doc, on="doc_title", how="inner")
)

numeric_cols = corpus.select_dtypes(include='number').columns
corpus[numeric_cols] = corpus[numeric_cols].fillna(0)

corpus.to_csv(DATA_DIR / 'corpus_ger_cze.csv', index=False)
