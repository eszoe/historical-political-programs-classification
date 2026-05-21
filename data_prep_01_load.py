import pandas as pd
from pathlib import Path


def load_corpus(data_dir: Path = Path('data')) -> pd.DataFrame:
    """
    Loads and merges HMD issue and document files into a single corpus.

    Args:
        data_dir: Path to the directory containing the Excel files.

    Returns:
        Merged DataFrame saved as corpus_ger_cze.csv
    """

    # 1. Load issue files
    ger_issues = pd.read_excel(data_dir / 'HMD_Ger_issues.xlsx', engine='openpyxl')
    cze_issues = pd.read_excel(data_dir / 'HMD_Cze_issues.xlsx', engine='openpyxl')

    # 2. Load document metadata files
    ger_doc = pd.read_excel(data_dir / 'HMD_Ger_doc.xlsx', engine='openpyxl')
    cze_doc = pd.read_excel(data_dir / 'HMD_Cze_doc.xlsx', engine='openpyxl')
    jew_doc = pd.read_excel(data_dir / 'HMD_Jew_doc.xlsx', engine='openpyxl')

    # 3. Merge issue files
    hmd_issues = pd.concat([ger_issues, cze_issues], axis=0, ignore_index=True)
    hmd_issues = hmd_issues.rename(columns={'Dokumentname': 'doc_title'})

    # 4. Merge all doc metadata into one table
    all_docs = pd.concat([ger_doc, cze_doc, jew_doc], axis=0, ignore_index=True)
    all_docs = all_docs.drop_duplicates(subset='doc_title')  # safety check

    # 5. Single left join: keeps all issues, adds metadata where available
    corpus = hmd_issues.merge(all_docs, on='doc_title', how='left')

    # 6. Fill numeric NaNs
    numeric_cols = corpus.select_dtypes(include='number').columns
    corpus[numeric_cols] = corpus[numeric_cols].fillna(0)

    # 7. Save
    output_path = data_dir / 'corpus_ger_cze.csv'
    corpus.to_csv(output_path, index=False)
    print(f"Corpus saved to {output_path} — shape: {corpus.shape}")

    return corpus

if __name__ == '__main__':
    corpus = load_corpus(Path('data'))