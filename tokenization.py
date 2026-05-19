import random
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_preparation import load_corpus
from recode_variables import recode_corpus
from overview_labels import get_category_counts

DATA_DIR = Path('data')

# 1. Load and recode data
corpus     = load_corpus(Path('data'))
corpus_rec = recode_corpus(corpus)

# 2. Get filtered category columns
result        = get_category_counts(corpus_rec, min_count=40)
category_cols = result['Category'].tolist()

# 3. Create label column
corpus_rec['label'] = corpus_rec.apply(
    lambda row: next((col for col in category_cols if row[col] == 1), None),
    axis=1
)

print(corpus_rec['label'].value_counts())
print(f"Unlabeled rows: {corpus_rec['label'].isna().sum()}")

# 4. Drop unlabeled rows
df = corpus_rec[corpus_rec['label'].notna()].copy()
df = df[['doc_title', 'sentence', 'label']].reset_index(drop=True)

# 5. Create label mappings
unique_labels = sorted(df['label'].unique())
label2id      = {label: idx for idx, label in enumerate(unique_labels)}
id2label      = {idx: label for label, idx in label2id.items()}

print("\nLABEL2ID:")
print(label2id)

# 6. Convert labels to integers
df['labels'] = df['label'].map(label2id)

print("\nDATA WITH INTEGER LABELS:")
print(df.head())

# 7. Load tokenizer
model_name = (
    "manifesto-project/"
    "manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1"
)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

# splitting documents

unique_docs = corpus_rec["document_id"].unique()

# Shuffle documents: take also into account that documents have different length

random.seed(42)

unique_docs = list(unique_docs)
random.shuffle(unique_docs)

# 80/20 split
split_idx = int(0.8 * len(unique_docs))

train_docs = unique_docs[:split_idx]
test_docs = unique_docs[split_idx:]

# Create train/test dataframes
train_corpus_rec = [corpus_rec["document_id"].isin(train_docs)]
test_corpus_rec = corpus_rec[corpus_rec["document_id"].isin(test_docs)]

print("\nTRAIN SIZE:", len(train_corpus_rec))
print("TEST SIZE:", len(test_corpus_rec))
