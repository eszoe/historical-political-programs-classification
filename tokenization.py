import random
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_prep_load import load_corpus
from data_prep_recode import recode_corpus
from label_counts import get_frequent_label_counts

DATA_DIR = Path('data')

# 1. Load and recode data
corpus     = load_corpus(Path('data'))
corpus_rec = recode_corpus(corpus)

# 2. Get filtered category columns
result        = get_frequent_label_counts(corpus_rec, min_count=40)
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

unique_docs = df["doc_title"].unique()

# Shuffle documents: take also into account that documents have different length
unique_docs = df["doc_title"].unique().tolist()

random.seed(42)
random.shuffle(unique_docs)

# 80/20 split
split_idx = int(0.8 * len(unique_docs))

train_docs = unique_docs[:split_idx]
test_docs = unique_docs[split_idx:]

# Create train/test dataframes
train_df    = df[df["doc_title"].isin(train_docs)]
test_df     = df[df["doc_title"].isin(test_docs)]

print("\nTRAIN SIZE:", len(train_df))
print("TEST SIZE:", len(test_df))


# convert to Huggingface documents

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# tokenization

def tokenize(batch):

    return tokenizer(
        batch["sentence"],

        # ManifestoBERTa was fine-tuned with max_length=200
        max_length=200,

        # Pad all sequences to length 200
        padding="max_length",

        # Truncate long sentences
        truncation=True
    )

# tokenize datasets

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)


# Remove unused columns and keep only:
# - input_ids
# - attention_mask
# - labels

columns_to_remove = [
    col for col in train_dataset.column_names
    if col not in [
        "input_ids",
        "attention_mask",
        "labels"
    ]
]

train_dataset = train_dataset.remove_columns(columns_to_remove)
test_dataset = test_dataset.remove_columns(columns_to_remove)

# set PyTorch format

train_dataset.set_format("torch")
test_dataset.set_format("torch")


# load model with own labels

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id), # number of our labels
    id2label=id2label, # mapping of our label
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# sanity check

print("\nTOKENIZED EXAMPLE:")
print(train_dataset[0])

print("\nMODEL LABELS:")
print(model.config.id2label)
