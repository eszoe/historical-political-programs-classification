import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import random

BASE_DIR = Path.cwd()
DATA_DIR = Path('data')

corpus = pd.read_csv(DATA_DIR / 'corpus_ger_cze.csv')

# load tokenizer and model

model_name = (
    "manifesto-project/"
    "manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# create label mappings, columns "label" still needs to be created

# Step 1: get categories with >= 40 observations
result = get_category_counts(corpus_rec, min_count=40)
category_cols = result['Category'].tolist()

# Step 2: create label column
corpus_rec['label'] = corpus_rec.apply(
    lambda row: next((col for col in category_cols if row[col] == 1), None),
    axis=1
)

# Step 3: check
print(corpus_rec['label'].value_counts())
print(f"Unlabeled rows: {corpus_rec['label'].isna().sum()}")

# converting labels to integers

df["labels"] = df["label"].map(label2id)

print("\nDATA WITH INTEGER LABELS:")
print(df.head())

# splitting documents

unique_docs = df["document_id"].unique()

# Shuffle documents: take also into account that documents have different length

random.seed(42)

unique_docs = list(unique_docs)
random.shuffle(unique_docs)

# 80/20 split
split_idx = int(0.8 * len(unique_docs))

train_docs = unique_docs[:split_idx]
test_docs = unique_docs[split_idx:]

# Create train/test dataframes
train_df = df[df["document_id"].isin(train_docs)]
test_df = df[df["document_id"].isin(test_docs)]

print("\nTRAIN SIZE:", len(train_df))
print("TEST SIZE:", len(test_df))

# convert to Huggingface documents

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# tokenization (why after splitting??)

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

    # Number of YOUR labels
    num_labels=len(label2id),

    # Your mappings
    id2label=id2label,
    label2id=label2id
)

# sanity check

print("\nTOKENIZED EXAMPLE:")
print(train_dataset[0])

print("\nMODEL LABELS:")
print(model.config.id2label)