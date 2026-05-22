import random
from datasets import Dataset
from transformers import AutoTokenizer

TOKENIZER_NAME = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def split_by_document(df, train_ratio=0.8, seed=42):
    unique_docs = df["doc_title"].unique().tolist()
    random.seed(seed)
    random.shuffle(unique_docs)

    split_idx  = int(train_ratio * len(unique_docs))
    train_docs = unique_docs[:split_idx]
    test_docs  = unique_docs[split_idx:]

    train_df = df[df["doc_title"].isin(train_docs)].reset_index(drop=True)
    test_df  = df[df["doc_title"].isin(test_docs)].reset_index(drop=True)

    return train_df, test_df

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        max_length=200,
        padding="max_length",
        truncation=True
    )

def build_hf_dataset(df):
    """
    Convert a pandas DataFrame into a tokenized HuggingFace Dataset
    """
    keep_cols = ["input_ids", "attention_mask", "labels"]

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in keep_cols]
    )
    dataset.set_format("torch")
    return dataset
