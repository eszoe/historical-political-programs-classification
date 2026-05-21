import random
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score
from data_prep_load import load_corpus
from data_prep_recode import recode_corpus
from overview_labels import get_category_counts

DATA_DIR = Path('data')

# ── 1. Load and recode ───────────────────────────────────────────
corpus     = load_corpus(DATA_DIR)
corpus_rec = recode_corpus(corpus)

# ── 2. Get filtered categories ───────────────────────────────────
result        = get_category_counts(corpus_rec, min_count=40)
category_cols = result['Category'].tolist()

# ── 3. Create label column ───────────────────────────────────────
corpus_rec['label'] = corpus_rec.apply(
    lambda row: next((col for col in category_cols if row[col] == 1), None),
    axis=1
)

# ── 4. Drop unlabeled rows ───────────────────────────────────────
df = corpus_rec[corpus_rec['label'].notna()].copy()
df = df[['doc_title', 'sentence', 'label']].reset_index(drop=True)

# ── 5. Label mappings ────────────────────────────────────────────
unique_labels = sorted(df['label'].unique())
label2id      = {label: idx for idx, label in enumerate(unique_labels)}
id2label      = {idx: label for label, idx in label2id.items()}
df['labels']  = df['label'].map(label2id)

print(f"Number of labels: {len(label2id)}")
print(f"Dataset size: {len(df)}")

# ── 6. Train/test split ──────────────────────────────────────────
unique_docs = df['doc_title'].unique().tolist()
random.seed(42)
random.shuffle(unique_docs)

split_idx  = int(0.8 * len(unique_docs))
train_docs = unique_docs[:split_idx]
test_docs  = unique_docs[split_idx:]

train_df = df[df['doc_title'].isin(train_docs)].reset_index(drop=True)
test_df  = df[df['doc_title'].isin(test_docs)].reset_index(drop=True)

print(f"Train size: {len(train_df)}")
print(f"Test size:  {len(test_df)}")

# ── 7. Tokenizer ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        max_length=200,
        padding="max_length",
        truncation=True
    )

# ── 8. HuggingFace datasets ──────────────────────────────────────
train_dataset = Dataset.from_pandas(train_df)
test_dataset  = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

keep_cols = ["input_ids", "attention_mask", "labels"]
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in keep_cols]
)
test_dataset = test_dataset.remove_columns(
    [col for col in test_dataset.column_names if col not in keep_cols]
)

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# ── 9. Load model ────────────────────────────────────────────────
model_name = (
    "manifesto-project/"
    "manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1"
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# ── 10. Metrics ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "macro_f1":    f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted")
    }

# ── 11. Training arguments ───────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
)

# ── 12. Train ────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# ── 13. Save model ───────────────────────────────────────────────
model.save_pretrained('./results/final_model')
tokenizer.save_pretrained('./results/final_model')
print("Model saved to ./results/final_model")