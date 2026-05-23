from pathlib import Path
from data_prep_load import load_corpus
from data_prep_recode import recode_corpus
from label_counts import get_frequent_label_counts
from dataset import split_by_document, build_hf_dataset
from model import load_model
from metrics import compute_metrics
from train import get_training_args, build_trainer
from evaluate import run_evaluation

DATA_DIR = Path('data')

def main():
    # ── 1. Load and recode ───────────────────────────────────────
    corpus     = load_corpus(DATA_DIR)
    corpus_rec = recode_corpus(corpus)

    # ── 2. Filter categories ─────────────────────────────────────
    result        = get_frequent_label_counts(corpus_rec, min_count=40)
    category_cols = result['Topic'].tolist()

    # ── 3. Create label column ───────────────────────────────────
    corpus_rec['label'] = corpus_rec.apply(
        lambda row: next((col for col in category_cols if row[col] == 1), None),
        axis=1
    )

    # ── 4. Drop unlabeled rows ───────────────────────────────────
    df = corpus_rec[corpus_rec['label'].notna()].copy()
    df = df[['doc_title', 'sentence', 'label']].reset_index(drop=True)

    # ── 5. Label mappings ────────────────────────────────────────
    unique_labels = sorted(df['label'].unique())
    label2id      = {label: idx for idx, label in enumerate(unique_labels)}
    id2label      = {idx: label for label, idx in label2id.items()}
    df['labels']  = df['label'].map(label2id)

    print(f"Number of labels: {len(label2id)}")
    print(f"Dataset size:     {len(df)}")

    # ── 6. Split and build datasets ──────────────────────────────
    train_df, test_df = split_by_document_and_language(df)
    print(f"Train size: {len(train_df)}")
    print(f"Test size:  {len(test_df)}")

    train_dataset = build_hf_dataset(train_df)
    test_dataset  = build_hf_dataset(test_df)

    # ── 7. Load model ────────────────────────────────────────────
    model = load_model(label2id, id2label)

    # ── 8. Train ─────────────────────────────────────────────────
    training_args = get_training_args()
    trainer       = build_trainer(
        model, training_args, train_dataset, test_dataset, compute_metrics
    )
    trainer.train()

    # ── 9. Evaluate ──────────────────────────────────────────────
    run_evaluation(trainer, test_dataset, id2label)

    # ── 10. Save ─────────────────────────────────────────────────
    model.save_pretrained('./results/final_model')
    print("Model saved to ./results/final_model")

if __name__ == "__main__":
    main()
