from pathlib import Path
from data_prep_load import load_corpus
from data_prep_recode import recode_corpus
from label_counts import get_frequent_label_counts
from dataset import split_by_document_and_language, build_hf_dataset
from model import load_model, get_model
from metrics import compute_metrics
from train_frozen import build_trainer_frozen
from train_lora import build_trainer_lora
from evaluate import run_evaluation

DATA_DIR = Path('data')


def run_experiment(model, build_trainer_fn, output_dir, train_dataset, test_dataset, id2label):
    trainer = build_trainer_fn(model, train_dataset, test_dataset, compute_metrics)
    trainer.train()
    run_evaluation(trainer, test_dataset, id2label)
    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


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
    df = df[['doc_lan', 'doc_title', 'sentence', 'label']].reset_index(drop=True)

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

    # ── 7. Experiment A: Frozen layers ───────────────────────────
    print("\n" + "="*50)
    print("EXPERIMENT A: Partially frozen layers")
    print("="*50)
    model_frozen = load_model(label2id, id2label)
    run_experiment(
        model=model_frozen,
        build_trainer_fn=build_trainer_frozen,
        output_dir="./results/frozen_model",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        id2label=id2label,
    )

    # ── 8. Experiment B: LoRA ────────────────────────────────────
    print("\n" + "="*50)
    print("EXPERIMENT B: LoRA")
    print("="*50)
    model_lora = get_model(label2id, id2label)
    run_experiment(
        model=model_lora,
        build_trainer_fn=build_trainer_lora,
        output_dir="./results/lora_model",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        id2label=id2label,
    )


if __name__ == "__main__":
    main()