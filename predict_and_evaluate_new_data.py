import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    matthews_corrcoef, accuracy_score
)
from torch.nn.functional import cross_entropy
from config import MODEL_NAME

# # predict_and_evaluate_new_data.py
# # Loads a new Excel dataset, maps labels to the training scheme,
# # runs predictions with both the frozen and LoRA models,
# # and evaluates performance metrics on the new data.



# ── Dataset class for inference ──────────────────────────────────
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=200):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }


# ── Prediction function ──────────────────────────────────────────
def predict(model, tokenizer, texts, batch_size=16):
    model.eval()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = model.to(device)
    dataset = InferenceDataset(texts, tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size)

    all_predictions   = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=-1)
            preds   = torch.argmax(probs, dim=-1)

            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    return all_predictions, all_probabilities


# ── Evaluation function ──────────────────────────────────────────
def evaluate(true_labels, predictions, probabilities, model_name="model"):
    logits_tensor = torch.tensor(np.array(probabilities))
    labels_tensor = torch.tensor(true_labels)
    ce_loss = cross_entropy(logits_tensor, labels_tensor).item()

    metrics = {
        "model":             model_name,
        "accuracy":          accuracy_score(true_labels, predictions),
        "precision_macro":   precision_score(true_labels, predictions, average="macro", zero_division=0),
        "recall_macro":      recall_score(true_labels, predictions, average="macro", zero_division=0),
        "macro_f1":          f1_score(true_labels, predictions, average="macro", zero_division=0),
        "weighted_f1":       f1_score(true_labels, predictions, average="weighted", zero_division=0),
        "mcc":               matthews_corrcoef(true_labels, predictions),
        "cross_entropy":     ce_loss,
    }

    print(f"\n── Evaluation: {model_name} ──────────────────────")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<20} {v:.4f}")

    return metrics


# ── Load frozen model ────────────────────────────────────────────
def load_frozen_model(model_dir="./results/frozen_model"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer


# ── Load LoRA model ──────────────────────────────────────────────
def load_lora_model(model_dir="./results/lora_model"):
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    return model, tokenizer


# ── Main ─────────────────────────────────────────────────────────
def main():
    # ── 1. Load new data ─────────────────────────────────────────
    df    = pd.read_excel("new_data.xlsx")
    texts = df["sentence"].tolist()    # adjust if column name differs

    # ── 2. Load label2id from saved frozen model config ──────────
    model_frozen, tokenizer = load_frozen_model()
    label2id = model_frozen.config.label2id
    id2label = model_frozen.config.id2label

    # ── 3. Map labels to integers ────────────────────────────────
    # Drop rows whose label wasn't seen during training
    df = df[df["label"].isin(label2id)].copy().reset_index(drop=True)
    texts = df["sentence"].tolist()

    unknown = set(df["label"].unique()) - set(label2id.keys())
    if unknown:
        print(f"Warning: {len(unknown)} unknown labels dropped: {unknown}")

    df["label_id"] = df["label"].map(label2id)
    true_labels    = df["label_id"].tolist()

    print(f"New dataset size after label filtering: {len(df)}")

    # ── 4. Predict with frozen model ─────────────────────────────
    print("\nRunning frozen model predictions...")
    preds_frozen, probs_frozen = predict(model_frozen, tokenizer, texts)

    # ── 5. Predict with LoRA model ───────────────────────────────
    print("Running LoRA model predictions...")
    model_lora, tokenizer = load_lora_model()
    preds_lora, probs_lora = predict(model_lora, tokenizer, texts)

    # ── 6. Evaluate both models ──────────────────────────────────
    metrics_frozen = evaluate(true_labels, preds_frozen, probs_frozen, model_name="frozen")
    metrics_lora   = evaluate(true_labels, preds_lora,   probs_lora,   model_name="lora")

    # ── 7. Save metrics comparison ───────────────────────────────
    metrics_df = pd.DataFrame([metrics_frozen, metrics_lora])
    metrics_df.to_excel("new_data_metrics.xlsx", index=False)
    print("\nMetrics saved to new_data_metrics.xlsx")

    # ── 8. Add predictions to dataframe and save ─────────────────
    df["pred_frozen"]       = [id2label[p] for p in preds_frozen]
    df["pred_lora"]         = [id2label[p] for p in preds_lora]
    df["confidence_frozen"] = [round(max(p), 4) for p in probs_frozen]
    df["confidence_lora"]   = [round(max(p), 4) for p in probs_lora]

    df.to_excel("new_data_predictions.xlsx", index=False)
    print("Predictions saved to new_data_predictions.xlsx")


if __name__ == "__main__":
    main()