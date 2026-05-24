import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, matthews_corrcoef
)
from torch.nn.functional import cross_entropy
import torch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    ce_loss = cross_entropy(
        torch.tensor(logits),
        torch.tensor(labels)
    ).item()

    return {
        "accuracy":        (predictions == labels).mean(),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro":    recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1":        f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1":     f1_score(labels, predictions, average="weighted", zero_division=0),
        "mcc":             matthews_corrcoef(labels, predictions),
        "cross_entropy":   ce_loss,
    }