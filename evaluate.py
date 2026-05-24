from sklearn.metrics import classification_report
import numpy as np

def run_evaluation(trainer, test_dataset, id2label):
    predictions = trainer.predict(test_dataset)
    logits      = predictions.predictions
    labels      = predictions.label_ids
    preds       = np.argmax(logits, axis=-1)

    # Only use labels that actually appear in the test set
    present_labels = sorted(set(labels.tolist()))
    target_names   = [id2label[i] for i in present_labels]

    print(classification_report(
        labels,
        preds,
        labels=present_labels,
        target_names=target_names,
    ))
