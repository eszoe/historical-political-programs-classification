import numpy as np
from sklearn.metrics import classification_report

def run_evaluation(trainer, test_dataset, id2label):
    predictions = trainer.predict(test_dataset)
    preds  = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    unique_labels = [id2label[i] for i in sorted(id2label.keys())]
    print(classification_report(labels, preds, target_names=unique_labels))

    return preds, labels
