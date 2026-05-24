from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from config import TRAINING_CONFIG_FROZEN

# 18 out of 24, 75% of the model are frozen, keeping only the top 6 layers and the classifier head trainable.
# This is a reasonable default for a small dataset; you can tune this down to 12 if results are poor.
def freeze_base_layers(model, num_layers_to_freeze=18):
    # Freeze embedding layer: often overlooked but important; embedding layers have many parameters
    # and should also be frozen when doing partial fine-tuning on small data.
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False

    # Freeze first N transformer layers
    for layer in model.base_model.encoder.layer[:num_layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False

    # Report trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    return model


def get_training_args_frozen(output_dir="./results/frozen_model"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_CONFIG_FROZEN["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG_FROZEN["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG_FROZEN["per_device_eval_batch_size"],
        learning_rate=TRAINING_CONFIG_FROZEN["learning_rate"],
        weight_decay=TRAINING_CONFIG_FROZEN["weight_decay"],
        warmup_ratio=TRAINING_CONFIG_FROZEN["warmup_ratio"],
        fp16=TRAINING_CONFIG_FROZEN["fp16"],
        gradient_accumulation_steps=TRAINING_CONFIG_FROZEN["gradient_accumulation_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir="./logs/frozen",
        logging_steps=10,
    )


def build_trainer_frozen(model, train_dataset, test_dataset, compute_metrics):
    model = freeze_base_layers(
        model,
        num_layers_to_freeze=TRAINING_CONFIG_FROZEN["num_layers_to_freeze"]
    )
    training_args = get_training_args_frozen()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=TRAINING_CONFIG_FROZEN["early_stopping_patience"]
        )],
    )