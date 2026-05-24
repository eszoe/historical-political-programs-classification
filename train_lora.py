from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from config import TRAINING_CONFIG_LORA


def get_training_args_lora(output_dir="./results/lora_model"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_CONFIG_LORA["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG_LORA["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG_LORA["per_device_eval_batch_size"],
        learning_rate=TRAINING_CONFIG_LORA["learning_rate"],
        weight_decay=TRAINING_CONFIG_LORA["weight_decay"],
        warmup_ratio=TRAINING_CONFIG_LORA["warmup_ratio"],
        fp16=TRAINING_CONFIG_LORA["fp16"],
        gradient_accumulation_steps=TRAINING_CONFIG_LORA["gradient_accumulation_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir="./logs/lora",
        logging_steps=10,
    )


def build_trainer_lora(model, train_dataset, test_dataset, compute_metrics):
    training_args = get_training_args_lora()

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=TRAINING_CONFIG_LORA["early_stopping_patience"]
        )],
    )