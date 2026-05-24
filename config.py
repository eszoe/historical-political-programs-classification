MODEL_NAME = (
    "manifesto-project/"
    "manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1"
)

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.2,
    "target_modules": ["query", "value"],
    "bias": "none",
}

TRAINING_CONFIG_LORA = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "fp16": True,
    "gradient_accumulation_steps": 2,
    "early_stopping_patience": 3,
}

TRAINING_CONFIG_FROZEN = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    # between full fine-tuning (2e-5) and LoRA (3e-4). The unfrozen layers are pretrained weights
    # so they don't need as high a rate as LoRA adapters,
    # but they need more than full fine-tuning since fewer layers share the learning burden.
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "fp16": True,
    "gradient_accumulation_steps": 2,
    "early_stopping_patience": 3,
    "num_layers_to_freeze": 18,
}