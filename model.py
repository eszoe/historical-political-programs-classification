from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from config import MODEL_NAME, LORA_CONFIG


def load_model(label2id, id2label):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def apply_lora(model):
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def get_model(label2id, id2label):
    model = load_model(label2id, id2label)
    model = apply_lora(model)
    return model