"""
04_train_goemotions_deberta.py

Train DeBERTa-v3-large on the GoEmotions dataset for multi-label emotion classification.

Dataset:
    GoEmotions

Task type:
    Multi-label classification

Outputs:
    - trained model checkpoints
    - final saved model and tokenizer in OUTPUT_DIR
"""

import os
import numpy as np
from datasets import load_dataset, Sequence, Value
from sklearn.metrics import f1_score
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

os.environ["WANDB_DISABLED"] = "true"

MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_DIR = os.path.expanduser("~/thesis_models/deberta_goemo")
MAX_LENGTH = 256
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3


def build_label_mappings(dataset):
    """Create label lookup dictionaries from the GoEmotions dataset."""
    label_names = dataset["train"].features["labels"].feature.names
    num_labels = len(label_names)
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    return label_names, num_labels, id2label, label2id


def preprocess_function(batch, tokenizer, num_labels):
    """
    Tokenize text and convert label lists into multi-hot vectors.
    This is required because GoEmotions is a multi-label dataset.
    """
    encodings = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    labels = []
    for sample_labels in batch["labels"]:
        multi_hot = np.zeros(num_labels, dtype=np.float32)
        for label_id in sample_labels:
            multi_hot[label_id] = 1.0
        labels.append(multi_hot.tolist())

    encodings["labels"] = labels
    return encodings


def compute_metrics(eval_pred):
    """Compute macro-F1 for multi-label predictions."""
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"macro_f1": macro_f1}


def main():
    print("Loading GoEmotions dataset...")
    raw = load_dataset("go_emotions")

    label_names, num_labels, id2label, label2id = build_label_mappings(raw)

    print(f"Number of labels: {num_labels}")
    print(f"First 10 labels: {label_names[:10]}")

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    print("Tokenizing dataset...")
    encoded = raw.map(
        lambda batch: preprocess_function(batch, tokenizer, num_labels),
        batched=True,
    )

    # Remove raw text after tokenization; keep encoded labels.
    encoded = encoded.remove_columns(["text"])

    # Cast labels to float32 because BCEWithLogitsLoss expects floating-point targets.
    encoded = encoded.cast_column("labels", Sequence(Value("float32")))
    encoded.set_format("torch")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure Hugging Face Trainer settings for supervised training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=100,
        save_strategy="steps",
        save_steps=2500,
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        load_best_model_at_end=False,
        fp16=False,
    )

# Initialize Trainer with training and validation splits
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(encoded["test"])
    print("Test results:", test_results)

    print(f"Saving final model to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()