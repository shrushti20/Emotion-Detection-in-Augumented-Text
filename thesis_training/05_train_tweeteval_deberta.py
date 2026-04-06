"""
05_train_tweeteval_deberta.py

Train DeBERTa-v3-large on the TweetEval Emotion dataset for single-label emotion classification.

Dataset:
    TweetEval Emotion

Task type:
    Single-label multiclass classification

Outputs:
    - trained model checkpoints
    - final saved model and tokenizer in OUTPUT_DIR
"""

import os
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

os.environ["WANDB_DISABLED"] = "true"

MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_DIR = os.path.expanduser("~/thesis_models/deberta_tweets")
MAX_LENGTH = 256
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3


def build_label_mappings(dataset):
    """Create label lookup dictionaries from the TweetEval dataset."""
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}
    return label_names, num_labels, id2label, label2id


def preprocess_function(batch, tokenizer):
    """Tokenize input text and keep the original single-label targets."""
    encodings = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    encodings["labels"] = batch["label"]
    return encodings


def compute_metrics(eval_pred):
    """Compute accuracy and macro-F1 for single-label classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "macro_f1": macro_f1}


def main():
    print("Loading TweetEval Emotion dataset...")
    raw = load_dataset("tweet_eval", "emotion")

    label_names, num_labels, id2label, label2id = build_label_mappings(raw)

    print(f"Number of labels: {num_labels}")
    print(f"Labels: {label_names}")

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    print("Tokenizing dataset...")
    encoded = raw.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
    )

    # Remove raw text and original label column after creating the Trainer-ready labels field.
    encoded = encoded.remove_columns(["text", "label"])
    encoded.set_format("torch")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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