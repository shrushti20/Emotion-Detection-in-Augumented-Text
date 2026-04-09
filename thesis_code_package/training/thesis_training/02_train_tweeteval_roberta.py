"""
Script: 02_train_tweeteval_roberta.py

Purpose:
Train RoBERTa-base on the TweetEval Emotion dataset for single-label
emotion classification.

Dataset:
    TweetEval Emotion

Task type:
    Single-label multiclass classification

Outputs:
    - trained model checkpoints
    - final saved model and tokenizer in OUTPUT_DIR
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ---------------- CONFIG ----------------
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./ait_roberta_base"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
# ----------------------------------------


def main():
    # Load TweetEval Emotion dataset
    tweetemo = load_dataset("tweet_eval", "emotion")
    print(tweetemo)
    print(tweetemo["train"][0])

    # Build label mappings from dataset metadata
    label_names = tweetemo["train"].features["label"].names
    print("Labels:", label_names)

    num_labels = len(label_names)
    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load RoBERTa for multiclass classification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize_batch(batch):
        """Tokenize input text."""
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    # Tokenize dataset and rename label column to the Trainer-expected name
    encoded = tweetemo.map(tokenize_batch, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def compute_metrics(eval_pred):
        """Compute accuracy and macro-F1 for single-label classification."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
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

    print(f"Saving model and tokenizer to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()