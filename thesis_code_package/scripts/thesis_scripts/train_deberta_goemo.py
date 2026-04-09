"""
Script: train_deberta_goemo.py

Purpose:
Train DeBERTa on the GoEmotions dataset for multi-label emotion classification.

Notes:
- This version uses a reduced subset of GoEmotions for faster experimentation.
- It appears to be an earlier / lightweight training variant compared with the
  cleaner final training script in thesis_training.

Dataset:
    GoEmotions

Task type:
    Multi-label classification

Outputs:
    - trained checkpoints in OUT_DIR
    - saved tokenizer in OUT_DIR
"""

import os
import numpy as np
from datasets import load_dataset
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from sklearn.metrics import f1_score
import torch

# ---------------- CONFIG ----------------
MODEL_NAME = "microsoft/deberta-v3-base"
OUT_DIR = "/home/hpc/v121ca/v121ca21/thesis_models/deberta_goemo"
MAX_LEN = 128
BATCH_SIZE = 4          # conservative batch size for DeBERTa
EPOCHS = 1              # reduced run for quick experimentation
LR = 2e-5
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.environ["WANDB_DISABLED"] = "true"


def data_collator(features):
    """
    Use the default Hugging Face collator, but ensure labels are float tensors.
    This is required for BCEWithLogitsLoss in multi-label classification.
    """
    batch = default_data_collator(features)
    if "labels" in batch:
        batch["labels"] = batch["labels"].float()
    return batch


print("Loading GoEmotions dataset...")
raw = load_dataset("go_emotions")

# Use a smaller subset for faster experimentation
raw["train"] = raw["train"].select(range(5000))
raw["validation"] = raw["validation"].select(range(1000))
raw["test"] = raw["test"].select(range(1000))

label_names = raw["train"].features["labels"].feature.names
num_labels = len(label_names)

id2label = {i: l for i, l in enumerate(label_names)}
label2id = {l: i for i, l in enumerate(label_names)}

print("Number of labels:", num_labels)

tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)


def preprocess(batch):
    """
    Tokenize text and convert label lists into multi-hot vectors.
    """
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

    labels = []
    for labs in batch["labels"]:
        vec = np.zeros(num_labels, dtype=np.float32)
        vec[labs] = 1.0
        labels.append(vec.tolist())

    enc["labels"] = labels
    return enc


print("Tokenizing...")
encoded = raw.map(preprocess, batched=True, remove_columns=["text"])
encoded.set_format("torch")

# Load DeBERTa for multi-label classification
model = DebertaV2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
)


def compute_metrics(eval_pred):
    """
    Compute macro-F1 for multi-label predictions.
    """
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


# Configure training settings
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=True,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

print("Evaluating on test set...")
metrics = trainer.evaluate(encoded["test"])
print(metrics)

print("Saving model + tokenizer...")
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("Done. Model saved to:", OUT_DIR)