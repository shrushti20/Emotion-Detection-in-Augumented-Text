"""
Script: train_deberta_contarga.py

Purpose:
Fine-tune a DeBERTa model directly on the CONTARGA dataset for
8-label single-label emotion classification.

This script:
- loads train/validation/test CSV files
- normalizes emotion labels
- tokenizes argumentative text
- fine-tunes DeBERTa on CONTARGA
- evaluates the best checkpoint on validation and test sets
- saves final test metrics to the output directory

Inputs:
    --train      Path to CONTARGA training CSV
    --valid      Path to CONTARGA validation CSV
    --test       Path to CONTARGA test CSV
    --out_dir    Directory for checkpoints and saved metrics

Expected columns:
    - text column (default: "text")
    - label column (default: "emotion")

Outputs:
    - trained checkpoints in out_dir
    - test_metrics.txt with final evaluation results
"""

import os
import argparse
import random
import numpy as np
import torch

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# Fixed CONTARGA label space used in thesis experiments
LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "pride",
    "relief",
    "sadness",
    "surprise",
]

# Label lookup dictionaries for model training and readable outputs
label2id = {lab: i for i, lab in enumerate(LABELS)}
id2label = {i: lab for lab, i in label2id.items()}


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for single-label classification.

    Metrics:
    - accuracy
    - macro precision
    - macro recall
    - macro F1
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }


def normalize_label(x):
    """
    Normalize label text by stripping whitespace and lowercasing.
    """
    if x is None:
        return None
    return str(x).strip().lower()


def main():
    print("MAIN STARTED (DeBERTa CONTARGA)", flush=True)

    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to training CSV")
    ap.add_argument("--valid", required=True, help="Path to validation CSV")
    ap.add_argument("--test", required=True, help="Path to test CSV")
    ap.add_argument("--out_dir", required=True, help="Directory to save model outputs")

    ap.add_argument("--model_name", default="microsoft/deberta-v3-base", help="Base model name")
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    ap.add_argument("--batch", type=int, default=8, help="Per-device batch size")
    ap.add_argument("--max_len", type=int, default=128, help="Maximum sequence length")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--text_col", default="text", help="Name of text column in input CSV")
    ap.add_argument("--label_col", default="emotion", help="Name of label column in input CSV")
    args = ap.parse_args()

    # Prepare output directory and random seeds
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Load train/validation/test CSV files as a Hugging Face DatasetDict
    data_files = {
        "train": args.train,
        "validation": args.valid,
        "test": args.test,
    }

    ds = load_dataset("csv", data_files=data_files)
    print(
        f"Loaded dataset sizes: train={len(ds['train'])}, "
        f"valid={len(ds['validation'])}, test={len(ds['test'])}",
        flush=True,
    )

    # Load tokenizer for the selected DeBERTa model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(batch):
        """
        Tokenize text and convert string emotion labels into numeric IDs.
        """
        texts = batch[args.text_col]
        labels_raw = batch[args.label_col]

        labels = []
        for lab in labels_raw:
            norm_lab = normalize_label(lab)
            if norm_lab not in label2id:
                raise ValueError(
                    f"Unknown label '{lab}' after normalization -> '{norm_lab}'. "
                    f"Expected one of: {LABELS}"
                )
            labels.append(label2id[norm_lab])

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_len,
        )
        tokenized["labels"] = labels
        return tokenized

    # Tokenize all dataset splits and remove original CSV columns
    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    # Dynamically pad batches at runtime
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load DeBERTa sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # Configure Hugging Face training settings
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(args.batch, 8),
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # Initialize Trainer for supervised fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("=== TRAINING DeBERTa on CONTARGA ===", flush=True)
    trainer.train()

    print("=== VALIDATION (best checkpoint) ===", flush=True)
    valid_metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    print(valid_metrics, flush=True)

    print("=== TEST (best checkpoint) ===", flush=True)
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    print(test_metrics, flush=True)

    # Save final test metrics in a plain text file for later reporting
    metrics_path = os.path.join(args.out_dir, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in sorted(test_metrics.items()):
            f.write(f"{k}\t{v}\n")

    print(f"Saved test metrics to: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()