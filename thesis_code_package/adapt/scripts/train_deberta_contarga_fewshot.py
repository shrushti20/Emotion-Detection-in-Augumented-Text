#!/usr/bin/env python3
"""
Few-shot / balanced adaptation for DeBERTa:
Fine-tune a GoEmotions-trained DeBERTa checkpoint on CONTARGA (8 labels),
evaluate on dev + test, and save the adapted model.

Expected files inside --data_dir:
  - train.csv  (columns: text, emotion)
  - dev.csv    (columns: text, emotion)
  - test.csv   (columns: text, emotion)
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score


LABELS = ["anger", "disgust", "fear", "joy", "pride", "relief", "sadness", "surprise"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_path", required=True, help="GoEmotions-trained DeBERTa checkpoint path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--save_total_limit", type=int, default=2)
    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_df(path: str, label2id: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "emotion" not in df.columns:
        raise ValueError(f"{path} must contain columns: text, emotion. Found: {list(df.columns)}")

    df = df[df["emotion"].isin(label2id.keys())].copy()
    df["labels"] = df["emotion"].map(label2id).astype(int)
    return df[["text", "labels"]]


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    y_pred = np.argmax(preds, axis=1)
    return {
        "acc": float(accuracy_score(labels, y_pred)),
        "macro_f1": float(f1_score(labels, y_pred, average="macro")),
    }


def main():
    args = parse_args()

    train_csv = os.path.join(args.data_dir, "train.csv")
    dev_csv   = os.path.join(args.data_dir, "dev.csv")
    test_csv  = os.path.join(args.data_dir, "test.csv")

    for p in [train_csv, dev_csv, test_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing expected file: {p}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"--model_path does not exist: {args.model_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    train_df = load_df(train_csv, label2id)
    dev_df   = load_df(dev_csv, label2id)
    test_df  = load_df(test_csv, label2id)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    dev_ds   = Dataset.from_pandas(dev_df, preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    dev_ds   = dev_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds  = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.problem_type = "single_label_classification"

    train_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    dev_metrics = trainer.evaluate(dev_ds)
    test_metrics = trainer.evaluate(test_ds)

    print("Best model eval on DEV:", dev_metrics)
    print("Final eval on TEST:", test_metrics)

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Saved adapted model to:", args.out_dir)


if __name__ == "__main__":
    main()


