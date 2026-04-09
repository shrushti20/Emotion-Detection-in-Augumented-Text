#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["anger", "joy", "optimism", "sadness"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def load_csv(path: str):
    df = pd.read_csv(path)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df = df[df["emotion"].isin(LABELS)].copy()
    df["label"] = df["emotion"].map(label2id).astype(int)
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)


def main():
    print("MAIN STARTED (DeBERTa)", flush=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = load_csv(args.train)
    valid_ds = load_csv(args.valid)
    test_ds = load_csv(args.test)

    print(f"Loaded sizes: train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

    train_ds = train_ds.map(tok, batched=True)
    valid_ds = valid_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    base_kwargs = dict(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(args.batch, 16),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    try:
        training_args = TrainingArguments(
            **base_kwargs,
            eval_strategy="epoch",
            save_strategy="epoch",
        )
    except TypeError:
        training_args = TrainingArguments(
            **base_kwargs,
            eval_strategy="epoch",
            save_strategy="epoch",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    print("=== TRAINING DeBERTa on TweetEval ===", flush=True)
    trainer.train()

    print("=== EVALUATING on TweetEval test ===", flush=True)
    results = trainer.evaluate(test_ds)
    print(results, flush=True)

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Saved model to:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
