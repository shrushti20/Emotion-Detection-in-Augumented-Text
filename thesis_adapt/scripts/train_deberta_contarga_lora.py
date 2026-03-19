#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score

from peft import LoraConfig, get_peft_model, TaskType


LABELS = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}


def load_split(path_csv, tok, max_len):
    df = pd.read_csv(path_csv)
    df = df[df["emotion"].isin(label2id)].copy()
    df["labels"] = df["emotion"].map(label2id).astype(int)

    ds = Dataset.from_pandas(df[["text","labels"]], preserve_index=False)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=max_len)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds_tok


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"acc": acc, "macro_f1": macro_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing train.csv/dev.csv/test.csv")
    ap.add_argument("--model_path", required=True, help="Base checkpoint path")
    ap.add_argument("--out_dir", required=True, help="Where to save LoRA-adapted model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)  # LoRA often uses higher LR
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--save_total_limit", type=int, default=1)
    args = ap.parse_args()

    # check files
    for fn in ["train.csv","dev.csv","test.csv"]:
        p = os.path.join(args.data_dir, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing expected file: {p}")

    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    train_ds = load_split(os.path.join(args.data_dir, "train.csv"), tok, args.max_len)
    dev_ds   = load_split(os.path.join(args.data_dir, "dev.csv"), tok, args.max_len)
    test_ds  = load_split(os.path.join(args.data_dir, "test.csv"), tok, args.max_len)

    data_collator = DataCollatorWithPadding(tokenizer=tok)

    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    base.config.problem_type = "single_label_classification"


    # ---- LoRA config (works well for DeBERTa family in practice)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query_proj", "key_proj", "value_proj"],  # safe default for DeBERTa v2/v3
    )

    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out_dir,
        seed=args.seed,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=25,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    best_dev = trainer.evaluate(dev_ds)
    print("Best model eval on DEV:", best_dev)

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="eval")
    print("Final eval on TEST:", test_metrics)

    # Save adapter + tokenizer
    trainer.model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Saved LoRA-adapted model to:", args.out_dir)


if __name__ == "__main__":
    main()

