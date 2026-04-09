#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to a trained RoBERTa model directory (contains config.json, model.safetensors, tokenizer files)",
    )
    parser.add_argument(
        "--test_csv",
        default="/home/hpc/v121ca/v121ca21/thesis_adapt/data/contarga/test.csv",
        help="Path to CONTARGA test split CSV (expects columns: text, emotion)",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/hpc/v121ca/v121ca21/thesis_adapt/results",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="Max sequence length for tokenization",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    run_name = os.path.basename(os.path.normpath(model_dir))
    out_csv = os.path.join(args.out_dir, f"{run_name}_test_preds.csv")
    out_txt = os.path.join(args.out_dir, f"{run_name}_report.txt")

    os.makedirs(args.out_dir, exist_ok=True)

    # Label space (CONTARGA-8)
    LABELS = ["anger", "disgust", "fear", "joy", "pride", "relief", "sadness", "surprise"]
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    df = pd.read_csv(args.test_csv)
    if "text" not in df.columns or "emotion" not in df.columns:
        raise ValueError(f"{args.test_csv} must contain columns: text, emotion")

    # keep only supported labels
    df = df[df["emotion"].isin(label2id)].copy()
    df["labels"] = df["emotion"].map(label2id).astype(int)

    ds = Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tok)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds_tok = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    trainer = Trainer(model=model, data_collator=data_collator)

    pred = trainer.predict(ds_tok)
    logits = pred.predictions
    y_true = pred.label_ids
    y_pred = np.argmax(logits, axis=1)

    # Save per-row predictions
    out = df.copy()
    out["pred_id"] = y_pred
    out["pred_label"] = [id2label[int(i)] for i in y_pred]
    out.to_csv(out_csv, index=False)

    # Save report + summary metrics
    rep = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    with open(out_txt, "w") as f:
        f.write(f"Model: {model_dir}\n")
        f.write(f"Test CSV: {args.test_csv}\n\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Macro-F1:  {macro_f1:.6f}\n\n")
        f.write(rep + "\n")

    print("Saved predictions:", out_csv)
    print("Saved report:", out_txt)
    print(f"Accuracy: {acc:.6f}")
    print(f"Macro-F1:  {macro_f1:.6f}")
    print(rep)


if __name__ == "__main__":
    main()
