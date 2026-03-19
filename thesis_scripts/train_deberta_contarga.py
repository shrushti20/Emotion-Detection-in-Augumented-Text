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

label2id = {lab: i for i, lab in enumerate(LABELS)}
id2label = {i: lab for lab, i in label2id.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
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
    if x is None:
        return None
    return str(x).strip().lower()


def main():
    print("MAIN STARTED (DeBERTa CONTARGA)", flush=True)

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
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="emotion")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def preprocess(batch):
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

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

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

    metrics_path = os.path.join(args.out_dir, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in sorted(test_metrics.items()):
            f.write(f"{k}\t{v}\n")

    print(f"Saved test metrics to: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()