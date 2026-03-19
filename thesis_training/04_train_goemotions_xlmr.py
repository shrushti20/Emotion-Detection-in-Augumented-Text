import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


def filter_single_label(example):
    return len(example["labels"]) == 1


def squeeze_label(example):
    example["label"] = example["labels"][0]
    return example


def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def main():
    model_name = "xlm-roberta-base"
    output_dir = "./goemo_xlmr_base"

    print("Loading GoEmotions dataset...")
    goemo_raw = load_dataset("go_emotions")

    print("Filtering to single-label examples...")
    goemo_filtered = goemo_raw.filter(filter_single_label)
    goemo_filtered = goemo_filtered.map(squeeze_label)
    goemo_filtered = goemo_filtered.remove_columns(["labels"])

    label_names = goemo_raw["train"].features["labels"].feature.names
    num_labels = len(label_names)

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    print(f"Number of labels: {num_labels}")
    print(f"First 10 labels: {label_names[:10]}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    print("Tokenizing dataset...")
    encoded = goemo_filtered.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
    )
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        report_to="none",
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
    eval_results = trainer.evaluate(encoded["test"])
    print("Test results:", eval_results)

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    main()