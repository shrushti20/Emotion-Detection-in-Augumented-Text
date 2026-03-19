import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./goemo_roberta_base"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3


def main():
    ds = load_dataset("go_emotions", "simplified")
    label_names = ds["train"].features["labels"].feature.names
    num_labels = len(label_names)

    print("Number of labels:", num_labels)
    print("Sample labels:", label_names[:10])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode_batch(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

        batch_size = len(batch["text"])
        mh = np.zeros((batch_size, num_labels), dtype="float32")

        for i, label_ids in enumerate(batch["labels"]):
            for lid in label_ids:
                mh[i, lid] = 1.0

        enc["labels"] = mh
        return enc

    encoded = ds.map(encode_batch, batched=True, remove_columns=["text"])
    encoded.set_format(type="torch")
    print(encoded)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits))
        preds = (probs > 0.5).int().numpy()

        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
        return {"macro_f1": macro_f1, "micro_f1": micro_f1}

    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"].to(model.device).float()
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.logits
            loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
            if return_outputs:
                return loss, outputs
            return loss

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        logging_steps=100,
        report_to=[],
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_results = trainer.evaluate(encoded["test"])
    print("Test results:", test_results)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved model and tokenizer to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
