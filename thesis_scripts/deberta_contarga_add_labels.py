import os
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification

BASE_DIR = "/home/hpc/v121ca/v121ca21"

# DeBERTa checkpoint (NOT RoBERTa)
MODEL_DIR = f"{BASE_DIR}/thesis_models/deberta_goemo/checkpoint-5000"

# Input/output
IN_PATH = f"{BASE_DIR}/thesis_results/deberta/contarga_eval_probs.csv"
OUT_PATH = f"{BASE_DIR}/thesis_results/deberta/deberta_contarga_eval_with_labels.csv"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print("Loading model config from:", MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
id2label = model.config.id2label   # {0: 'admiration', ...}
num_labels = model.config.num_labels
print("num_labels:", num_labels)
print("id2label:", id2label)

print("Loading CSV:", IN_PATH)
df = pd.read_csv(IN_PATH)
print("Input shape:", df.shape)

# collect probability columns p_0 ... p_27
prob_cols = [f"p_{i}" for i in range(num_labels)]
missing = [c for c in prob_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing prob columns in CSV: {missing[:5]} ... (total {len(missing)})")

probs = df[prob_cols].values

# ---- TOP 1 ----
top1_idx = probs.argmax(axis=1)
df["pred_top1_idx"] = top1_idx
df["pred_top1_emotion"] = [id2label[i] for i in top1_idx]

# ---- TOP 3 ----
top3_idx = probs.argsort(axis=1)[:, -3:][:, ::-1]
df["pred_top3_idx"] = [";".join(map(str, row)) for row in top3_idx]
df["pred_top3_emotions"] = [", ".join(id2label[i] for i in row) for row in top3_idx]

# ---- SIMPLE METRIC: gold emotion in top-3? ----
if "emotion" in df.columns:
    gold = df["emotion"].astype(str)
    in_top3 = []
    for g, row in zip(gold, top3_idx):
        pred_emos = [id2label[i] for i in row]
        in_top3.append(g in pred_emos)
    top3_acc = sum(in_top3) / len(in_top3)
    print(f"Gold emotion in top-3 prediction: {top3_acc:.3f} ({top3_acc*100:.1f}%)")

df.to_csv(OUT_PATH, index=False)
print("Saved →", OUT_PATH)
