import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification

BASE_DIR   = "/home/hpc/v121ca/v121ca21"
MODEL_DIR  = f"{BASE_DIR}/thesis_models/goemo_roberta_base/checkpoint-1500"
IN_PATH    = f"{BASE_DIR}/thesis_data/contarga_llm/roberta_contarga_eval_hpc.csv"
RESULTS_DIR = f"{BASE_DIR}/thesis_results/roberta"
OUT_PATH = f"{RESULTS_DIR}/contarga_eval_with_labels.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)

# GoEmotions label names in correct order
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

print("Loading model config from:", MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
id2label = model.config.id2label   # keys are INTEGERS
num_labels = model.config.num_labels
print("num_labels:", num_labels)
print("id2label:", id2label)

assert num_labels == len(EMOTIONS), "Label mismatch: EMOTIONS list != model.num_labels"

print("Loading CSV:", IN_PATH)
df = pd.read_csv(IN_PATH)
print("Input shape:", df.shape)

# collect probability columns p_0 ... p_27
prob_cols = [f"p_{i}" for i in range(num_labels)]
probs = df[prob_cols].values

# ---- TOP 1 ----
top1_idx = probs.argmax(axis=1)                      # integer indices 0–27
df["pred_top1_idx"] = top1_idx
df["pred_top1_label"] = [id2label[i] for i in top1_idx]      # LABEL_3 format
df["pred_top1_emotion"] = [EMOTIONS[i] for i in top1_idx]    # 'anger', 'joy', ...

# ---- TOP 3 ----
top3_idx = probs.argsort(axis=1)[:, -3:][:, ::-1]      # highest 3
df["pred_top3_idx"] = [";".join(str(i) for i in row) for row in top3_idx]
df["pred_top3_emotions"] = [
    ", ".join(EMOTIONS[i] for i in row) for row in top3_idx
]

# ---- SIMPLE METRIC: gold emotion in top-3? ----
if "emotion" in df.columns:
    gold = df["emotion"].astype(str)
    in_top3 = []

    for g, row in zip(gold, top3_idx):
        pred_emos = [EMOTIONS[i] for i in row]
        in_top3.append(g in pred_emos)

    top3_acc = sum(in_top3) / len(in_top3)
    print(f"Gold emotion in top-3 prediction: {top3_acc:.3f} ({top3_acc*100:.1f}%)")

df.to_csv(OUT_PATH, index=False)
print("Saved with label names →", OUT_PATH)
