import os
import pandas as pd

BASE = "/home/hpc/v121ca/v121ca21"
INP  = f"{BASE}/thesis_results/deberta/deberta_contarga_eval_with_labels.csv"
OUTD = f"{BASE}/thesis_results/mapped"
OUT  = f"{OUTD}/deberta_contarga_eval_mapped.csv"
os.makedirs(OUTD, exist_ok=True)

# Map GoEmotions -> CONTARGA 8 labels (simple baseline mapping)
MAP = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "nervousness": "fear",
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "love": "joy",
    "pride": "pride",
    "relief": "relief",
    "sadness": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "surprise": "surprise",
}

df = pd.read_csv(INP)

# map top1 predicted emotion into 8-label space; anything else -> "other"
df["pred_top1_mapped"] = df["pred_top1_emotion"].map(MAP).fillna("other")

df.to_csv(OUT, index=False)
print("Saved:", OUT)
print(df["pred_top1_mapped"].value_counts().head(10))
