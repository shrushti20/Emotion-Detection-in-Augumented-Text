import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

BASE = "/home/hpc/v121ca/v121ca21"
IN_PATH = f"{BASE}/thesis_results/llm/gemma_contarga_eval.csv"
OUT_PATH = f"{BASE}/thesis_results/llm/gemma_contarga_metrics.txt"

df = pd.read_csv(IN_PATH)

# gold labels
gold = df["emotion"].astype(str)

# LLM prediction = any emotion with value 1
emotion_cols = ["llm_anger","llm_disgust","llm_fear","llm_joy",
                "llm_pride","llm_relief","llm_sadness","llm_surprise"]

preds = df[emotion_cols].idxmax(axis=1).str.replace("llm_","")

acc = accuracy_score(gold, preds)
f1 = f1_score(gold, preds, average="macro")
report = classification_report(gold, preds)

with open(OUT_PATH, "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Macro-F1: {f1:.4f}\n\n")
    f.write(report)

print("Saved metrics →", OUT_PATH)
