import os
import pandas as pd

BASE = "/home/hpc/v121ca/v121ca21"

# CORRECT paths
ROBERTA_PATH = f"{BASE}/thesis_data/contarga_llm/roberta_contarga_eval_with_labels.csv"
LLM_PATH     = f"{BASE}/thesis_results/llm/gemma_contarga_eval.csv"
OUT_DIR      = f"{BASE}/thesis_results/combined"
OUT_PATH     = f"{OUT_DIR}/roberta_vs_llm_contarga.csv"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading RoBERTa results from:", ROBERTA_PATH)
df_r = pd.read_csv(ROBERTA_PATH)
print("RoBERTa shape:", df_r.shape)

print("Loading LLM results from:", LLM_PATH)
df_l = pd.read_csv(LLM_PATH)
print("LLM shape:", df_l.shape)

# ----- build LLM predictions -----
emotion_cols = [
    "llm_anger","llm_disgust","llm_fear","llm_joy",
    "llm_pride","llm_relief","llm_sadness","llm_surprise"
]

# single best emotion = argmax over 0/1 indicators
df_l["llm_pred"] = df_l[emotion_cols].idxmax(axis=1).str.replace("llm_", "")

# all active emotions (in case there are multiple 1s)
df_l["llm_pred_all"] = df_l.apply(
    lambda row: ", ".join(
        col.replace("llm_", "") for col in emotion_cols if row[col] == 1
    ),
    axis=1
)

# ----- merge on (text, emotion, convincingness) so rows align correctly -----
merge_keys = ["text", "emotion", "convincingness"]
df_combined = df_r.merge(
    df_l[merge_keys + ["llm_pred", "llm_pred_all"]],
    on=merge_keys,
    how="inner",
)

print("Combined shape:", df_combined.shape)

# keep only the important columns
df_combined = df_combined[[
    "text",
    "emotion",               # gold CONTARGA emotion
    "convincingness",
    "pred_top1_emotion",     # RoBERTa top-1 (GoEmotions mapped)
    "pred_top3_emotions",
    "llm_pred",              # LLM single best label
    "llm_pred_all",          # LLM all active labels
]]

df_combined.to_csv(OUT_PATH, index=False)
print("Saved merged comparison →", OUT_PATH)
