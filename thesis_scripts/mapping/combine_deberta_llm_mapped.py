import os
import pandas as pd

BASE = "/home/hpc/v121ca/v121ca21"
DEB  = f"{BASE}/thesis_results/mapped/deberta_contarga_eval_mapped.csv"
LLM  = f"{BASE}/thesis_results/llm/mistral_contarga_multilabel_eval.csv"

OUTD = f"{BASE}/thesis_results/combined"
OUT  = f"{OUTD}/deberta_vs_llm_contarga_mapped.csv"
os.makedirs(OUTD, exist_ok=True)

df_d = pd.read_csv(DEB).head(300).reset_index(drop=True)
df_l = pd.read_csv(LLM).reset_index(drop=True)

# LLM prediction column depends on your script; adjust if needed:
# If your file is multi-hot (llm_anger, llm_disgust, ...), we take argmax over those.
llm_cols = [c for c in df_l.columns if c.startswith("llm_")]
if not llm_cols:
    raise ValueError("No llm_* columns found in LLM file.")

# pick top-1 LLM label (highest value)
df_l["llm_pred"] = df_l[llm_cols].idxmax(axis=1).str.replace("llm_", "", regex=False)

out = pd.DataFrame({
    "text": df_d["text"],
    "gold": df_d["emotion"],
    "deberta_pred_8": df_d["pred_top1_mapped"],
    "llm_pred_8": df_l["llm_pred"],
})

out.to_csv(OUT, index=False)
print("Saved:", OUT, "shape:", out.shape)
