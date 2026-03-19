import pandas as pd

BASE = "/home/hpc/v121ca/v121ca21"
PATH = f"{BASE}/thesis_results/combined/roberta_vs_llm_contarga.csv"

df = pd.read_csv(PATH)

acc_roberta = (df["pred_top1_emotion"] == df["emotion"]).mean()
acc_llm     = (df["llm_pred"] == df["emotion"]).mean()
agree       = (df["pred_top1_emotion"] == df["llm_pred"]).mean()

print(f"RoBERTa matches gold: {acc_roberta:.3f}")
print(f"LLM matches gold    : {acc_llm:.3f}")
print(f"RoBERTa = LLM        : {agree:.3f}")
print(f"RoBERTa = LLM (full precision): {(df['pred_top1_emotion'] == df['llm_pred']).mean():.6f}")
