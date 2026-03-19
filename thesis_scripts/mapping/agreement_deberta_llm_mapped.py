import pandas as pd

P = "/home/hpc/v121ca/v121ca21/thesis_results/combined/deberta_vs_llm_contarga_mapped.csv"
df = pd.read_csv(P)

def acc(a, b):
    return (a == b).mean()

print("DeBERTa(8) matches gold:", acc(df["deberta_pred_8"], df["gold"]))
print("LLM(8) matches gold    :", acc(df["llm_pred_8"], df["gold"]))
print("DeBERTa(8) = LLM(8)     :", acc(df["deberta_pred_8"], df["llm_pred_8"]))
