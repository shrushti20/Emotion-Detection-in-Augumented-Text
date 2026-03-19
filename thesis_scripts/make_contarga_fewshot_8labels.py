import pandas as pd

LABELS = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]
SRC = "/home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/contarga_emotion_subset.csv"
OUT = "/home/hpc/v121ca/v121ca21/thesis_results/llm/contarga/fewshot_balanced_k8.txt"

df = pd.read_csv(SRC)
df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()

shots = []
for lab in LABELS:
    rows = df[df["emotion"] == lab].head(1)
    if len(rows) == 0:
        raise RuntimeError(f"No examples found for label: {lab}")
    t = str(rows.iloc[0]["text"]).replace("\n"," ").strip()
    if len(t) > 350:
        t = t[:350].rstrip() + "..."
    shots.append(f'Text: "{t}"\nFINAL: {lab}\n')

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(shots).strip() + "\n")

print("Wrote:", OUT)
