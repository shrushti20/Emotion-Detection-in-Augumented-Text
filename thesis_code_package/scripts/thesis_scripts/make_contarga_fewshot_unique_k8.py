import pandas as pd

LABELS = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]
SRC = "/home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/contarga_emotion_subset.csv"
OUT = "/home/hpc/v121ca/v121ca21/thesis_results/llm/contarga/fewshot_unique_k8.txt"

df = pd.read_csv(SRC)
df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
df["text"] = df["text"].astype(str).str.replace("\n"," ").str.strip()

used_texts = set()
shots = []

for lab in LABELS:
    cand = df[df["emotion"] == lab].copy()
    cand = cand[~cand["text"].isin(used_texts)]
    if len(cand) == 0:
        raise RuntimeError(f"No UNIQUE examples left for label: {lab}")

    row = cand.iloc[0]
    t = row["text"]
    if len(t) > 350:
        t = t[:350].rstrip() + "..."

    used_texts.add(row["text"])
    shots.append(f'Text: "{t}"\nFINAL: {lab}\n')

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(shots).strip() + "\n")

print("Wrote:", OUT)
