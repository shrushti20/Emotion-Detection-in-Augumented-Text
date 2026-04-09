import pandas as pd
from pathlib import Path

IN_PATH  = "/home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/contarga_emotion_subset.csv"
OUT_DIR  = Path("/home/hpc/v121ca/v121ca21/thesis_adapt/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# choose your label set (8-label pilot)
KEEP = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]

N_PER_LABEL = 10     # your supervisor suggestion
SEED = 42

df = pd.read_csv(IN_PATH)

# keep only 8-label pilot rows
df = df[df["emotion"].isin(KEEP)].copy()

# shuffle
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# sample N per label for train
train = (
    df.groupby("emotion", group_keys=False)
      .apply(lambda x: x.sample(n=min(N_PER_LABEL, len(x)), random_state=SEED))
      .reset_index(drop=True)
)

# remaining for dev/test
rest = df.drop(train.index, errors="ignore").reset_index(drop=True)

# simple dev/test split
dev = rest.sample(frac=0.1, random_state=SEED)
test = rest.drop(dev.index).reset_index(drop=True)

train.to_csv(OUT_DIR / "contarga_8_train.csv", index=False)
dev.to_csv(OUT_DIR / "contarga_8_dev.csv", index=False)
test.to_csv(OUT_DIR / "contarga_8_test.csv", index=False)

print("Saved:")
print(" train:", train.shape, OUT_DIR / "contarga_8_train.csv")
print(" dev  :", dev.shape, OUT_DIR / "contarga_8_dev.csv")
print(" test :", test.shape, OUT_DIR / "contarga_8_test.csv")
