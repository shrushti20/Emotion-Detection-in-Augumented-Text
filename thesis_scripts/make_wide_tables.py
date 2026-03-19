import os
import pandas as pd

BASE = "/home/hpc/v121ca/v121ca21"
INP  = f"{BASE}/thesis_results/tables/contarga_per_class_table.csv"
OUTD = f"{BASE}/thesis_results/tables"
os.makedirs(OUTD, exist_ok=True)

df = pd.read_csv(INP)

# create one column that uniquely identifies each run
df["run"] = df["model"].astype(str) + " | " + df["setting"].astype(str)

def make_wide(metric: str, out_name: str):
    wide = (
        df.pivot_table(
            index="emotion",
            columns="run",
            values=metric,
            aggfunc="first"
        )
        .reset_index()
    )

    # optional: nicer order of emotions
    emotion_order = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]
    wide["emotion"] = pd.Categorical(wide["emotion"], categories=emotion_order, ordered=True)
    wide = wide.sort_values("emotion")

    out_path = f"{OUTD}/{out_name}"
    wide.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(wide)

make_wide("f1",        "contarga_per_class_wide_f1.csv")
make_wide("precision", "contarga_per_class_wide_precision.csv")
make_wide("recall",    "contarga_per_class_wide_recall.csv")
