#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INP = Path("/home/hpc/v121ca/v121ca21/thesis_results/tables/domain_adaptation_summary.csv")
OUT_DIR = Path("/home/hpc/v121ca/v121ca21/thesis_results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP)
df["macro_f1"] = df["macro_f1"].astype(float)

def barplot(sub, title, out_png, out_pdf):
    sub = sub.sort_values("model")
    x = range(len(sub))
    plt.figure()
    plt.bar(x, sub["macro_f1"])
    plt.xticks(x, sub["model"], rotation=15, ha="right")
    plt.ylabel("Macro-F1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()

# GoEmotions -> CONTARGA (8-label)
g = df[(df["source"]=="GoEmotions") & (df["target"]=="CONTARGA") & (df["eval_labels"]=="8-label CONTARGA (mapped)")].copy()
barplot(
    g,
    "GoEmotions → CONTARGA (8-label mapped)",
    OUT_DIR / "goemotions_to_contarga_8label_macro_f1.png",
    OUT_DIR / "goemotions_to_contarga_8label_macro_f1.pdf",
)

# TweetEval -> CONTARGA (aligned)
t = df[(df["source"]=="TweetEval") & (df["target"]=="CONTARGA")].copy()
barplot(
    t,
    "TweetEval → CONTARGA (aligned labels)",
    OUT_DIR / "tweeteval_to_contarga_aligned_macro_f1.png",
    OUT_DIR / "tweeteval_to_contarga_aligned_macro_f1.pdf",
)

print("Wrote plots to:", OUT_DIR)
