#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

INP = Path("/home/hpc/v121ca/v121ca21/thesis_results/tables/domain_adaptation_summary.csv")
OUT_DIR = Path("/home/hpc/v121ca/v121ca21/thesis_results/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP)

# helper formatting
def fmt(x, d=3):
    return f"{x:.{d}f}"

df["accuracy"] = df["accuracy"].astype(float)
df["macro_f1"] = df["macro_f1"].astype(float)
df["n_eval"] = df["n_eval"].astype(int)

# ---- Table 1: GoEmotions -> CONTARGA (8-label mapped) ----
t1 = df[(df["source"]=="GoEmotions") & (df["target"]=="CONTARGA") & (df["eval_labels"]=="8-label CONTARGA (mapped)")].copy()
t1 = t1[["model","accuracy","macro_f1","n_eval"]].sort_values("model")
t1["accuracy"] = t1["accuracy"].map(fmt)
t1["macro_f1"] = t1["macro_f1"].map(fmt)

t1_csv = OUT_DIR / "table_goemotions_to_contarga_8label.csv"
t1.to_csv(t1_csv, index=False)

# ---- Table 2: TweetEval -> CONTARGA (aligned labels) ----
t2 = df[(df["source"]=="TweetEval") & (df["target"]=="CONTARGA")].copy()
t2 = t2[["model","eval_labels","accuracy","macro_f1","n_eval"]].sort_values("model")
t2["accuracy"] = t2["accuracy"].map(fmt)
t2["macro_f1"] = t2["macro_f1"].map(fmt)

t2_csv = OUT_DIR / "table_tweeteval_to_contarga_aligned.csv"
t2.to_csv(t2_csv, index=False)

# ---- LaTeX outputs (Overleaf-ready) ----
def to_latex(df_, caption, label):
    # minimal, clean latex table
    return df_.to_latex(index=False, escape=False, caption=caption, label=label)

t1_tex = OUT_DIR / "table_goemotions_to_contarga_8label.tex"
t1_tex.write_text(to_latex(t1, "GoEmotions → CONTARGA (8-label mapped).", "tab:goemotions_contarga_8"))

t2_tex = OUT_DIR / "table_tweeteval_to_contarga_aligned.tex"
t2_tex.write_text(to_latex(t2, "TweetEval → CONTARGA (aligned labels: anger,fear,joy,pride; love→{joy,pride}).", "tab:tweeteval_contarga_aligned"))

print("Wrote:")
print(" -", t1_csv)
print(" -", t2_csv)
print(" -", t1_tex)
print(" -", t2_tex)
