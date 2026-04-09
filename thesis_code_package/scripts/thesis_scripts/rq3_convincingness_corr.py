#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

LABELS_8 = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]

def onehot_to_label(df, labels):
    score_cols = [f"llm_{l}" for l in labels]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing one-hot columns: {missing}\nAvailable columns: {list(df.columns)}")
    idx = df[score_cols].values.argmax(axis=1)
    return pd.Series([labels[i] for i in idx], index=df.index)

def safe_spearman(x, y):
    # drop NaNs
    m = ~(pd.isna(x) | pd.isna(y))
    x = x[m]
    y = y[m]
    if len(x) < 5:
        return np.nan, np.nan, int(len(x))
    rho, p = spearmanr(x, y)
    return float(rho), float(p), int(len(x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--setting", required=True, help="e.g., supervised, mistral_zero, zephyr_cot, tfidf_k8")
    ap.add_argument("--pred_col", default=None, help="predicted label column (for supervised)")
    ap.add_argument("--labels", default="8", choices=["8"], help="label set (fixed to 8-label CONTARGA here)")
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.infile)

    # required columns
    for col in ["convincingness", "emotion"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {args.infile}. Got: {list(df.columns)}")

    # get predicted labels
    if args.pred_col:
        if args.pred_col not in df.columns:
            raise ValueError(f"pred_col '{args.pred_col}' not found. Columns: {list(df.columns)}")
        pred = df[args.pred_col].astype(str).str.strip().str.lower()
    else:
        # assume LLM one-hot columns exist
        pred = onehot_to_label(df, LABELS_8).astype(str).str.strip().str.lower()

    df["pred_label"] = pred
    df["convincingness"] = pd.to_numeric(df["convincingness"], errors="coerce")

    # RQ3: association between each predicted emotion (binary) and convincingness
    rows = []
    for lab in LABELS_8:
        is_lab = (df["pred_label"] == lab).astype(int)
        rho, p, n = safe_spearman(is_lab, df["convincingness"])
        rows.append({
            "Model": args.model,
            "Setting": args.setting,
            "Emotion": lab,
            "Spearman_rho": None if np.isnan(rho) else round(rho, 4),
            "p_value": None if np.isnan(p) else round(p, 6),
            "n_used": n
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.outfile, index=False)
    print("Saved:", args.outfile)
    print(out.sort_values(["Model","Setting","Spearman_rho"], ascending=[True,True,False]).head(12))

if __name__ == "__main__":
    main()
