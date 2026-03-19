import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

LABELS = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]

def compute_summary(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    return acc, p, r, f1

def compute_per_class(y_true, y_pred):
    # returns dict with per-class precision/recall/f1/support
    rep = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    # Keep only the 8 labels, and keep support as int
    rows = []
    for lab in LABELS:
        rows.append({
            "emotion": lab,
            "precision": rep[lab]["precision"],
            "recall": rep[lab]["recall"],
            "f1": rep[lab]["f1-score"],
            "support": int(rep[lab]["support"]),
        })
    return pd.DataFrame(rows)

def read_encoder(path, n=3020):
    df = pd.read_csv(path).head(n).reset_index(drop=True)

    # Your roberta file uses these columns
    # Your deberta mapping uses pred_top1_mapped
    for col in [
        "pred_top1_mapped",
        "roberta_top1_8",
        "deberta_top1_8",
        "pred_8",
        "mapped_pred",
        "deberta_pred_8",
        "roberta_pred_8",
    ]:
        if col in df.columns:
            return df["emotion"].astype(str), df[col].astype(str)

    raise ValueError(f"No mapped prediction column found in {path}. Columns={df.columns.tolist()}")

def read_llm(path, n=3020):
    df = pd.read_csv(path).head(n).reset_index(drop=True)
    llm_cols = [c for c in df.columns if c.startswith("llm_")]
    if not llm_cols:
        raise ValueError(f"No llm_* columns found in {path}. Columns={df.columns.tolist()}")

    pred = df[llm_cols].idxmax(axis=1).str.replace("llm_", "", regex=False)
    return df["emotion"].astype(str), pred.astype(str)

def main():
    BASE = "/home/hpc/v121ca/v121ca21"
    N = 3020  # set 300 if you want to match the earlier 300-row LLM comparison

    summary_rows = []
    per_class_rows = []

    encoders = [
        ("RoBERTa (GoEmo→CONTARGA)", "supervised-transfer",
         f"{BASE}/thesis_results/mapped/roberta_contarga_eval_mapped.csv"),
        ("DeBERTa (GoEmo→CONTARGA)", "supervised-transfer",
         f"{BASE}/thesis_results/mapped/deberta_contarga_eval_mapped.csv"),
    ]

    llms = [
        ("Mistral-7B-Instruct", "zero-shot", f"{BASE}/thesis_results/llm/contarga/mistral_zero.csv"),
        ("Mistral-7B-Instruct", "few-shot",  f"{BASE}/thesis_results/llm/contarga/mistral_few.csv"),
        ("Mistral-7B-Instruct", "cot",       f"{BASE}/thesis_results/llm/contarga/mistral_cot.csv"),
    ]

    # Encoders
    for model, setting, path in encoders:
        if not os.path.exists(path):
            print(f"[WARN] Missing encoder file: {path}")
            continue

        y_true, y_pred = read_encoder(path, n=N)
        acc, mp, mr, mf1 = compute_summary(y_true, y_pred)

        summary_rows.append({
            "model": model, "setting": setting,
            "acc": acc, "macro_p": mp, "macro_r": mr, "macro_f1": mf1
        })

        pc = compute_per_class(y_true, y_pred)
        pc.insert(0, "setting", setting)
        pc.insert(0, "model", model)
        per_class_rows.append(pc)

    # LLMs
    for model, setting, path in llms:
        if not os.path.exists(path):
            print(f"[WARN] Missing LLM file: {path}")
            continue

        y_true, y_pred = read_llm(path, n=N)
        acc, mp, mr, mf1 = compute_summary(y_true, y_pred)

        summary_rows.append({
            "model": model, "setting": setting,
            "acc": acc, "macro_p": mp, "macro_r": mr, "macro_f1": mf1
        })

        pc = compute_per_class(y_true, y_pred)
        pc.insert(0, "setting", setting)
        pc.insert(0, "model", model)
        per_class_rows.append(pc)

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()

    out_dir = f"{BASE}/thesis_results/tables"
    os.makedirs(out_dir, exist_ok=True)

    summary_path = f"{out_dir}/contarga_summary_table.csv"
    perclass_path = f"{out_dir}/contarga_per_class_table.csv"

    summary_df.to_csv(summary_path, index=False)
    per_class_df.to_csv(perclass_path, index=False)

    print("Saved:", summary_path)
    print(summary_df)
    print("\nSaved:", perclass_path)
    print(per_class_df.head(16))  # show first couple of blocks

if __name__ == "__main__":
    main()
