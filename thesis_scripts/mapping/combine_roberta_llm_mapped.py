import os
import pandas as pd

BASE_DIR = "/home/hpc/v121ca/v121ca21"

ROBERTA_MAPPED_PATH = os.path.join(
    BASE_DIR,
    "thesis_results/mapped/roberta_contarga_eval_mapped.csv"
)
LLM_PATH = os.path.join(
    BASE_DIR,
    "thesis_results/llm/gemma_contarga_eval.csv"
)
OUT_DIR = os.path.join(BASE_DIR, "thesis_results/mapped")
OUT_PATH = os.path.join(OUT_DIR, "roberta_vs_llm_contarga_mapped.csv")

os.makedirs(OUT_DIR, exist_ok=True)

EMOTIONS = ["anger", "disgust", "fear", "joy", "pride", "relief", "sadness", "surprise"]


if __name__ == "__main__":
    print("Loading mapped RoBERTa:", ROBERTA_MAPPED_PATH)
    df_r = pd.read_csv(ROBERTA_MAPPED_PATH)
    print("RoBERTa mapped shape:", df_r.shape)

    print("Loading LLM results:", LLM_PATH)
    df_l = pd.read_csv(LLM_PATH)
    print("LLM shape:", df_l.shape)

    # Align lengths: LLM was run on a 300-row subset, RoBERTa on all 3020 rows.
    n = len(df_l)
    if len(df_r) < n:
        raise ValueError("RoBERTa file has fewer rows than LLM file, cannot align.")
    if len(df_r) > n:
        print(f"Truncating RoBERTa from {len(df_r)} to {n} rows to match LLM.")
        df_r = df_r.iloc[:n].reset_index(drop=True)
    df_l = df_l.reset_index(drop=True)

    # Optional sanity check on text
    if "text" in df_r.columns and "text" in df_l.columns:
        mismatch = (df_r["text"] != df_l["text"]).sum()
        if mismatch > 0:
            print(f"WARNING: {mismatch} rows have different 'text' between RoBERTa and LLM.")

    # Build LLM predictions from llm_anger, llm_disgust, ...
    for e in EMOTIONS:
        col = f"llm_{e}"
        if col not in df_l.columns:
            raise ValueError(f"Expected column '{col}' not found in LLM CSV.")

    def extract_llm_preds(row):
        active = [e for e in EMOTIONS if row[f"llm_{e}"] >= 0.5]
        if not active:
            return "none", ""
        return active[0], ", ".join(active)

    llm_top = []
    llm_all = []
    for _, row in df_l.iterrows():
        top, all_ = extract_llm_preds(row)
        llm_top.append(top)
        llm_all.append(all_)

    df_comb = pd.DataFrame({
        "text": df_r.get("text", df_l.get("text")),
        "emotion": df_r.get("emotion", df_l.get("emotion")),
        "convincingness": df_r.get("convincingness", df_l.get("convincingness")),
        "roberta_top1_28": df_r["roberta_top1_28"],
        "roberta_top3_28": df_r["roberta_top3_28"],
        "roberta_top1_8": df_r["roberta_top1_8"],
        "roberta_top3_8": df_r["roberta_top3_8"],
        "llm_pred": llm_top,
        "llm_pred_all": llm_all,
    })

    print("Combined shape:", df_comb.shape)
    df_comb.to_csv(OUT_PATH, index=False)
    print(f"Saved mapped combined file → {OUT_PATH}")
