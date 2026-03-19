import os
import pandas as pd

BASE_DIR = "/home/hpc/v121ca/v121ca21"
IN_PATH = os.path.join(
    BASE_DIR,
    "thesis_results/mapped/roberta_vs_llm_contarga_mapped.csv"
)

EMOTIONS = ["anger", "disgust", "fear", "joy", "pride", "relief", "sadness", "surprise"]


if __name__ == "__main__":
    print("Loading mapped combined file:", IN_PATH)
    df = pd.read_csv(IN_PATH)
    print("Shape:", df.shape)

    if not {"emotion", "roberta_top1_8", "llm_pred"}.issubset(df.columns):
        raise ValueError("Missing required columns in mapped combined CSV.")

    y_gold = df["emotion"].astype(str)
    y_r = df["roberta_top1_8"].astype(str)
    y_l = df["llm_pred"].astype(str)

    # Optionally restrict to the 8 target emotions
    mask_valid = y_gold.isin(EMOTIONS)
    y_gold = y_gold[mask_valid]
    y_r = y_r[mask_valid]
    y_l = y_l[mask_valid]

    roberta_gold = (y_r == y_gold).mean()
    llm_gold = (y_l == y_gold).mean()
    roberta_llm = (y_r == y_l).mean()

    print(f"Mapped RoBERTa (8-label) matches gold: {roberta_gold:.3f}")
    print(f"LLM matches gold (8-label space)      : {llm_gold:.3f}")
    print(f"RoBERTa (8) = LLM                     : {roberta_llm:.3f}")
