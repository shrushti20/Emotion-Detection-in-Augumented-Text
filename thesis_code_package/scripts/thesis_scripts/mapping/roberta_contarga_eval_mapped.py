import os
import pandas as pd
from mapping_dict import GOEMO_TO_CONTARGA

BASE_DIR = "/home/hpc/v121ca/v121ca21"
IN_PATH = os.path.join(
    BASE_DIR,
    "thesis_data/contarga_llm/roberta_contarga_eval_with_labels.csv"
)
OUT_DIR = os.path.join(BASE_DIR, "thesis_results/mapped")
OUT_PATH = os.path.join(OUT_DIR, "roberta_contarga_eval_mapped.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def map_goemo_to_contarga(label: str) -> str:
    if pd.isna(label):
        return "neutral"
    label = str(label).strip()
    return GOEMO_TO_CONTARGA.get(label, "neutral")


def map_top3_string(s: str) -> str:
    if pd.isna(s):
        return ""
    items = [x.strip() for x in str(s).split(",")]
    mapped = []
    for lab in items:
        m = map_goemo_to_contarga(lab)
        # you can choose to keep or drop "neutral" here; I drop it for top-3 display
        if m not in mapped and m != "neutral":
            mapped.append(m)
    return ", ".join(mapped)


if __name__ == "__main__":
    print(f"Loading RoBERTa CSV from: {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    print("Input shape:", df.shape)

    if "pred_top1_emotion" not in df.columns:
        raise ValueError("Column 'pred_top1_emotion' not found in input CSV.")
    if "pred_top3_emotions" not in df.columns:
        raise ValueError("Column 'pred_top3_emotions' not found in input CSV.")

    # Copy core columns
    out = df.copy()

    # Map top-1 and top-3 predictions into 8-label space
    out["roberta_top1_28"] = out["pred_top1_emotion"]
    out["roberta_top1_8"] = out["pred_top1_emotion"].apply(map_goemo_to_contarga)
    out["roberta_top3_28"] = out["pred_top3_emotions"]
    out["roberta_top3_8"] = out["pred_top3_emotions"].apply(map_top3_string)

    # Quick diagnostic: accuracy of mapped RoBERTa vs gold
    if "emotion" in out.columns:
        acc = (out["roberta_top1_8"] == out["emotion"]).mean()
        print(f"Mapped RoBERTa top-1 (8 labels) vs gold accuracy: {acc:.3f}")

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved mapped RoBERTa results → {OUT_PATH}")
