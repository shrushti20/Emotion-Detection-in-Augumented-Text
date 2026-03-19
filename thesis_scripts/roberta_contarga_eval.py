import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------- PATHS (adapt only if your username/path is different) ----------
BASE_DIR   = "/home/hpc/v121ca/v121ca21"
DATA_PATH  = f"{BASE_DIR}/thesis_data/contarga_llm/contarga_emotion_subset.csv"
RESULTS_DIR = f"{BASE_DIR}/thesis_results/roberta"
OUT_PATH = f"{RESULTS_DIR}/contarga_eval_probs.csv"

MODEL_DIR  = f"{BASE_DIR}/thesis_models/goemo_roberta_base/checkpoint-1500"
MODEL_NAME = "roberta-base"  # tokenizer base model
TEXT_COL   = "text"          # change to the correct column name if needed
BATCH_SIZE = 16
MAX_LEN    = 256
# -------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading model from:", MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print("Loading data from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if TEXT_COL not in df.columns:
        raise ValueError(
            f"TEXT_COL='{TEXT_COL}' not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )

    texts = df[TEXT_COL].astype(str).tolist()
    all_probs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch_texts = texts[start:start + BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(device)

            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()  # multi-label probs
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    print("Probability matrix shape:", probs.shape)

    # build column names p_0, p_1, ... or use label names if you have them
    num_labels = probs.shape[1]
    prob_cols = [f"p_{i}" for i in range(num_labels)]
    probs_df = pd.DataFrame(probs, columns=prob_cols)

    out_df = pd.concat([df.reset_index(drop=True), probs_df], axis=1)
    out_df.to_csv(OUT_PATH, index=False)

    print("Saved evaluation results →", OUT_PATH)


if __name__ == "__main__":
    main()
