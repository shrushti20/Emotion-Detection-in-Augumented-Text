import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------- PATHS (adapt only if your username/path is different) ----------
BASE_DIR   = "/home/hpc/v121ca/v121ca21"
DATA_PATH  = f"{BASE_DIR}/thesis_data/contarga_llm/contarga_emotion_subset.csv"

RESULTS_DIR = f"{BASE_DIR}/thesis_results/deberta"
OUT_PATH    = f"{RESULTS_DIR}/contarga_eval_probs.csv"

# Your fine-tuned DeBERTa checkpoint directory:
MODEL_DIR  = f"{BASE_DIR}/thesis_models/deberta_goemo/checkpoint-5000"

# Tokenizer base model (should match the architecture you fine-tuned):
MODEL_NAME = "microsoft/deberta-v3-base"

TEXT_COL   = "text"   # change if your CSV uses a different name
BATCH_SIZE = 4
MAX_LEN    = 128
# -------------------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading model from:", args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    df = pd.read_csv(DATA_PATH)
    texts = df[TEXT_COL].astype(str).tolist()

    all_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)

    prob_cols = [f"p_{i}" for i in range(probs.shape[1])]
    out_df = pd.concat([df.reset_index(drop=True),
                        pd.DataFrame(probs, columns=prob_cols)], axis=1)

    out_df.to_csv(OUT_PATH, index=False)
    print("Saved →", OUT_PATH)


if __name__ == "__main__":
    main()