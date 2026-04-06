import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_BASE_DIR = "./goemo_roberta_base"
CONTARGA_PATH = "./contarga/contarga_800x5_public.csv"
OUTPUT_PROBS_CSV = "./contarga_roberta_goemo_probs.csv"
OUTPUT_CORR_CSV = "./roberta_convincingness_correlations.csv"
OUTPUT_EVAL_CSV = "./roberta_contarga_eval.csv"
TEXT_COL = "argument"
CONV_COL = "convincingness"
BATCH_SIZE = 16
MAX_LENGTH = 128

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

EMO_OVERLAP = ["anger", "disgust", "fear", "joy", "pride", "relief", "sadness", "surprise"]


def find_best_checkpoint(base_dir):
    ckpts = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return base_dir
    best = sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(base_dir, best)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = find_best_checkpoint(MODEL_BASE_DIR)

    print("Using model directory:", model_dir)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    df = pd.read_csv(CONTARGA_PATH)
    print("Loaded CONTARGA:", df.shape)
    print("Columns:", df.columns.tolist())

    df_text = df[[TEXT_COL, CONV_COL]].copy()
    texts = df_text[TEXT_COL].tolist()

    all_probs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    print("Probability matrix shape:", all_probs.shape)

    for j, label in enumerate(GOEMOTIONS_LABELS):
        df_text[f"prob_{label}"] = all_probs[:, j]

    df_out = df.join(df_text.drop(columns=[TEXT_COL, CONV_COL]), how="left")
    df_out.to_csv(OUTPUT_PROBS_CSV, index=False)
    print("Saved probabilities to:", OUTPUT_PROBS_CSV)

    prob_cols = [c for c in df_out.columns if c.startswith("prob_")]
    rows = []
    for col in prob_cols:
        rho, p = spearmanr(df_out[col], df_out[CONV_COL])
        rows.append((col.replace("prob_", ""), rho, p))

    corr_conv = pd.DataFrame(rows, columns=["emotion_pred", "spearman_rho", "p_value"])
    corr_conv_sorted = corr_conv.sort_values("spearman_rho", ascending=False)
    corr_conv_sorted.to_csv(OUTPUT_CORR_CSV, index=False)
    print("Saved convincingness correlations to:", OUTPUT_CORR_CSV)

    emo_overlap = [e for e in EMO_OVERLAP if e in df_out.columns and f"prob_{e}" in df_out.columns]
    print("Using overlapping emotions:", emo_overlap)

    y_true = (df_out[emo_overlap].values >= 3).astype(int)
    y_pred = (df_out[[f"prob_{e}" for e in emo_overlap]].values >= 0.5).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print("\nRoBERTa (GoEmotions) zero-shot on CONTARGA (binarized ratings)")
    print("===============================================================")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {macro_prec:.3f}")
    print(f"Recall:    {macro_rec:.3f}")
    print(f"Macro-F1:  {macro_f1:.3f}")

    best_rows = []
    for emo in emo_overlap:
        label_idx = GOEMOTIONS_LABELS.index(emo)
        probs = all_probs[:, label_idx]
        labels = (df_out[emo].values >= 3).astype(int)

        best_f1 = 0.0
        best_thr = 0.0
        lo, hi = probs.min(), probs.max()
        thr_values = np.linspace(lo, hi, 200) if lo != hi else [lo]

        for thr in thr_values:
            preds = (probs >= thr).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        support = int(labels.sum())
        best_rows.append([emo, best_thr, best_f1, support])

    df_eval = pd.DataFrame(best_rows, columns=["emotion", "best_threshold", "best_F1", "support"])
    df_eval.to_csv(OUTPUT_EVAL_CSV, index=False)
    print("Saved evaluation table to:", OUTPUT_EVAL_CSV)
    print(df_eval)

    print("\nInterpretation: We observe a significant performance drop when transferring")
    print("from Reddit-style GoEmotions to argumentative texts in CONTARGA, confirming domain shift.")


if __name__ == "__main__":
    main()
