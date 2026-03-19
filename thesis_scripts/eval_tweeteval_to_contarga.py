#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TWEETEVAL_LABELS = ["anger", "fear", "joy", "love"]
CONTARGA_EVAL_LABELS = ["anger", "fear", "joy", "pride"]
contarga_label2id = {l: i for i, l in enumerate(CONTARGA_EVAL_LABELS)}

def soft_map_pred(pred_te: str, gold_ct: str) -> str:
    """
    Your mapping rule (Option 2):
    - anger -> anger
    - fear  -> fear
    - joy   -> joy
    - love  -> counts as correct if gold is joy or pride

    For metrics, we must output a single CONTARGA label.
    We implement the rule by mapping:
      - if pred is love and gold is joy/pride => mapped_pred = gold (counts as correct)
      - if pred is love and gold is anger/fear => mapped_pred = joy (arbitrary but will be counted wrong)
    """
    pred_te = str(pred_te).strip().lower()
    gold_ct = str(gold_ct).strip().lower()

    if pred_te in ("anger", "fear", "joy"):
        return pred_te
    if pred_te == "love":
        if gold_ct in ("joy", "pride"):
            return gold_ct
        return "joy"  # will be wrong for gold anger/fear; keeps label in-space
    return "joy"  # fallback (shouldn't happen)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to TweetEval-trained model directory")
    ap.add_argument("--contarga_csv", required=True, help="Filtered CONTARGA CSV (anger/fear/joy/pride)")
    ap.add_argument("--out_csv", required=True, help="Where to save predictions CSV")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    df = pd.read_csv(args.contarga_csv)
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df = df[df["emotion"].isin(CONTARGA_EVAL_LABELS)].copy()
    df = df.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    texts = df["text"].astype(str).tolist()

    all_pred_ids = []
    all_probs = []

    for i in range(0, len(texts), args.batch):
        batch_texts = texts[i:i+args.batch]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=args.max_len,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=-1)

        all_pred_ids.extend(pred_ids.tolist())
        all_probs.extend(probs.tolist())

    pred_te = [TWEETEVAL_LABELS[i] for i in all_pred_ids]
    gold_ct = df["emotion"].tolist()

    mapped_pred = [soft_map_pred(p, g) for p, g in zip(pred_te, gold_ct)]

    y_true = [contarga_label2id[g] for g in gold_ct]
    y_pred = [contarga_label2id[m] for m in mapped_pred]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print({"accuracy": float(acc), "macro_f1": float(macro_f1), "n": int(len(df))}, flush=True)

    out_df = df.copy()
    out_df["pred_tweeteval"] = pred_te
    out_df["pred_mapped_contarga"] = mapped_pred
    out_df["correct_soft"] = (out_df["pred_mapped_contarga"] == out_df["emotion"]).astype(int)

    # store probs too (4 columns)
    probs_arr = np.array(all_probs)
    for j, lab in enumerate(TWEETEVAL_LABELS):
        out_df[f"prob_{lab}"] = probs_arr[:, j]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("Saved predictions to:", args.out_csv, flush=True)


if __name__ == "__main__":
    main()
