#!/usr/bin/env python3
"""
Script: contarga_llm_mistral_modes.py

Purpose:
Run prompt-based emotion classification on the CONTARGA dataset using
instruction-tuned large language models (LLMs).

This script supports multiple prompting strategies:
    - zero-shot
    - few-shot (static examples)
    - chain-of-thought (cot)
    - tf-idf retrieval-based few-shot prompting

Model:
    Default: Mistral-7B-Instruct (can be changed via --model_id)

Inputs:
    - CONTARGA dataset (CSV with text column)
    - optional few-shot examples or training data (for tf-idf mode)

Outputs:
    - CSV file with predicted emotion labels
    - one-vs-rest binary columns for each emotion
"""

import os
import argparse
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# TF-IDF retrieval for dynamic few-shot prompting
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- CONFIG ----------------
BASE_DIR = "/home/hpc/v121ca/v121ca21"
DATA_PATH_DEFAULT = f"{BASE_DIR}/thesis_data/contarga_llm/contarga_emotion_subset.csv"

# Target label space (CONTARGA-compatible)
LABELS = ["anger", "disgust", "fear", "joy", "pride", "relief", "sadness", "surprise"]
# ----------------------------------------


def clean_text(s: str) -> str:
    """Normalize whitespace in input text."""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_fewshot_block(rows: pd.DataFrame, max_chars_per_example: int = 350) -> str:
    """
    Format few-shot examples into prompt-ready text.

    Each example is formatted as:
    Text: "..."
    FINAL: <label>
    """
    lines = []
    for _, r in rows.iterrows():
        t = clean_text(r["text"])
        lab = str(r["emotion"]).strip().lower()

        # Skip labels not in target set
        if lab not in LABELS:
            continue

        # Truncate long examples
        if len(t) > max_chars_per_example:
            t = t[:max_chars_per_example].rstrip() + "..."

        lines.append(f'Text: "{t}"\nFINAL: {lab}\n')

    return "\n".join(lines).strip()


def build_prompt(text: str, mode: str, fewshot_text: str | None):
    """
    Construct the full prompt depending on the selected mode.
    """
    instr = (
        "You are an emotion classifier for argumentative text.\n"
        "Choose EXACTLY ONE label from this list:\n"
        f"{', '.join(LABELS)}\n\n"
    )

    # Add few-shot examples if needed
    if mode in ("few", "tfidf") and fewshot_text:
        instr += "Examples:\n" + fewshot_text.strip() + "\n\n"

    # Add CoT instruction if enabled
    if mode == "cot":
        instr += (
            "First think step-by-step (privately), then output ONLY the final label.\n"
            "Output format must be exactly:\n"
            "FINAL: <label>\n\n"
        )
    else:
        instr += (
            "Output format must be exactly:\n"
            "FINAL: <label>\n\n"
        )

    # Add input text
    instr += f'Text: "{clean_text(text)}"\nFINAL:'
    return instr


def parse_final_label(generated: str):
    """
    Extract the predicted label from model output.
    Handles both strict and fallback cases.
    """
    for line in generated.splitlines():
        line = line.strip()
        if line.lower().startswith("final:"):
            lab = line.split(":", 1)[1].strip().lower()
            lab = lab.split()[0].strip(",.;")
            if lab in LABELS:
                return lab

    # Fallback: find any label mention
    low = generated.lower()
    for lab in LABELS:
        if lab in low:
            return lab

    return "unknown"


def tfidf_retriever_setup(train_df: pd.DataFrame, text_col: str, label_col: str):
    """
    Prepare TF-IDF vectorizer for retrieval-based few-shot selection.
    """
    df = train_df.copy()
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).map(clean_text)
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()

    # Keep only valid labels
    df = df[df[label_col].isin(LABELS)].reset_index(drop=True)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    X = vectorizer.fit_transform(df[text_col].tolist())

    return df.rename(columns={text_col: "text", label_col: "emotion"}), vectorizer, X


def tfidf_select_examples(query_text, train_df_std, vectorizer, X_train, k, balance):
    """
    Select k most similar examples using cosine similarity.
    Optionally enforce class balance.
    """
    q = clean_text(query_text)
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X_train).flatten()
    order = np.argsort(-sims)

    if not balance:
        return train_df_std.iloc[order[:k]][["text", "emotion"]]

    # Balanced selection across labels
    per_class = int(np.ceil(k / len(LABELS)))
    picked, counts = [], {lab: 0 for lab in LABELS}

    for i in order:
        lab = train_df_std.iloc[i]["emotion"]
        if counts[lab] < per_class:
            picked.append(i)
            counts[lab] += 1
        if len(picked) >= k:
            break

    # Fallback if not enough samples
    if len(picked) < k:
        for i in order:
            if i not in picked:
                picked.append(i)
            if len(picked) >= k:
                break

    return train_df_std.iloc[picked][["text", "emotion"]]


def main():
    """
    Main pipeline:
    - load data
    - build prompts
    - run LLM inference
    - parse predictions
    - save results
    """
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=["zero", "few", "cot", "tfidf"],
        default="zero",
        help="Prompting mode: zero-shot, few-shot, chain-of-thought, or TF-IDF retrieval",
    )
    ap.add_argument(
        "--fewshot",
        default=None,
        help="Path to static few-shot examples file",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Path to output CSV file",
    )
    ap.add_argument(
        "--data",
        default=DATA_PATH_DEFAULT,
        help="Path to input CONTARGA CSV file",
    )
    ap.add_argument(
        "--model_id",
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Hugging Face model ID",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum number of new tokens to generate",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference",
    )

    # TF-IDF options
    ap.add_argument(
        "--train_data",
        default=None,
        help="CSV with labeled examples for TF-IDF retrieval",
    )
    ap.add_argument(
        "--text_col",
        default="text",
        help="Name of text column in train_data",
    )
    ap.add_argument(
        "--label_col",
        default="emotion",
        help="Name of label column in train_data",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of retrieved examples",
    )
    ap.add_argument(
        "--balance",
        action="store_true",
        help="Use class-balanced retrieval",
    )

    args = ap.parse_args()

    # Validate mode-specific inputs
    if args.mode == "few" and not args.fewshot:
        raise ValueError("--fewshot must be provided when --mode few is used")

    if args.mode == "tfidf" and not args.train_data:
        raise ValueError("--train_data must be provided when --mode tfidf is used")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Mode:", args.mode)
    print("Model:", args.model_id)
    print("Input data:", args.data)
    print("Output file:", args.out)

    # Load evaluation dataset
    df = pd.read_csv(args.data)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column")

    print("Loaded rows:", len(df))

    # Setup few-shot sources
    static_fewshot_text = None
    retr_train_df = retr_vec = retr_X = None

    if args.mode == "few":
        with open(args.fewshot, "r", encoding="utf-8") as f:
            static_fewshot_text = f.read()

    if args.mode == "tfidf":
        train_df = pd.read_csv(args.train_data)
        retr_train_df, retr_vec, retr_X = tfidf_retriever_setup(
            train_df, args.text_col, args.label_col
        )
        print("TF-IDF train rows:", len(retr_train_df))
        print("Retrieved examples (k):", args.k)
        print("Balanced retrieval:", args.balance)

    # Load tokenizer and model
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    preds = []

    # Run batched inference
    for start in tqdm(range(0, len(df), args.batch_size)):
        batch = df.iloc[start:start + args.batch_size]

        prompts = []
        for t in batch["text"].astype(str).tolist():
            if args.mode == "tfidf":
                ex_rows = tfidf_select_examples(
                    t, retr_train_df, retr_vec, retr_X,
                    k=args.k, balance=args.balance
                )
                fewshot_text = build_fewshot_block(ex_rows)

            elif args.mode == "few":
                fewshot_text = static_fewshot_text
            else:
                fewshot_text = None

            prompts.append(build_prompt(t, args.mode, fewshot_text))

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

        decoded = tok.batch_decode(out, skip_special_tokens=True)

        for full in decoded:
            preds.append(parse_final_label(full))

    # Save results
    out_df = df.copy()
    out_df["llm_pred"] = preds

    # One-hot encoding of predictions
    for lab in LABELS:
        out_df[f"llm_{lab}"] = [1 if p == lab else 0 for p in preds]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print("Saved:", args.out)
    print("Prediction counts:")
    print(out_df["llm_pred"].value_counts(dropna=False))


if __name__ == "__main__":
    main()