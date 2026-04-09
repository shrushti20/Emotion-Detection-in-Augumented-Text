import os
import re
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM


LABELS = ["anger", "disgust", "fear", "joy", "pride", "relief", "sadness", "surprise"]


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def make_example_block(text: str, label: str) -> str:
    text = (text or "").strip()
    label = (label or "").strip().lower()
    return f"Text: {text}\nLabel: {label}\n"


def build_prompt(query_text: str, fewshot_blocks: list[str]) -> str:
    examples = "\n".join(fewshot_blocks).strip()

    prompt = (
        "You are a classifier. Output ONLY one token from this list:\n"
        "anger, disgust, fear, joy, pride, relief, sadness, surprise\n\n"
    )

    if examples:
        prompt += "Examples:\n" + examples + "\n\n"

    prompt += (
        "Text: " + query_text.strip() + "\n"
        "Label: "
    )
    return prompt


def parse_final_label(decoded_text: str) -> str:
    txt = (decoded_text or "").strip().lower()

    # If model output is long, just search for the first valid label anywhere
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", txt):
            return lab

    # If it output just one token with punctuation like "joy." or "joy,"
    first = re.split(r"\s+", txt)[0] if txt else ""
    first = re.sub(r"[^\w]", "", first)
    return first if first in LABELS else "unknown"



def tfidf_retrieve(
    train_texts: list[str],
    train_labels: list[str],
    query: str,
    k: int,
    balance: bool,
) -> list[int]:
    query = normalize_text(query)

    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform([normalize_text(t) for t in train_texts])
    q = vec.transform([query])

    sims = (X @ q.T).toarray().reshape(-1)
    order = np.argsort(-sims)  # descending similarity

    if not balance:
        return order[:k].tolist()

    picked: list[int] = []
    seen_labels: set[str] = set()

    # First pass: distinct labels
    for idx in order:
        lab = (train_labels[idx] or "").strip().lower()
        if lab in LABELS and lab not in seen_labels:
            picked.append(int(idx))
            seen_labels.add(lab)
            if len(picked) >= k:
                return picked

    # Second pass: fill remainder
    for idx in order:
        if int(idx) not in picked:
            picked.append(int(idx))
            if len(picked) >= k:
                break

    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tfidf"], default="tfidf")

    ap.add_argument("--data", required=True)
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="emotion")

    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--balance", action="store_true")

    ap.add_argument("--out", required=True)

    ap.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.1")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=1)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device, flush=True)
    print("Mode:", args.mode, flush=True)
    print("Model:", args.model_id, flush=True)
    print("Cache dir:", args.cache_dir, flush=True)
    print("Local files only:", args.local_files_only, flush=True)
    print("Data:", args.data, flush=True)
    print("Train:", args.train_data, flush=True)
    print("Out:", args.out, flush=True)

    df = pd.read_csv(args.data)
    train_df = pd.read_csv(args.train_data)

    # Filter TF-IDF pool to label space
    train_df = train_df[train_df[args.label_col].isin(LABELS)].reset_index(drop=True)

    print("Loaded rows:", len(df), flush=True)
    print("TF-IDF train rows (filtered to LABELS):", len(train_df), flush=True)
    print("TF-IDF k:", args.k, "balance:", args.balance, flush=True)

    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading model...", flush=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        device_map=None,
    )
    model.to(device)
    model.eval()

    print("MODEL DEVICE:", next(model.parameters()).device, flush=True)

    train_texts = train_df[args.text_col].astype(str).tolist()
    train_labels = train_df[args.label_col].astype(str).tolist()

    llm_pred: list[str] = []

    for start in tqdm(range(0, len(df), args.batch_size)):
        batch = df.iloc[start : start + args.batch_size]

        prompts: list[str] = []
        for t in batch[args.text_col].astype(str).tolist():
            idxs = tfidf_retrieve(train_texts, train_labels, t, k=args.k, balance=args.balance)
            fewshot_blocks = [make_example_block(train_texts[i], train_labels[i]) for i in idxs]
            prompts.append(build_prompt(t, fewshot_blocks))

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

        # Decode ONLY newly generated tokens (not the prompt)
        gen_only = out[:, enc["input_ids"].shape[1] :]
        decoded = tok.batch_decode(gen_only, skip_special_tokens=True)

        for full in decoded:
            pred = parse_final_label(full)
            llm_pred.append(pred)

    out_df = df.copy()
    out_df["llm_pred"] = llm_pred
    for lab in LABELS:
        out_df[f"llm_{lab}"] = [1 if p == lab else 0 for p in llm_pred]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print("Saved:", args.out, flush=True)
    print(out_df["llm_pred"].value_counts(dropna=False), flush=True)


if __name__ == "__main__":
    main()
