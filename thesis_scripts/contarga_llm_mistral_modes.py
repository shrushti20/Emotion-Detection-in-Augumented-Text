#!/usr/bin/env python3
import os
import argparse
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# TF-IDF retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = "/home/hpc/v121ca/v121ca21"
DATA_PATH_DEFAULT = f"{BASE_DIR}/thesis_data/contarga_llm/contarga_emotion_subset.csv"

LABELS = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]

def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_fewshot_block(rows: pd.DataFrame, max_chars_per_example: int = 350) -> str:
    """
    Format examples as:
    Text: "..."
    FINAL: <label>
    """
    lines = []
    for _, r in rows.iterrows():
        t = clean_text(r["text"])
        lab = str(r["emotion"]).strip().lower()
        if lab not in LABELS:
            continue
        if len(t) > max_chars_per_example:
            t = t[:max_chars_per_example].rstrip() + "..."
        lines.append(f'Text: "{t}"\nFINAL: {lab}\n')
    return "\n".join(lines).strip()

def build_prompt(text: str, mode: str, fewshot_text: str | None):
    instr = (
        "You are an emotion classifier for argumentative text.\n"
        "Choose EXACTLY ONE label from this list:\n"
        f"{', '.join(LABELS)}\n\n"
    )

    if mode in ("few", "tfidf") and fewshot_text:
        instr += "Examples:\n" + fewshot_text.strip() + "\n\n"

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

    instr += f'Text: "{clean_text(text)}"\nFINAL:'
    return instr

def parse_final_label(generated: str):
    for line in generated.splitlines():
        line = line.strip()
        if line.lower().startswith("final:"):
            lab = line.split(":", 1)[1].strip().lower()
            lab = lab.split()[0].strip(",.;")
            if lab in LABELS:
                return lab
    low = generated.lower()
    for lab in LABELS:
        if lab in low:
            return lab
    return "unknown"

def tfidf_retriever_setup(train_df: pd.DataFrame, text_col: str, label_col: str):
    df = train_df.copy()
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).map(clean_text)
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df = df[df[label_col].isin(LABELS)].reset_index(drop=True)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    X = vectorizer.fit_transform(df[text_col].tolist())
    return df.rename(columns={text_col: "text", label_col: "emotion"}), vectorizer, X

def tfidf_select_examples(
    query_text: str,
    train_df_std: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    X_train,
    k: int,
    balance: bool
) -> pd.DataFrame:
    q = clean_text(query_text)
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X_train).flatten()
    order = np.argsort(-sims)

    if not balance:
        idx = order[:k]
        return train_df_std.iloc[idx][["text","emotion"]]

    # Balanced: try to pick ~k/|labels| per class, in similarity order
    per_class = int(np.ceil(k / len(LABELS)))
    picked = []
    counts = {lab: 0 for lab in LABELS}

    for i in order:
        lab = train_df_std.iloc[i]["emotion"]
        if counts[lab] < per_class:
            picked.append(i)
            counts[lab] += 1
        if len(picked) >= k:
            break

    # fallback (if not enough due to label sparsity)
    if len(picked) < k:
        for i in order:
            if i not in picked:
                picked.append(i)
            if len(picked) >= k:
                break

    return train_df_std.iloc[picked][["text","emotion"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero","few","cot","tfidf"], default="zero")
    ap.add_argument("--fewshot", default=None, help="Text file containing static few-shot examples (used in mode=few)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DATA_PATH_DEFAULT)
    ap.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.1")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4)

    # TF-IDF retrieval options
    ap.add_argument("--train_data", default=None, help="CSV with in-domain labeled examples (for mode=tfidf)")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="emotion")
    ap.add_argument("--k", type=int, default=8, help="number of retrieved examples")
    ap.add_argument("--balance", action="store_true", help="class-balanced retrieval")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Mode:", args.mode)
    print("Model:", args.model_id)
    print("Data:", args.data)
    print("Out:", args.out)

    # Load eval data
    df = pd.read_csv(args.data)
    if "text" not in df.columns:
        raise ValueError(f"Expected 'text' column in --data CSV, got columns: {list(df.columns)}")
    print("Loaded rows:", len(df))

    # Prepare few-shot provider
    static_fewshot_text = None
    retr_train_df = None
    retr_vec = None
    retr_X = None

    if args.mode == "few":
        if not args.fewshot:
            raise ValueError("--fewshot is required when --mode few")
        with open(args.fewshot, "r", encoding="utf-8") as f:
            static_fewshot_text = f.read()

    if args.mode == "tfidf":
        if not args.train_data:
            raise ValueError("--train_data is required when --mode tfidf")
        train_df = pd.read_csv(args.train_data)
        retr_train_df, retr_vec, retr_X = tfidf_retriever_setup(train_df, args.text_col, args.label_col)
        print("TF-IDF train rows (filtered to 8 labels):", len(retr_train_df))
        print("TF-IDF k:", args.k, "balance:", args.balance)

    # Load tokenizer + model
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
    for start in tqdm(range(0, len(df), args.batch_size)):
        batch = df.iloc[start:start+args.batch_size]

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

        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
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
            pred = parse_final_label(full)
            preds.append(pred)

    out_df = df.copy()
    out_df["llm_pred"] = preds
    for lab in LABELS:
        out_df[f"llm_{lab}"] = [1 if p == lab else 0 for p in preds]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print("Saved:", args.out)
    print("Pred label counts:")
    print(out_df["llm_pred"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
