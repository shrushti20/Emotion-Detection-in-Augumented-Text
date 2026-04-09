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

# Default template (used if --prompt_file is not provided)
DEFAULT_PROMPT_TEMPLATE = """You are an emotion classification system.
Choose exactly one label from the following list:
{labels}

{examples}
Text:
{text}

Answer:
"""


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def make_example_block(text: str, label: str) -> str:
    text = (text or "").strip()
    label = (label or "").strip().lower()
    return f"Text: {text}\nLabel: {label}\n"


def load_static_fewshot_blocks(csv_path: str) -> list[str]:
    """
    Load fixed few-shot examples from a CSV.
    Required columns: text, label
    Optional: row_id (used for stable sorting)
    Returns: list[str] blocks in the same format as make_example_block().
    """
    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"--static_fewshot_file must have columns 'text' and 'label'. Found: {list(df.columns)}"
        )

    sort_cols = [c for c in ["label", "row_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    blocks: list[str] = []
    for _, r in df.iterrows():
        blocks.append(make_example_block(str(r["text"]), str(r["label"])))
    return blocks


def load_prompt_template(prompt_file: str | None) -> str:
    if not prompt_file:
        return DEFAULT_PROMPT_TEMPLATE
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt_from_template(
    template: str,
    query_text: str,
    fewshot_blocks: list[str],
    labels: list[str],
) -> str:
    examples = "\n".join(fewshot_blocks).strip()
    examples_text = ""
    if examples:
        examples_text = "Examples:\n" + examples + "\n\n"

    return template.format(
        labels=", ".join(labels),
        examples=examples_text,
        text=query_text.strip(),
    )


def parse_final_label(decoded_text: str, cot: bool = False) -> str:
    """
    Extract the predicted label from model output.

    If cot=True, we first try to parse a strict line:
      Final label: <label>
    This avoids accidentally matching labels inside the reasoning text.
    """
    txt = (decoded_text or "").strip()

    if cot:
        m = re.search(r"(?im)^\s*final\s*label\s*:\s*([a-z]+)\s*$", txt)
        if m:
            lab = m.group(1).strip().lower()
            return lab if lab in LABELS else "unknown"

        # Fallback: choose the LAST label occurrence in the output
        low = txt.lower()
        hits: list[tuple[int, str]] = []
        for lab in LABELS:
            for mm in re.finditer(rf"\b{lab}\b", low):
                hits.append((mm.start(), lab))
        if hits:
            hits.sort()
            return hits[-1][1]
        return "unknown"

    low = txt.lower()

    for lab in LABELS:
        if re.search(rf"\b{lab}\b", low):
            return lab

    first = re.split(r"\s+", low)[0] if low else ""
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

    ap.add_argument(
        "--prompt_file",
        default=None,
        help="Path to a prompt template file with placeholders: {labels}, {examples}, {text}.",
    )
    ap.add_argument(
        "--print_prompts",
        action="store_true",
        help="Print the constructed prompts (first batch) and exit.",
    )
    ap.add_argument(
        "--cot",
        action="store_true",
        help="Enable CoT parsing. Expects model output to contain a 'Final label: <label>' line.",
    )

    # NEW: fixed/traceable few-shot examples from CSV
    ap.add_argument(
        "--static_fewshot_file",
        type=str,
        default=None,
        help="Optional CSV with fixed few-shot examples (columns: text,label; optional row_id). If set, overrides TF-IDF retrieval.",
    )

    args = ap.parse_args()

    # Load static few-shot blocks once (if provided)
    static_fewshot_blocks: list[str] | None = None
    if args.static_fewshot_file:
        static_fewshot_blocks = load_static_fewshot_blocks(args.static_fewshot_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_template = load_prompt_template(args.prompt_file)

    print("Device:", device, flush=True)
    print("Mode:", args.mode, flush=True)
    print("Model:", args.model_id, flush=True)
    print("Cache dir:", args.cache_dir, flush=True)
    print("Local files only:", args.local_files_only, flush=True)
    print("Data:", args.data, flush=True)
    print("Train:", args.train_data, flush=True)
    print("Out:", args.out, flush=True)
    print("Prompt file:", args.prompt_file or "<DEFAULT_PROMPT_TEMPLATE>", flush=True)
    print("CoT:", args.cot, flush=True)
    print("Static few-shot:", args.static_fewshot_file or "<none>", flush=True)

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
        torch_dtype=dtype,
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
            # Choose examples source:
            # 1) static fixed few-shot if provided
            # 2) else TF-IDF retrieval if k > 0
            # 3) else zero-shot
            if static_fewshot_blocks is not None:
                fewshot_blocks = static_fewshot_blocks
            else:
                if args.k and args.k > 0:
                    idxs = tfidf_retrieve(
                        train_texts, train_labels, t, k=args.k, balance=args.balance
                    )
                    fewshot_blocks = [
                        make_example_block(train_texts[i], train_labels[i]) for i in idxs
                    ]
                else:
                    fewshot_blocks = []

            prompt = build_prompt_from_template(
                template=prompt_template,
                query_text=t,
                fewshot_blocks=fewshot_blocks,
                labels=LABELS,
            )
            prompts.append(prompt)

        # Print prompts once (first batch) and exit
        if args.print_prompts:
            for i, p in enumerate(prompts):
                print("\n" + "=" * 40)
                print(f"PROMPT #{i}")
                print("=" * 40)
                print(p)
            raise SystemExit(0)

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
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

        # Decode ONLY newly generated tokens (not the prompt)
        gen_only = out[:, enc["input_ids"].shape[1] :]
        decoded = tok.batch_decode(gen_only, skip_special_tokens=True)

        for full in decoded:
            pred = parse_final_label(full, cot=args.cot)
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
