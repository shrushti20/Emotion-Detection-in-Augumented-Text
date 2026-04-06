"""
Script: contarga_llm_mistral_multilabel.py

Purpose:
Run multi-label emotion inference on a subset of the CONTARGA dataset
using Mistral-7B-Instruct.

Unlike the single-label prompting script, this version asks the model
to return a Python list of all emotions expressed in the text.

Input:
- CONTARGA CSV file with a 'text' column

Output:
- CSV with one binary column per emotion label

Notes:
- This script evaluates only the first N_SAMPLES instances.
- It is intended as an exploratory multi-label prompting setup.
"""

import os
import ast
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ----------------
BASE_DIR = "/home/hpc/v121ca/v121ca21"
DATA_PATH = f"{BASE_DIR}/thesis_data/contarga_llm/contarga_emotion_subset.csv"
RESULTS_DIR = f"{BASE_DIR}/thesis_results/llm"
OUT_PATH = f"{RESULTS_DIR}/mistral_contarga_multilabel_eval.csv"

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
EMOTIONS = ["anger", "disgust", "fear", "joy", "pride", "relief", "sadness", "surprise"]

# Use a smaller fixed subset for practical inference
N_SAMPLES = 300
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64
MAX_INPUT_LEN = 512
# ----------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)

# Restrict evaluation to a fixed subset
df = df.iloc[:N_SAMPLES].copy().reset_index(drop=True)
print("Subset shape:", df.shape)

print("Loading model:", MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Set pad token if the tokenizer does not define one explicitly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Ensure model config also knows the pad token
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Model on:", device)


def build_prompt(text: str) -> str:
    """
    Build a multi-label prompt asking the LLM to return
    all detected emotions as a Python list.
    """
    return f"""
You are an expert emotion classifier.
Identify ALL emotions expressed in this argumentative text.

TEXT: "{text}"

Possible emotions: {EMOTIONS}

Return ONLY a Python list of emotions, for example:
["anger", "fear"]
"""


def parse_emotions(output_text: str):
    """
    Extract the Python list from the model output.
    If parsing fails, return an empty list.
    """
    try:
        start = output_text.rfind("[")
        end = output_text.rfind("]") + 1
        return ast.literal_eval(output_text[start:end])
    except Exception:
        return []


# Prepare binary output columns for each emotion
for emo in EMOTIONS:
    df[f"llm_{emo}"] = 0

all_preds = []

print("Running LLM in batches...")
for start in tqdm(range(0, len(df), BATCH_SIZE)):
    end = min(start + BATCH_SIZE, len(df))
    batch_texts = df.loc[start:end - 1, "text"].tolist()

    # Build prompts for the current batch
    prompts = [build_prompt(t) for t in batch_texts]

    # Tokenize input prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LEN,
    ).to(device)

    # Generate model outputs without sampling
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Parse each generated response into a list of emotions
    batch_lists = [parse_emotions(txt) for txt in decoded]
    all_preds.extend(batch_lists)

# Write binary predictions into the dataframe
for idx, emo_list in enumerate(all_preds):
    for emo in emo_list:
        if emo in EMOTIONS:
            df.loc[idx, f"llm_{emo}"] = 1

print("Saving results to:", OUT_PATH)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print("Done. Saved", len(df), "rows.")