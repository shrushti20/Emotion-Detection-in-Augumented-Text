import os
import ast
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- CONFIG ----------
BASE_DIR  = "/home/hpc/v121ca/v121ca21"  # <-- replace with the pwd you saw
DATA_PATH = f"{BASE_DIR}/thesis_data/contarga_llm/contarga_emotion_subset.csv"
RESULTS_DIR = f"{BASE_DIR}/thesis_results/llm"
OUT_PATH = f"{RESULTS_DIR}/gemma_contarga_eval.csv"

MODEL_DIR = f"{BASE_DIR}/thesis_models/goemo_roberta_base"  # or your actual model folder

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
EMOTIONS  = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]
N_SAMPLES = 300                      # start small; you can increase to 500 later
BATCH_SIZE = 4                       # keep modest for GPU memory
MAX_NEW_TOKENS = 64
MAX_INPUT_LEN   = 512
# -----------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)
df = df.iloc[:N_SAMPLES].copy().reset_index(drop=True)
print("Subset shape:", df.shape)

print("Loading model:", MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# --- IMPORTANT: set a pad token for Mistral/Gemma ---
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# make sure model knows the pad token
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Model on:", device)


def build_prompt(text: str) -> str:
    return f"""
You are an expert emotion classifier.
Identify ALL emotions expressed in this argumentative text.

TEXT: "{text}"

Possible emotions: {EMOTIONS}

Return ONLY a Python list of emotions, for example:
["anger", "fear"]
"""

def parse_emotions(output_text: str):
    # try to find a Python list in the LLM output
    try:
        start = output_text.rfind("[")
        end   = output_text.rfind("]") + 1
        return ast.literal_eval(output_text[start:end])
    except Exception:
        return []

# prepare columns
for emo in EMOTIONS:
    df[f"llm_{emo}"] = 0

all_preds = []

print("Running LLM in batches...")
for start in tqdm(range(0, len(df), BATCH_SIZE)):
    end = min(start + BATCH_SIZE, len(df))
    batch_texts = df.loc[start:end-1, "text"].tolist()

    prompts = [build_prompt(t) for t in batch_texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LEN,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch_lists = [parse_emotions(txt) for txt in decoded]
    all_preds.extend(batch_lists)

# write predictions into df
for idx, emo_list in enumerate(all_preds):
    for emo in emo_list:
        if emo in EMOTIONS:
            df.loc[idx, f"llm_{emo}"] = 1

print("Saving results to:", OUT_PATH)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print("Done. Saved", len(df), "rows.")
