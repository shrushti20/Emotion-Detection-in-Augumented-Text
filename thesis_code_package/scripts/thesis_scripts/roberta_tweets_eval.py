import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

BASE = "/home/hpc/v121ca/v121ca21"
MODEL_PATH = f"{BASE}/thesis_models/goemo_roberta_base/checkpoint-1500"
DATA_PATH = f"{BASE}/thesis_data/tweeteval_emotion/tweets_test.csv"
OUT_PATH = f"{BASE}/thesis_results/tweets/roberta_tweets_eval.csv"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print("Loading model:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Device:", device)

df = pd.read_csv(DATA_PATH)
texts = df["text"].tolist()

all_probs = []

print("Running RoBERTa on TweetEval test set...")
for text in tqdm(texts):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = logits.softmax(dim=1).cpu().numpy()[0]
    all_probs.append(probs)

# Convert to DataFrame
prob_df = pd.DataFrame(all_probs, columns=[f"p_{i}" for i in range(model.config.num_labels)])
df_out = pd.concat([df, prob_df], axis=1)

df_out.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
