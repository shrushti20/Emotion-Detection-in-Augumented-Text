from datasets import load_dataset
import pandas as pd
import os

OUT_DIR = "/home/hpc/v121ca/v121ca21/thesis_data/tweeteval_emotion"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading TweetEval Emotion dataset...")
ds = load_dataset("tweet_eval", "emotion")

# Combine train + validation + test (optional)
df_train = pd.DataFrame(ds["train"])
df_valid = pd.DataFrame(ds["validation"])
df_test  = pd.DataFrame(ds["test"])

# Label mapping
label_names = ["anger", "fear", "joy", "love", "sadness", "surprise"]

def map_label(id):
    return label_names[id]

for df in [df_train, df_valid, df_test]:
    df["emotion"] = df["label"].apply(map_label)
    df.drop(columns=["label"], inplace=True)

# Save
df_train.to_csv(f"{OUT_DIR}/tweets_train.csv", index=False)
df_valid.to_csv(f"{OUT_DIR}/tweets_validation.csv", index=False)
df_test.to_csv(f"{OUT_DIR}/tweets_test.csv", index=False)

print("Saved TweetEval CSV files to:", OUT_DIR)
