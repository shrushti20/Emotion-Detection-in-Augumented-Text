#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

LABELS = ["anger","disgust","fear","joy","pride","relief","sadness","surprise"]
SEED = 42

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_label", type=int, required=True)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_path)
    df = df[df["emotion"].isin(LABELS)].copy()
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    train = (
        df.groupby("emotion", group_keys=False)
          .apply(lambda x: x.sample(n=min(args.n_per_label, len(x)), random_state=SEED))
          .reset_index(drop=True)
    )

    rest = df.drop(train.index, errors="ignore").reset_index(drop=True)
    dev = rest.sample(frac=args.dev_frac, random_state=SEED)
    test = rest.drop(dev.index).reset_index(drop=True)

    train.to_csv(out_dir / "train.csv", index=False)
    dev.to_csv(out_dir / "dev.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print("Saved:", out_dir)
    print(" train:", train["emotion"].value_counts().to_dict(), "shape", train.shape)
    print(" dev  :", dev.shape)
    print(" test :", test.shape)

if __name__ == "__main__":
    main()

