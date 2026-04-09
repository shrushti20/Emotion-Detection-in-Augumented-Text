#!/bin/bash
#SBATCH --job-name=deberta_tweeteval
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_logs/deberta_tweeteval_%j.out
#SBATCH --error=/home/hpc/v121ca/v121ca21/thesis_logs/deberta_tweeteval_%j.err

set -euo pipefail

echo "HOST=$(hostname)"
echo "PWD=$(pwd)"

PY=/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python
echo "PY=$PY"
ls -l "$PY" || true
"$PY" -V || true

echo "which python: $(which python || true)"
python -V || true

echo "=== starting training ==="
"$PY" /home/hpc/v121ca/v121ca21/thesis_scripts/train_deberta_tweets.py \
  --train /home/hpc/v121ca/v121ca21/thesis_data/tweeteval_emotion/tweets_train.csv \
  --valid /home/hpc/v121ca/v121ca21/thesis_data/tweeteval_emotion/tweets_validation.csv \
  --test  /home/hpc/v121ca/v121ca21/thesis_data/tweeteval_emotion/tweets_test.csv \
  --out_dir /home/hpc/v121ca/v121ca21/thesis_models/deberta_tweeteval \
  --epochs 3 \
  --batch 8 \
  --lr 2e-5 \
  --max_len 128
