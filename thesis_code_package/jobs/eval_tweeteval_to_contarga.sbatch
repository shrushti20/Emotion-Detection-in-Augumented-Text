#!/bin/bash
#SBATCH --job-name=eval_te2ct
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_results/logs/eval_te2ct_%j.out
#SBATCH --error=/home/hpc/v121ca/v121ca21/thesis_results/logs/eval_te2ct_%j.err

set -euo pipefail
mkdir -p /home/hpc/v121ca/v121ca21/thesis_results/logs

PY=/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python
EVAL_CSV=/home/hpc/v121ca/v121ca21/thesis_results/tweeteval_to_contarga/contarga_eval_anger_fear_joy_pride.csv

$PY /home/hpc/v121ca/v121ca21/thesis_scripts/eval_tweeteval_to_contarga.py \
  --model_dir /home/hpc/v121ca/v121ca21/thesis_models/roberta_tweeteval \
  --contarga_csv "$EVAL_CSV" \
  --out_csv /home/hpc/v121ca/v121ca21/thesis_results/tweeteval_to_contarga/roberta_tweeteval_on_contarga.csv \
  --batch 64 --max_len 256

$PY /home/hpc/v121ca/v121ca21/thesis_scripts/eval_tweeteval_to_contarga.py \
  --model_dir /home/hpc/v121ca/v121ca21/thesis_models/deberta_tweeteval \
  --contarga_csv "$EVAL_CSV" \
  --out_csv /home/hpc/v121ca/v121ca21/thesis_results/tweeteval_to_contarga/deberta_tweeteval_on_contarga.csv \
  --batch 32 --max_len 256
