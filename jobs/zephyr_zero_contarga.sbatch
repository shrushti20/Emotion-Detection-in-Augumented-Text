#!/bin/bash
#SBATCH --job-name=zephyr_zero_ct
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_results/logs/zephyr_zero_%j.out
#SBATCH --error=/home/hpc/v121ca/v121ca21/thesis_results/logs/zephyr_zero_%j.err

set -euo pipefail

PY=/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python
DATA=/home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/contarga_emotion_subset.csv
OUT=/home/hpc/v121ca/v121ca21/thesis_results/llm/contarga/zephyr_zero.csv
MODEL=HuggingFaceH4/zephyr-7b-beta

mkdir -p /home/hpc/v121ca/v121ca21/thesis_results/logs
mkdir -p /home/hpc/v121ca/v121ca21/thesis_results/llm/contarga

$PY /home/hpc/v121ca/v121ca21/thesis_scripts/contarga_llm_mistral_modes.py \
  --model "$MODEL" \
  --data "$DATA" \
  --out  "$OUT"
