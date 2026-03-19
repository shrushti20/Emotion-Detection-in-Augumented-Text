#!/bin/bash
#SBATCH --job-name=zephyr_cot_ct
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_results/logs/zephyr_cot_%j.out
#SBATCH --error=/home/hpc/v121ca/v121ca21/thesis_results/logs/zephyr_cot_%j.err

set -euo pipefail

PY=/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python

$PY /home/hpc/v121ca/v121ca21/thesis_scripts/contarga_llm_mistral_modes.py \
  --mode cot \
  --model_id HuggingFaceH4/zephyr-7b-beta \
  --data /home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/contarga_emotion_subset.csv \
  --out /home/hpc/v121ca/v121ca21/thesis_results/llm/contarga/zephyr_cot.csv \
  --batch_size 4 \
  --max_new_tokens 64
