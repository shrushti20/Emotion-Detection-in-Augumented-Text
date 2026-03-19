#!/bin/bash
#SBATCH --job-name=zephyr_few_ct
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_results/logs/zephyr_few_%j.out
#SBATCH --error=/home/hpc/v121ca/v121ca21/thesis_results/logs/zephyr_few_%j.err

set -euo pipefail

PY=/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python

$PY /home/hpc/v121ca/v121ca21/thesis_scripts/contarga_llm_mistral_modes.py \
  --mode few \
  --model_id HuggingFaceH4/zephyr-7b-beta \
  --data /home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/contarga_emotion_subset.csv \
  --fewshot /home/hpc/v121ca/v121ca21/thesis_results/llm/contarga/fewshot_unique_k8.txt \
  --out /home/hpc/v121ca/v121ca21/thesis_results/llm/contarga/zephyr_few.csv \
  --batch_size 4 \
  --max_new_tokens 32
