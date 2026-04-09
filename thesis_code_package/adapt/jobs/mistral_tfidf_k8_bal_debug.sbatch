#!/bin/bash
#SBATCH --job-name=mistral_tfidf_k8_dbg
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=thesis_adapt/logs/mistral_tfidf_k8_dbg_%j.out
#SBATCH --error=thesis_adapt/logs/mistral_tfidf_k8_dbg_%j.err

set -euo pipefail

echo "=== START ==="
date
hostname
pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# Use the env python directly (no module dependency)
PYTHON="$HOME/.conda/envs/thesis/bin/python"
echo "PYTHON=$PYTHON"
"$PYTHON" -V

# Caches
export HF_HOME=/home/vault/v121ca/v121ca21/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=$HF_HOME
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

# Force unbuffered logs
export PYTHONUNBUFFERED=1

echo "=== GPU (before python) ==="
nvidia-smi || true

# Background GPU monitor (prints every 30s into .out)
( while true; do
    echo "=== GPU MONITOR $(date) ==="
    nvidia-smi || true
    sleep 30
  done ) &

echo "=== RUN SCRIPT ==="
"$PYTHON" -u /home/hpc/v121ca/v121ca21/thesis_scripts/contarga_llm_mistral_modes.py \
  --mode tfidf \
  --data /home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/contarga_emotion_subset_200.csv \
  --train_data /home/hpc/v121ca/v121ca21/thesis_adapt/data/contarga/train.csv \
  --text_col text \
  --label_col emotion \
  --k 8 \
  --balance \
  --max_new_tokens 8 \
  --batch_size 1 \
  --out /home/hpc/v121ca/v121ca21/thesis_results/llm/mistral_tfidf_balanced_k8_200.csv

echo "=== DONE ==="
date
