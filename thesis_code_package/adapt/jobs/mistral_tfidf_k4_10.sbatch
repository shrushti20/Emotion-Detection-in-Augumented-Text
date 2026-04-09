#!/bin/bash
#SBATCH --job-name=mistral_tfidf_k2_10
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_adapt/logs/mistral_tfidf_k4_10_%j.out
#SBATCH --error=/home/hpc/v121ca/v121ca21/thesis_adapt/logs/mistral_tfidf_k4_10_%j.err

set -euo pipefail

mkdir -p /home/hpc/v121ca/v121ca21/thesis_adapt/logs
mkdir -p /home/hpc/v121ca/v121ca21/thesis_results/llm

echo "=== START ==="
date
hostname
pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# --- Conda env (no 'module' dependency) ---
# If conda isn't found in batch shells, source your shell init.
# Use one of these that exists on your system:
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
[ -f "$HOME/.bash_profile" ] && source "$HOME/.bash_profile"

conda activate thesis

# --- Use thesis env python directly (no conda needed) ---
PYTHON=/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python
echo "PYTHON=$PYTHON"
"$PYTHON" -V

# sanity check torch/cuda
"$PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())
PY

# --- HF cache in HOME (compute nodes can see this) ---
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Offline mode on compute nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_DIR="$HF_HUB_CACHE/models--mistralai--Mistral-7B-Instruct-v0.1"
echo "MODEL_DIR=$MODEL_DIR"
SNAP=$(ls -dt "$MODEL_DIR"/snapshots/* 2>/dev/null | head -n 1 || true)
echo "SNAP=$SNAP"
test -d "$SNAP" || { echo "ERROR: Local snapshot not found under $MODEL_DIR/snapshots/*"; exit 1; }

echo "=== RUN SCRIPT ==="
"$PYTHON" -u /home/hpc/v121ca/v121ca21/thesis_adapt/scripts/contarga_llm_mistral_tfidf.py \
  --model_id "$SNAP" \
  --local_files_only \
  --mode tfidf \
  --data /home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/contarga_emotion_subset_10.csv \
  --train_data /home/hpc/v121ca/v121ca21/thesis_adapt/data/contarga/train.csv \
  --text_col text \
  --label_col emotion \
  --k 4 \
  --balance \
  --batch_size 1 \
  --max_new_tokens 16 \
  --out /home/hpc/v121ca/v121ca21/thesis_results/llm/mistral_tfidf_balanced_k4_10.csv

echo "=== DONE ==="
date
ls -lh /home/hpc/v121ca/v121ca21/thesis_results/llm/mistral_tfidf_balanced_k4_10.csv || true
