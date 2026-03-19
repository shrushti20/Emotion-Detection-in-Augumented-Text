#!/bin/bash
#SBATCH --job-name=mistral_tfidf_k8_200
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=thesis_adapt/logs/mistral_tfidf_k8_200_%j.out
#SBATCH --error=thesis_adapt/logs/mistral_tfidf_k8_200_%j.err

set -euo pipefail
PYTHON="/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python"

export HF_HOME=/home/vault/v121ca/v121ca21/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME
export XDG_CACHE_HOME=$HF_HOME
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

echo "=== START ==="
date
hostname
$PYTHON -V
nvidia-smi

$PYTHON -u /home/hpc/v121ca/v121ca21/thesis_adapt/scripts/contarga_llm_mistral_tfidf.py \
  --mode tfidf \
  --model_id mistralai/Mistral-7B-Instruct-v0.1 \
  --data /home/hpc/v121ca/v121ca21/thesis_data/contarga_llm/contarga_emotion_subset_200.csv \
  --train_data /home/hpc/v121ca/v121ca21/thesis_adapt/data/contarga/train.csv \
  --text_col text \
  --label_col emotion \
  --k 8 \
  --balance \
  --batch_size 4 \
  --max_new_tokens 16 \
  --out /home/hpc/v121ca/v121ca21/thesis_results/llm/mistral_tfidf_balanced_k8_200.csv

echo "=== DONE ==="
date
ls -lh /home/hpc/v121ca/v121ca21/thesis_results/llm/mistral_tfidf_balanced_k8_200.csv || true
