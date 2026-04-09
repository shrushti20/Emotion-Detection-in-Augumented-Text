#!/bin/bash
#SBATCH --job-name=prefetch_mistral
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=thesis_adapt/logs/prefetch_mistral_scratch_%j.out
#SBATCH --error=thesis_adapt/logs/prefetch_mistral_scratch_%j.err

set -euo pipefail

echo "=== PREFETCH START ==="
date
hostname
pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# Use env python directly (no modules needed)
PYTHON="$HOME/.conda/envs/thesis/bin/python"
echo "PYTHON=$PYTHON"
"$PYTHON" -V

# HF caches on scratch
export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

"$PYTHON" - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
m="mistralai/Mistral-7B-Instruct-v0.1"
print("Loading tokenizer...")
AutoTokenizer.from_pretrained(m)
print("Loading model...")
AutoModelForCausalLM.from_pretrained(m)
print("Done prefetch.")
PY

echo "=== PREFETCH DONE ==="
date
