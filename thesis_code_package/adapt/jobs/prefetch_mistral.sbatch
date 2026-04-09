#!/bin/bash
#SBATCH --job-name=prefetch_mistral
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=thesis_adapt/logs/prefetch_mistral_%j.out
#SBATCH --error=thesis_adapt/logs/prefetch_mistral_%j.err

set -euo pipefail

PYTHON=/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python

export HF_HOME=/home/vault/v121ca/v121ca21/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=$HF_HOME

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

echo "=== PREFETCH START ==="
date
hostname
echo "PYTHON=$PYTHON"
$PYTHON -V
echo "HF_HOME=$HF_HOME"
nvidia-smi

$PYTHON -u - <<'PY'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

mid="mistralai/Mistral-7B-Instruct-v0.1"
print("HF_HOME:", os.environ.get("HF_HOME"), flush=True)

print("Downloading tokenizer...", flush=True)
_ = AutoTokenizer.from_pretrained(mid)
print("Tokenizer done", flush=True)

print("Downloading model...", flush=True)
_ = AutoModelForCausalLM.from_pretrained(mid)
print("Model done", flush=True)

print("PREFETCH DONE", flush=True)
PY

echo "=== PREFETCH END ==="
date
