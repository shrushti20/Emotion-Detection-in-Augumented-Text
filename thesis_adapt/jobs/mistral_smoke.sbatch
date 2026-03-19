#!/bin/bash
#SBATCH --job-name=mistral_smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=thesis_adapt/logs/mistral_smoke_%j.out
#SBATCH --error=thesis_adapt/logs/mistral_smoke_%j.err

set -euo pipefail

PYTHON="/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python"

# cache to vault (IMPORTANT)
export HF_HOME=/home/vault/v121ca/v121ca21/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME
export XDG_CACHE_HOME=$HF_HOME
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

echo "=== START ==="
date
hostname
echo "PYTHON=$PYTHON"
$PYTHON -V

echo "=== NVIDIA-SMI BEFORE ==="
nvidia-smi

$PYTHON -u - <<'PY'
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"
print("HF_HOME:", os.environ.get("HF_HOME"))
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# FORCE GPU (no device_map guessing)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to("cuda")
model.eval()

print("model device:", next(model.parameters()).device)

prompt = "You are a classifier. Choose one label: anger, joy.\nText: I am so happy.\nFINAL:"
enc = tok([prompt], return_tensors="pt").to("cuda")
out = model.generate(**enc, max_new_tokens=8, do_sample=False)
print("decoded:", tok.decode(out[0], skip_special_tokens=True))

print("cuda mem allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
PY

echo "=== NVIDIA-SMI AFTER ==="
nvidia-smi
echo "=== DONE ==="
date
