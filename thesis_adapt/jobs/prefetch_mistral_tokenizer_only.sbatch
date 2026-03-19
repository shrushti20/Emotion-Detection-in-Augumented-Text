#!/bin/bash
#SBATCH --job-name=prefetch_mistral_tok
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_adapt/logs/prefetch_mistral_tok_%j.out
#SBATCH --error=/home/hpc/v121ca/v121ca21/thesis_adapt/logs/prefetch_mistral_tok_%j.err

set -euo pipefail

mkdir -p /home/hpc/v121ca/v121ca21/thesis_adapt/logs
export PYTHONUNBUFFERED=1

echo "=== START ==="
date
hostname
pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

PYTHON="$HOME/.conda/envs/thesis/bin/python"
echo "PYTHON=$PYTHON"
"$PYTHON" -V

export HF_HOME=/scratch/$USER/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

# Fail fast if the node can't reach HF (10s connect timeout)
export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_HUB_ETAG_TIMEOUT=30

"$PYTHON" -u - <<'PY'
import os, time
from transformers import AutoTokenizer

m="mistralai/Mistral-7B-Instruct-v0.1"
print("Loading tokenizer:", m, flush=True)
t0=time.time()
tok = AutoTokenizer.from_pretrained(m)
print(f"Done. Tokenizer class: {tok.__class__.__name__}. Took {time.time()-t0:.1f}s", flush=True)
print("HF_HOME:", os.environ.get("HF_HOME"), flush=True)
PY

echo "Cache size now:"
du -sh "/scratch/$USER/hf" || true

echo "=== DONE ==="
date
