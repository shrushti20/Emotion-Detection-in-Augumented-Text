#!/bin/bash
#SBATCH --job-name=db_fix_n50
#SBATCH --partition=a40
#SBATCH --account=v121ca
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=thesis_adapt/logs/deberta_fixed_n50_%j.out
#SBATCH --error=thesis_adapt/logs/deberta_fixed_n50_%j.err

source /etc/profile
# -------------------------
# Load modules
# -------------------------
module load python/3.12-conda
module load cuda/12.4
module load cudnn/9.2.0.82-12

# -------------------------
# Activate environment
# -------------------------
conda activate thesis

# -------------------------
# REDIRECT ALL CACHES TO /home/vault
# -------------------------
export HF_HOME=/home/vault/v121ca/v121ca21/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=/home/vault/v121ca/v121ca21/.cache/torch
export XDG_CACHE_HOME=/home/vault/v121ca/v121ca21/.cache

mkdir -p \
  $HF_HOME \
  $TRANSFORMERS_CACHE \
  $HF_DATASETS_CACHE \
  $TORCH_HOME \
  $XDG_CACHE_HOME

# -------------------------
# Run training
# ------------------------
python thesis_adapt/scripts/train_deberta_contarga_fewshot.py \
  --data_dir thesis_adapt/data/contarga \
  --out_dir thesis_adapt/models/deberta_fixed_n50 \
  --model_path /home/hpc/v121ca/v121ca21/thesis_models/deberta_goemo \
  --seed 42 \
  --epochs 5 \
  --train_bs 8 \
  --eval_bs 16 \
  --save_total_limit 1


