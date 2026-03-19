#!/bin/bash
#SBATCH --job-name=db_fix_n95
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=thesis_adapt/logs/deberta_fixed_n95_%j.out
#SBATCH --error=thesis_adapt/logs/deberta_fixed_n95_%j.err

# -------------------------
# Environment setup
# -------------------------
source /etc/profile
module load python/3.12-conda
module load cuda/12.4
module load cudnn/9.2.0.82-12

conda activate thesis

# -------------------------
# Cache (use /home/vault)
# -------------------------
export HF_HOME=/home/vault/v121ca/v121ca21/.cache/huggingface
export TORCH_HOME=/home/vault/v121ca/v121ca21/.cache/torch
export XDG_CACHE_HOME=/home/vault/v121ca/v121ca21/.cache

mkdir -p $HF_HOME $TORCH_HOME $XDG_CACHE_HOME

# -------------------------
# Run training (DeBERTa fixed n95)
# -------------------------
python thesis_adapt/scripts/train_deberta_contarga_fewshot.py \
  --data_dir thesis_adapt/data_fixed_n95 \
  --model_path thesis_models/deberta_goemo \
  --out_dir thesis_adapt/models/deberta_fixed_n95 \
  --epochs 10 \
  --train_bs 8 \
  --eval_bs 16 \
  --save_total_limit 1 \
  --seed 42

