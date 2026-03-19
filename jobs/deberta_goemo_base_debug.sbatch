#!/bin/bash
#SBATCH --job-name=deberta_base_goemo
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_logs/deberta_base_goemo_%j.out

cd /home/hpc/v121ca/v121ca21

# call python from your conda env directly (no module/conda inside sbatch needed)
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python thesis_scripts/train_deberta_goemo.py
