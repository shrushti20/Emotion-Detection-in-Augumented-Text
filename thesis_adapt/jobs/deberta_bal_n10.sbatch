#!/bin/bash
#SBATCH --job-name=db_bal_n10
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_adapt/logs/deberta_bal_n10_%j.out

cd /home/hpc/v121ca/v121ca21

DEBERTA_CKPT="/home/hpc/v121ca/v121ca21/thesis_models/deberta_goemo/checkpoint-5000"

 /home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python thesis_adapt/scripts/train_deberta_contarga_fewshot.py \
  --data_dir /home/hpc/v121ca/v121ca21/thesis_adapt/data_bal_n10 \
  --out_dir  /home/hpc/v121ca/v121ca21/thesis_adapt/models/deberta_bal_n10 \
  --model_path "$DEBERTA_CKPT"
