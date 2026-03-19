#!/bin/bash
#SBATCH --job-name=rb_fix_n10
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_adapt/logs/roberta_fixed_n10_%j.out

cd /home/hpc/v121ca/v121ca21
/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python thesis_adapt/scripts/train_roberta_contarga_fewshot.py \
  --data_dir /home/hpc/v121ca/v121ca21/thesis_adapt/data_fixed_n10 \
  --out_dir  /home/hpc/v121ca/v121ca21/thesis_adapt/models/roberta_fixed_n10
