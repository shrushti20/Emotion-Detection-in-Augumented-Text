#!/bin/bash
#SBATCH --job-name=roberta_ctg_fs
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_adapt/logs/roberta_contarga_fewshot_%j.out

cd /home/hpc/v121ca/v121ca21

# run using your conda env python directly (works inside sbatch even when "module" isn't available)
 /home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python thesis_adapt/scripts/train_roberta_contarga_fewshot.py
