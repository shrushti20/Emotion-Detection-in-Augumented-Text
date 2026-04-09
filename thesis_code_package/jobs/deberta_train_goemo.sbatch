#!/bin/bash
#SBATCH --job-name=deberta_goemo
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_logs/deberta_goemo_%j.out

cd /home/hpc/v121ca/v121ca21
/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python thesis_scripts/train_deberta_goemo.py
