#!/bin/bash
#SBATCH --job-name=deberta_goemo
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_logs/deberta_goemo_%j.out

# go to project root
cd /home/hpc/v121ca/v121ca21

# IMPORTANT: use absolute python path (NO conda activate)
PYTHON=/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python

# sanity check (will appear in log)
echo "Using python at:"
$PYTHON --version
which $PYTHON

# run training
$PYTHON thesis_scripts/train_deberta_goemo.py
