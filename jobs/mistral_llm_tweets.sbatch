#!/bin/bash
#SBATCH --job-name=mistral_tweets
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_logs/mistral_tweets_%j.out

cd /home/hpc/v121ca/v121ca21

/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python thesis_scripts/tweets_llm_mistral.py
