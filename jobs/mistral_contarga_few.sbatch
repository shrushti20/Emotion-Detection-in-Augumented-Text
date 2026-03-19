#!/bin/bash
#SBATCH --job-name=mistral_few
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_logs/mistral_few_%j.out

cd /home/hpc/v121ca/v121ca21
/home/hpc/v121ca/v121ca21/.conda/envs/thesis/bin/python \
  thesis_scripts/contarga_llm_mistral_modes.py \
  --mode few \
  --fewshot thesis_scripts/prompts/contarga_fewshot.txt \
  --out thesis_results/llm/contarga/mistral_few.csv
