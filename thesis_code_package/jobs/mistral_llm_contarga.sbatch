#!/bin/bash
#SBATCH --job-name=mistral_llm
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/hpc/v121ca/v121ca21/thesis_logs/mistral_llm_%j.out

module load python/3.12-conda
module load cuda/12.4
module load cudnn/9.2.0.82-12
conda activate thesis

cd /home/hpc/v121ca/v121ca21

python thesis_scripts/contarga_llm_mistral_multilabel.py
