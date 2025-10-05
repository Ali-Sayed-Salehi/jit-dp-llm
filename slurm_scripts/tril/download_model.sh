#!/bin/bash

REPO_ROOT="/home/$USER/links/scratch/repos/jit-dp-llm"

echo "Preparing training environment"
source /home/$USER/links/scratch/repos/jit-dp-llm/slurm_scripts/tril/train_prepare.sh

export HF_HUB_ENABLE_HF_TRANSFER="1"

python /home/$USER/links/scratch/repos/jit-dp-llm/llama/download_model.py --model_id meta-llama/Llama-4-Scout-17B-16E