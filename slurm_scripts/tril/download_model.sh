#!/bin/bash


REPO_ROOT="/home/$USER/links/scratch/repos/jit-dp-llm"

echo "Preparing training environment"
source $REPO_ROTT/slurm_scripts/tril/train_prepare.sh

python $REPO_ROTT/llama/download_model.py --model_id meta-llama/Llama-3.1-70B