#!/bin/bash

#SBATCH --job-name=train-tril-clm-for-seqcls
#SBATCH --output=/home/alis/links/scratch/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --account=def-pcr
#SBATCH --partition=compute

REPO_ROOT="/home/$USER/links/scratch/repos/jit-dp-llm"

echo "Preparing training environment"
source $REPO_ROOT/slurm_scripts/tril/train_prepare.sh

echo "running the training script"
accelerate launch --config_file=$REPO_ROOT/llama/configs/accelerate_config_deepspeed.yaml --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID $REPO_ROOT/llama/train.py --config $REPO_ROOT/llama/configs/train_config_local.yaml
# python $REPO_ROOT/llama/train.py --config $REPO_ROOT/llama/configs/train_config.yaml
echo "training finished"
