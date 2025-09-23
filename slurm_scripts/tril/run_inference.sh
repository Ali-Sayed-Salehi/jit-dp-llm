#!/bin/bash

#SBATCH --job-name=inference-tril
#SBATCH --output=/home/alis/links/scratch/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --account=def-pcr
#SBATCH --partition=debug

REPO_ROOT="/home/$USER/links/scratch/repos/jit-dp-llm"

echo "Preparing training environment"
source $REPO_ROOT/slurm_scripts/tril/train_prepare.sh

echo "running the inference script ..."

python $REPO_ROOT/llama/run_inference.py \
  --base_model_path $REPO_ROOT/LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --model_path $REPO_ROOT/llama/training/run_2025-09-11_18-38-24/output/checkpoint-500 \
  --dataset_path $REPO_ROOT/datasets/apachejit/apachejit_small_llm_struc.jsonl \
  --mixed_precision bf16 \
  # --truncation_len 22000 \
  # --quant \
  # --debug

echo "inference finished"
