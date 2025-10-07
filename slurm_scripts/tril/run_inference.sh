#!/bin/bash

#SBATCH --job-name=inference-mean-pool
#SBATCH --output=/home/alis/links/scratch/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --account=def-pcr
#SBATCH --partition=compute

REPO_ROOT="/home/$USER/links/scratch/repos/jit-dp-llm"

echo "Preparing training environment"
source $REPO_ROOT/slurm_scripts/tril/train_prepare.sh

echo "running the inference script ..."

# python $REPO_ROOT/llama/run_inference.py \
#   --base_model_path $REPO_ROOT/LLMs/snapshots/meta-llama/Llama-3.1-8B \
#   --model_path $REPO_ROOT/llama/training/run_2025-09-11_18-38-24/output/checkpoint-500 \
#   --dataset_path $REPO_ROOT/datasets/apachejit/apachejit_small_llm_struc.jsonl \
#   --mixed_precision bf16 \
#   # --truncation_len 22000 \
#   # --quant \
#   # --debug

python $REPO_ROOT/llama/run_inference.py \
--base_model_path $REPO_ROOT/LLMs/snapshots/meta-llama/Llama-3.1-8B \
--model_path $REPO_ROOT/llama/training/run_2025-10-04_22-58-28/output/checkpoint-225 \
--dataset_path $REPO_ROOT/datasets/apachejit/apachejit_total_llm_struc_meta.jsonl \
--mixed_precision bf16 \
--truncation_len 22000 \
--quant \
--threshold 0.8 \
# --clm_for_seq_cls \
# --debug

echo "inference finished"
