#!/bin/bash

#SBATCH --job-name=inference-tril
#SBATCH --output=/home/alis/links/scratch/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --account=def-pcr
#SBATCH --partition=debug

SCRATCH_BASE="/home/$USER/links/scratch"

echo "Preparing training environment"
source $SCRATCH_BASE/repos/perf-pilot/slurm_scripts/tril/train_prepare.sh

echo "running the inference script ..."

python $SCRATCH_BASE/repos/perf-pilot/llama/run_inference.py \
  --base_model_path $SCRATCH_BASE/repos/perf-pilot/LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --model_path $SCRATCH_BASE/repos/perf-pilot/llama/training/run_2025-09-11_18-38-24/output/checkpoint-500 \
  --dataset_path $SCRATCH_BASE/repos/perf-pilot/datasets/apachejit/apachejit_small_llm_struc.jsonl \
  --mixed_precision bf16 \
  # --truncation_len 22000 \
  # --quant \
  # --debug

echo "inference finished"
