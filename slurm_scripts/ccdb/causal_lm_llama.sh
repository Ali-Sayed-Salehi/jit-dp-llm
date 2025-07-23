#!/bin/bash

#SBATCH --job-name=causal-lm-llama-ccdb
#SBATCH --output=/home/alis/scratch/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=def-pcr

echo "Preparing training environment"
source /home/$USER/scratch/perf-pilot/slurm_scripts/speed/train_prepare.csh

nvidia-smi

echo "running the training script"
# python /home/alis/scratch/repos/perf-pilot/llama/causal_lm.py --live_metrics --dataset_path eli5 --model_path /home/alis/scratch/repos/perf-pilot/LLMs/pretrained/causal-lm/distilbert/distilgpt2 --quant --lora --bf16 --gradient_checkpointing --truncation_len 128 --debug
python /home/alis/scratch/repos/perf-pilot/llama/causal_lm.py --live_metrics --dataset_path /home/alis/scratch/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /home/alis/scratch/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Meta-Llama-3-8B --quant --lora --bf16 --gradient_checkpointing --truncation_len 50 --chunking_len 50 --debug
accelerate launch --config_file=/home/alis/scratch/repos/perf-pilot/llama/configs/accelerate_config.yaml --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID /home/alis/scratch/repos/perf-pilot/llama/causal_lm.py --config /home/alis/scratch/repos/perf-pilot/llama/configs/causal_lm_config.yaml
echo "training finished"
