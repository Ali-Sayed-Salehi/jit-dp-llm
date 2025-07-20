#!/usr/bin/bash

#SBATCH --job-name=causal-lm-llama-virya
#SBATCH --output=/home/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --partition=phys
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=168:00:00

echo "Preparing training environment"
source /home/$USER/repos/perf-pilot/slurm_scripts/virya/train_prepare.csh

echo "running the training script"
# python /home/a_s87063/repos/perf-pilot/llama/causal_lm.py --live_metrics --dataset_path eli5 --model_path /home/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/distilbert/distilgpt2 --quant --lora --bf16 --gradient_checkpointing --truncation_len 128 --debug
# python /home/a_s87063/repos/perf-pilot/llama/causal_lm.py --live_metrics --dataset_path /home/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /home/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Meta-Llama-3-8B --quant --lora --bf16 --gradient_checkpointing --truncation_len 50 --chunking_len 50 --debug
accelerate launch --config_file=/home/a_s87063/repos/perf-pilot/llama/configs/accelerate_config_multi_gpu.yaml --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID /home/a_s87063/repos/perf-pilot/llama/causal_lm.py --config /home/a_s87063/repos/perf-pilot/llama/configs/causal_lm_config.yaml
echo "training finished"
