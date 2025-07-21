#!/usr/bin/bash

#SBATCH --job-name=seq-class-llama-virya
#SBATCH --output=/home/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --partition=phys
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=168:00:00

echo "Preparing training environment"
source /home/$USER/repos/perf-pilot/slurm_scripts/virya/train_prepare.sh

nvidia-smi

echo "running the training script"
# python /home/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --dataset_path /home/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /home/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom/meta-llama/Meta-Llama-3-8B --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing --selection_metric recall@top_30% --debug
# python /home/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --dataset_path /home/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /home/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/meta-llama/Llama-3.1-8B --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing --selection_metric recall@top_30% --truncation_len 14000 --debug
# python /home/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --dataset_path /home/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /home/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom/meta-llama/Llama-4-Scout-17B-16E --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing --selection_metric recall@top_30% --truncation_len 2000 --debug
# python /home/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --dataset_path /home/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /home/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/syzymon/long_llama_code_7b --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing --selection_metric recall@top_30% --truncation_len 14000 --debug
accelerate launch --config_file=/home/a_s87063/repos/perf-pilot/llama/configs/accelerate_config.yaml --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID /home/$USER/repos/perf-pilot/llama/sequence_classification.py --config /home/a_s87063/repos/perf-pilot/llama/configs/sequence_classification_config_llama_3.1.yaml
# python /home/$USER/repos/perf-pilot/llama/sequence_classification.py --config /home/a_s87063/repos/perf-pilot/llama/configs/sequence_classification_config.yaml
echo "training finished"
