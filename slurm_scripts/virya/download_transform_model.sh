#!/usr/bin/bash

#SBATCH --job-name=download-model-virya
#SBATCH --output=/home/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --partition=migs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/slurm_scripts/speed/train_prepare.sh

echo "Running the download script ..."
python /speed-scratch/a_s87063/repos/perf-pilot/llama/download_model.py --model_id meta-llama/Llama-3.1-8B --model_head causal-lm
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/download_model.py --model_id syzymon/long_llama_code_7b --model_head sequence-classification
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/attach_classification_head.py --base_lm_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Llama-3.1-8B --save_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/attach_classification_head_llama4.py --base_lm_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Llama-4-Scout-17B-16E --save_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom
echo "training finished"
