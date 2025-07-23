#!/bin/bash

#SBATCH --job-name=download-model-ccdb
#SBATCH --output=/home/alis/scratch/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --partition=migs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-04:00

echo "Preparing training environment"
source /home/alis/scratch/perf-pilot/slurm_scripts/virya/train_prepare.sh

echo "Running the download script ..."
python /home/alis/scratch/repos/perf-pilot/llama/download_model.py --model_id meta-llama/Llama-3.1-8B --model_head causal-lm
# python /home/alis/scratch/repos/perf-pilot/llama/download_model.py --model_id syzymon/long_llama_code_7b --model_head sequence-classification
# python /home/alis/scratch/repos/perf-pilot/llama/attach_classification_head.py --base_lm_path /home/alis/scratch/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Llama-3.1-8B --save_path /home/alis/scratch/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom
# python /home/alis/scratch/repos/perf-pilot/llama/attach_classification_head_llama4.py --base_lm_path /home/alis/scratch/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Llama-4-Scout-17B-16E --save_path /home/alis/scratch/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom
echo "download/transform finished"
