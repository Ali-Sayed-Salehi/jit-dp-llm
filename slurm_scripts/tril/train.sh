#!/bin/bash

#SBATCH --job-name=train-tril
#SBATCH --output=/home/$USER/links/scratch/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --account=def-pcr
#SBATCH --partition=compute

echo "Preparing training environment"
source /home/$USER/links/scratch/repos/perf-pilot/slurm_scripts/tril/train_prepare.sh

echo "running the training script"
accelerate launch --config_file=/home/alis/links/scratch/repos/perf-pilot/llama/configs/accelerate_config_deepspeed.yaml --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID /home/alis/links/scratch/repos/perf-pilot/llama/train.py --config /home/alis/links/scratch/repos/perf-pilot/llama/configs/train_config.yaml
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/train.py --config /speed-scratch/a_s87063/repos/perf-pilot/llama/configs/train_config.yaml
echo "training finished"
