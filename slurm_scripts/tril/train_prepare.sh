#!/bin/bash

echo "loading modules"
module load arch/avx512
module load StdEnv/2023 
module load intel/2025.2.0
module load nvhpc/25.1 
module load openmpi/5.0.3
module load cuda/12.6 
module load python/3.11.5 
module load arrow/21.0.0 
module load gcc/14.3
module list

REPO_ROOT="/home/$USER/links/scratch/repos/jit-dp-llm"

echo "setting env variables"
source $REPO_ROOT/slurm_scripts/tril/set_env.sh

echo "activating venv"
source $REPO_ROOT/venv/bin/activate

# nvidia-smi
# free -h
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_ENABLE_HF_TRANSFER="1"

# ds_report
# transformers env
