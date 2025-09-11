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

echo "setting env variables"
source /home/$USER/links/scratch/repos/perf-pilot/slurm_scripts/tril/set_env.sh

echo "activating venv"
source /home/$USER/links/scratch/repos/perf-pilot/venv/bin/activate

nvidia-smi
free -h
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ds_report
transformers env
