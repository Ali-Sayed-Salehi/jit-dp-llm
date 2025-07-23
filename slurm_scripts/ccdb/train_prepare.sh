#!/bin/bash

echo "loading modules"
module load python/default
module load cuda/default
module list

echo "Setting environment variables"
export HF_HOME="/home/$USER/scratch/.cache/huggingface/hf-home"
export HF_DATASETS_CACHE="/home/$USER/scratch/.cache/huggingface/hf-datasets"
export PIP_CACHE_DIR="/home/$USER/scratch/.cache/pip/pip-cache"
echo "HF_HOME: ${HF_HOME}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"

echo "activating venv"
source /home/$USER/scratch/repos/perf-pilot/venv/bin/activate
# echo "Python interpreter: `which python`"
