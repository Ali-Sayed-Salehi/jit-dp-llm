#!/bin/bash

echo "Preparing training environment"
source /home/$USER/links/scratch/repos/perf-pilot/slurm_scripts/tril/train_prepare.sh

python /home/$USER/links/scratch/repos/perf-pilot/llama/download_model.py --model_id meta-llama/Llama-3.1-70B