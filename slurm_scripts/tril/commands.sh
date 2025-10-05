#!/bin/bash

REPO_ROOT="/home/$USER/links/scratch/repos/jit-dp-llm"

echo "Preparing training environment"
source /home/$USER/links/scratch/repos/jit-dp-llm/slurm_scripts/tril/train_prepare.sh


python /home/alis/links/scratch/repos/jit-dp-llm/analysis/eval_thresholds.py \
--json /home/alis/links/scratch/repos/jit-dp-llm/llama/inference/eval_2025-10-03_23-04-38/final_test_results_clm_seqcls.json \
--out /home/alis/links/scratch/repos/jit-dp-llm/analysis/eval_results.json \
# -p 0.10 0.20 \



