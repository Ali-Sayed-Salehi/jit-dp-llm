#!/bin/bash

REPO_ROOT="/home/$USER/links/scratch/repos/jit-dp-llm"

echo "Preparing training environment"
source /home/$USER/links/scratch/repos/jit-dp-llm/slurm_scripts/tril/train_prepare.sh


# python /home/alis/links/scratch/repos/jit-dp-llm/analysis/eval_thresholds.py \
# --json /home/alis/links/scratch/repos/jit-dp-llm/llama/inference/eval_2025-10-03_23-04-38/final_test_results_clm_seqcls.json \
# --out /home/alis/links/scratch/repos/jit-dp-llm/analysis/eval_results.json \
# # -p 0.10 0.20 \


pip install chromadb sentence-transformers pydantic python-dotenv tiktoken openai# python /home/alis/links/scratch/repos/jit-dp-llm/analysis/token_length_violin.py \
# --dataset /home/alis/links/scratch/repos/jit-dp-llm/datasets/apachejit/apachejit_total_llm_struc_meta.jsonl \
# --model /home/alis/links/scratch/repos/jit-dp-llm/LLMs/snapshots/meta-llama/Llama-3.1-8B \
# --configs /home/alis/links/scratch/repos/jit-dp-llm/analysis/seq_len_confis.jsonl \
# --field prompt \
# --logy \
# --output /home/alis/links/scratch/repos/jit-dp-llm/analysis/results/violin_plot

python /home/alis/links/scratch/repos/jit-dp-llm/analysis/simulate_perf_reg_predictor.py