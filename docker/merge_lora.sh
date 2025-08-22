#!/usr/bin/env bash
set -euo pipefail

echo "running the script ..."

# CLM
# python llama/merge_lora.py \
#   --base_model LLMs/snapshots/meta-llama/Llama-3.1-8B \
#   --adapter_path llama/training/run_2025-08-21_17-12-54/model \
#   --save_path LLMs/pretrained/causal-lm/test \
#   --dtype fp32

# Sequence classification
python llama/merge_lora.py \
  --task seq-cls \
  --base_model LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --adapter_path llama/training/run_2025-08-21_17-12-54/model \
  --save_path LLMs/pretrained/causal-lm/test1 \
  --dtype fp32

echo "merging finished"