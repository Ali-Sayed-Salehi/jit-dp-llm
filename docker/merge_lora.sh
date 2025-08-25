#!/usr/bin/env bash
set -euo pipefail

echo "running the script ..."

# CLM
# python llama/merge_lora.py \
#   --base_model LLMs/snapshots/meta-llama/Llama-3.1-8B \
#   --task causal-lm \
#   --adapter_path llama/training/run_2025-08-24_03-27-46/model \
#   --save_path LLMs/pretrained/causal-lm/test \
#   --dtype fp32

# Sequence classification
python llama/merge_lora.py \
  --task seq-cls \
  --base_model LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --adapter_path llama/training/run_2025-08-24_03-37-23/model \
  --save_path LLMs/pretrained/causal-lm/test2 \
  # --dtype fp32

echo "merging finished"