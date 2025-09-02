#!/usr/bin/env bash
set -euo pipefail

echo "running the script ..."

## CLM
# python llama/merge_lora.py \
#   --base_model LLMs/snapshots/meta-llama/Llama-3.1-8B \
#   --task causal-lm \
#   --adapter_path llama/training/run_2025-08-25_01-48-36/model \
#   --save_path LLMs/pretrained/causal-lm/test \
#   # --dtype fp32

## Sequence classification
python llama/merge_lora.py \
  --task seq-cls \
  --base_model LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --adapter_path llama/adapter_models/model \
  --save_path LLMs/trained/sequence-classification/llama3.1_8B_apachejit_small \
  --dtype fp16

echo "merging finished"