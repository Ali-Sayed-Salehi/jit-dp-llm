#!/usr/bin/env bash
set -euo pipefail

echo "running the inference script ..."

python llama/run_inference.py \
  --model_path LLMs/snapshots/meta-llama/Llama-3.1-70B \
  --dataset_path datasets/jit_defects4j/jit_defects4j_small_llm_struc.jsonl \
  --mixed_precision fp16 \
  --batch_size 1 \
  --truncation_len 5000 \
  --quant \
  --debug

echo "inference finished"