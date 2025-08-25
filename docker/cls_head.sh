#!/usr/bin/env bash
set -euo pipefail

nvidia-smi

echo "Running the attach classification head script ..."

# Llama 4 backbone
# python make_classifier_from_lm.py \
#   --base_lm_path /models/my-org/llama4-8b-finetuned \
#   --seq_cls_config_path /configs/seq-cls-templates/llama4 \
#   --save_path /outputs/classifiers \
#   --llama4 \
# #   --dtype bf16

# llama
python llama/attach_classification_head.py \
  --base_lm_path LLMs/pretrained/causal-lm/test \
  --seq_cls_config_path LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --save_path LLMs/pretrained/sequence-classification/test \
#   --dtype fp16


echo "script finished"
