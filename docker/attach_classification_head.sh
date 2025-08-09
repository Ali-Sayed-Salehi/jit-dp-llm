#!/usr/bin/env bash
set -euo pipefail

nvidia-smi

echo "Running the download/transform script ..."
python llama/attach_classification_head.py --base_lm_path LLMs/pretrained/causal-lm/meta-llama/Llama-3.1-8B --save_path LLMs/pretrained/sequence-classification/custom
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/attach_classification_head.py --llama4 --base_lm_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Llama-4-Scout-17B-16E --save_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom
echo "download/transform finished"
