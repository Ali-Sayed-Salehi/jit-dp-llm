#!/usr/bin/env bash
set -euo pipefail

nvidia-smi

echo "running the training script"
accelerate launch --config_file=llama/configs/accelerate_config_deepspeed.yaml llama/sequence_classification.py --config llama/configs/sequence_classification_config_llama_3.1.yaml
echo "training finished"