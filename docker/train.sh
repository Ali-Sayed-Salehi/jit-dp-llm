#!/usr/bin/env bash
set -euo pipefail

nvidia-smi
free -h

echo "running the training script"
accelerate launch --config_file=llama/configs/accelerate_config_deepspeed.yaml llama/train.py --config llama/configs/train_config.yaml
echo "training finished"