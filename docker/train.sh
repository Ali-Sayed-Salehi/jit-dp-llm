#!/usr/bin/env bash
set -euo pipefail

nvidia-smi
free -h

echo "running the training script"
FSDP_CPU_RAM_EFFICIENT_LOADING=1 accelerate launch --config_file=llama/configs/accelerate_config_fsdp.yaml llama/train.py --config llama/configs/train_config.yaml
echo "training finished"