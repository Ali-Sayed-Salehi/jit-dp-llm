#!/usr/bin/env bash
set -euo pipefail

nvidia-smi

echo "running the training script"
accelerate launch --config_file=llama/configs/accelerate_config_deepspeed.yaml llama/causal_lm.py --config llama/configs/causal_lm_config.yaml
echo "training finished"