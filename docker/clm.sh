#!/usr/bin/env bash
set -euo pipefail

nvidia-smi

echo "running the training script"
ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch --config_file=llama/configs/accelerate_config_deepspeed.yaml llama/causal_lm.py --config llama/configs/causal_lm_config.yaml
echo "training finished"