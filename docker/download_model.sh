#!/usr/bin/env bash
set -euo pipefail

python llama/download_model.py --model_id meta-llama/Llama-3.1-8B --model_head sequence-classification