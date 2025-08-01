#!/bin/bash

echo "Setting environment variables"

export HF_HOME="/home/$USER/scratch/.cache/huggingface/hf-home"
export HF_DATASETS_CACHE="/home/$USER/scratch/.cache/huggingface/hf-datasets"
export PIP_CACHE_DIR="/home/$USER/scratch/.cache/pip/pip-cache"
export TORCH_EXTENSIONS_DIR="/home/$USER/scratch/.cache/torch-cache"

echo "HF_HOME: ${HF_HOME}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"
