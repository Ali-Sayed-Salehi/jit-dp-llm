#!/bin/bash

export HF_HOME="/home/$USER/links/scratch/huggingface/hf-home"
export HF_DATASETS_CACHE="/home/$USER/links/scratch/huggingface/hf-datasets"
export PIP_CACHE_DIR="/home/$USER/links/scratch/pip/pip-cache"
export TRITON_CACHE_DIR="/home/$USER/links/scratch/triton_cache"
export TORCH_EXTENSIONS_DIR="/home/$USER/links/scratch/torch-cache"

echo "TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
echo "HF_HOME: ${HF_HOME}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"