#!/bin/bash

SCRATCH_BASE="/home/$USER/links/scratch"

export HF_HOME="$SCRATCH_BASE/huggingface/hf-home"
export HF_DATASETS_CACHE="$SCRATCH_BASE/huggingface/hf-datasets"
export PIP_CACHE_DIR="$SCRATCH_BASE/pip/pip-cache"
export TRITON_CACHE_DIR="$SCRATCH_BASE/triton_cache"
export TORCH_EXTENSIONS_DIR="$SCRATCH_BASE/torch-cache"

rm -rf TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR

echo "TRITON_CACHE_DIR:     ${TRITON_CACHE_DIR}"
echo "TORCH_EXTENSIONS_DIR: ${TORCH_EXTENSIONS_DIR}"
echo "HF_HOME:              ${HF_HOME}"
echo "HF_DATASETS_CACHE:    ${HF_DATASETS_CACHE}"
echo "PIP_CACHE_DIR:        ${PIP_CACHE_DIR}"