#!/encs/bin/tcsh

# setenv HF_HOME /speed-scratch/$USER/huggingface/hf-home
# setenv HF_DATASETS_CACHE /speed-scratch/$USER/huggingface/hf-datasets
# setenv PIP_CACHE_DIR /speed-scratch/$USER/pip/pip-cache
setenv TRITON_CACHE_DIR /speed-scratch/$USER/triton_cache
# setenv TORCH_EXTENSIONS_DIR /speed-scratch/$USER/torch-cache

echo "TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"