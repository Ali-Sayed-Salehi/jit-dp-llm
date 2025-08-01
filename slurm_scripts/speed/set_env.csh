#!/encs/bin/tcsh

echo "setting env variables"
setenv HF_HOME /speed-scratch/$USER/huggingface/hf-home
setenv HF_DATASETS_CACHE /speed-scratch/$USER/huggingface/hf-datasets
setenv PIP_CACHE_DIR /speed-scratch/$USER/pip/pip-cache
setenv TRITON_CACHE_DIR /speed-scratch/$USER/triton_cache
echo "HF_HOME: ${HF_HOME}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"