#!/encs/bin/tcsh

echo "loading modules"
module load python/3.12.0/default
module load cuda/12.8/default
module list

echo "setting env variables"
# setenv HF_HOME /speed-scratch/$USER/huggingface/hf-home
# setenv HF_DATASETS_CACHE /speed-scratch/$USER/huggingface/hf-datasets
# setenv PIP_CACHE_DIR /speed-scratch/$USER/pip/pip-cache
# setenv TRITON_CACHE_DIR /speed-scratch/$USER/triton_cache
echo "HF_HOME: ${HF_HOME}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"

echo "activating venv"
source /speed-scratch/$USER/repos/perf-pilot/venv/bin/activate.csh
# echo "Python interpreter: `which python`"
