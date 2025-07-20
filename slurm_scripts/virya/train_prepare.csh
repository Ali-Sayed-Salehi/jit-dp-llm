#!usr/bin/bash

echo "loading modules"
source /etc/profile.d/modules.sh
module load python/3.11.6
module load cuda/12.5.0
module list

echo "env variables"
echo "HF_HOME: ${HF_HOME}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"

echo "activating venv"
source /home/$USER/repos/perf-pilot/venv/bin/activate
# echo "Python interpreter: $(which python)"
