#!/encs/bin/bash

#SBATCH --job-name=llama-finetune          # Job name
#SBATCH --output=/speed-scratch/${USER}/repos/perf-pilot/slurm_jobs/logs/output_%j.log        # Output log (%j = job ID)
#SBATCH --error=/speed-scratch/${USER}/repos/perf-pilot/slurm_jobs/logs/error_%j.log          # Error log
#SBATCH --partition=pt                     # Partition/queue name
#SBATCH --gpus=2                           # Number of GPUs
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task
#SBATCH --mem=32G                          # RAM
#SBATCH --time=8:00:00                    # Time limit (HH:MM:SS)
#SBATCH --account=pcr                      # Account name
#SBATCH --constraint=el9

echo "loading modules"
module load python/3.12.0/default
module load cuda/11.8/default
module list

echo "setting env variables"
source /speed-scratch/${USER}/repos/perf-pilot/scripts/set_env
echo "HF_HOME: ${HF_HOME}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"

echo "activating venv"
source /speed-scratch/$USER/repos/perf-pilot/venv/bin/activate
echo "pyhthon interpreter: $(which python)"

echo "running the training script"
python /speed-scratch/${USER}/repos/perf-pilot/llama/fine_tune_classification.py
echo "training finished"