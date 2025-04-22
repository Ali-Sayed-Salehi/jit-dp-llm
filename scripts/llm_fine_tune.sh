#!/encs/bin/tcsh

#SBATCH --job-name=llm-finetune
#SBATCH --mail-type=ALL
#SBATCH --partition=pt
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --account=pcr
#SBATCH --constraint=el9

echo "loading modules"
module load python/3.12.0/default
module load cuda/11.8/default
module list

echo "setting env variables"
source /speed-scratch/$USER/repos/perf-pilot/scripts/set_env.csh
echo "HF_HOME: ${HF_HOME}"
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"

echo "activating venv"
source /speed-scratch/$USER/repos/perf-pilot/venv/bin/activate.csh
echo "pyhthon interpreter: `which python`"

echo "running the training script"
python /speed-scratch/$USER/repos/perf-pilot/llama/fine_tune_classification.py
echo "training finished"