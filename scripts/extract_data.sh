#!/encs/bin/tcsh

#SBATCH --job-name=extract-data
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=150:00:00
#SBATCH --account=pcr
#SBATCH --constraint=el9

echo "loading modules"
module load python/3.12.0/default
module load cuda/11.8/default
module list

setenv PIP_CACHE_DIR /speed-scratch/$USER/pip/pip-cache
echo "PIP_CACHE_DIR: ${PIP_CACHE_DIR}"

echo "activating venv"
source /speed-scratch/$USER/repos/perf-pilot/venv/bin/activate.csh

echo "running the data extraction script . . ."
# python /speed-scratch/a_s87063/repos/perf-pilot/github_api/get_jit_dataset.py --struc --ast --small --debug
python /speed-scratch/a_s87063/repos/perf-pilot/llama/data_preparation.py --mode apachejit_llm --debug
echo "extraction finished"
