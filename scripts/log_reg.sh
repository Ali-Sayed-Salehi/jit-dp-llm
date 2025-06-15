#!/encs/bin/tcsh

#SBATCH --job-name=logistic-regression
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8s
#SBATCH --mem=128G
#SBATCH --time=20:00:00
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

echo "running logistic regression script . . ."
python /speed-scratch/a_s87063/repos/perf-pilot/llama/logistic_regression_grid_search.py --selection_metric f1_score
echo "optimization finished"
