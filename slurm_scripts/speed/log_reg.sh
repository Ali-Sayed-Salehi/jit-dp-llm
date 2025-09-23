#!/encs/bin/tcsh

#SBATCH --job-name=logistic-regression
#SBATCH --output=/speed-scratch/a_s87063/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --account=pcr
#SBATCH --constraint=el9


echo "Preparing training environment"
source /speed-scratch/a_s87063/repos/jit-dp-llm/slurm_scripts/speed/train_prepare.csh

echo "running logistic regression script . . ."
python /speed-scratch/a_s87063/repos/jit-dp-llm/llama/logistic_regression_grid_search.py --selection_metric recall@top_30%
echo "optimization finished"
