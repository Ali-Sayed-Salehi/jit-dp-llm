#!/encs/bin/tcsh

#SBATCH --job-name=download-model
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu16

echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_prepare.csh

echo "downloading LLM ..."
python /speed-scratch/a_s87063/repos/perf-pilot/llama/download_model.py --model_id meta-llama/Llama-4-Scout-17B-16E --add_classification_head
echo "training finished"

echo "Cleaning up"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_cleanup.csh