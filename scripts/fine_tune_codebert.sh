#!/encs/bin/tcsh

#SBATCH --job-name=codebert-finetune
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=pt,pg
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=100:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu16

echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_prepare.csh

echo "running the training script"
python /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/microsoft/codebert-base --live_metrics --dataset jit_small_struc_ast_meta --class_imbalance_fix oversampling --selection_metric recall@top_30%
echo "training finished"

echo "Cleaning up"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_cleanup.csh