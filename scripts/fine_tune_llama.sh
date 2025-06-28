#!/encs/bin/tcsh

#SBATCH --job-name=llama-finetune
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=150:00:00
#SBATCH --account=pcr
#SBATCH --constraint=el9

echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_prepare.csh

echo "running the training script"
python /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --dataset jit_small_struc_ast_meta --model_name meta-llama/Meta-Llama-3-8B --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing --selection_metric recall@top_30%
echo "training finished"

echo "Cleaning up"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_cleanup.csh