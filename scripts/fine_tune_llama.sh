#!/encs/bin/tcsh

#SBATCH --job-name=llm-finetune
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=pg
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=50:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu32

echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_prepare.csh

echo "running the training script"
python /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --perf_data --model_name llama3-8b --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing
echo "training finished"

echo "Cleaning up"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_cleanup.csh