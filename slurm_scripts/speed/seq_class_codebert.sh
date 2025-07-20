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
source /speed-scratch/$USER/repos/perf-pilot/slurm_scripts/speed/train_prepare.csh

echo "running the training script"
# python /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/microsoft/codebert-base --live_metrics --dataset_path /speed-scratch/a_s87063/repos/perf-pilot/datasets/mozilla_perf/perf_llm_struc.jsonl --class_imbalance_fix oversampling --selection_metric recall@top_30% --debug
accelerate launch --config_file=/speed-scratch/a_s87063/repos/perf-pilot/llama/configs/accelerate_config.yaml --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --config /speed-scratch/a_s87063/repos/perf-pilot/llama/configs/sequence_classification_config_codebert.yaml
echo "training finished"
