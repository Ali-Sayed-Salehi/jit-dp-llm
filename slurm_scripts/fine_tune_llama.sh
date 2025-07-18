#!/encs/bin/tcsh

#SBATCH --job-name=llama-finetune
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=150:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu20

echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_prepare.csh

echo "running the training script"
python /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --dataset_path /speed-scratch/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom/meta-llama/Meta-Llama-3-8B --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing --selection_metric recall@top_30% --debug
# python /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --dataset_path /speed-scratch/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/meta-llama/Llama-3.1-8B --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing --selection_metric recall@top_30% --truncation_len 14000 --debug
python /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --dataset_path /speed-scratch/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom/meta-llama/Llama-4-Scout-17B-16E --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing --selection_metric recall@top_30% --truncation_len 2000 --debug
python /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --live_metrics --dataset_path /speed-scratch/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/syzymon/long_llama_code_7b --class_imbalance_fix focal_loss --quant --lora --bf16 --gradient_checkpointing --selection_metric recall@top_30% --truncation_len 14000 --debug
accelerate launch --config_file=/speed-scratch/a_s87063/repos/perf-pilot/llama/configs/accelerate_config.yaml --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --config /speed-scratch/a_s87063/repos/perf-pilot/llama/configs/sequence_classification_config.yaml
python /speed-scratch/$USER/repos/perf-pilot/llama/sequence_classification.py --config /speed-scratch/a_s87063/repos/perf-pilot/llama/configs/sequence_classification_config.yaml

echo "training finished"

echo "Cleaning up"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_cleanup.csh