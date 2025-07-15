#!/encs/bin/tcsh

#SBATCH --job-name=llama-finetune
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu32

echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_prepare.csh

echo "running the training script"
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/causal_lm.py --live_metrics --dataset_path eli5 --model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/distilbert/distilgpt2 --quant --lora --bf16 --gradient_checkpointing --truncation_len 128 --debug
python /speed-scratch/a_s87063/repos/perf-pilot/llama/causal_lm.py --live_metrics --dataset_path /speed-scratch/a_s87063/repos/perf-pilot/datasets/jit_dp/apachejit_llm_small_struc.jsonl --model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Meta-Llama-3-8B --quant --lora --bf16 --gradient_checkpointing --truncation_len 50 --chunking_len 50 --debug
echo "training finished"

echo "Cleaning up"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_cleanup.csh