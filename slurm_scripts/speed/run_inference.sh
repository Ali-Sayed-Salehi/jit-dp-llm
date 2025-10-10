#!/encs/bin/tcsh

#SBATCH --job-name=inference-debug
#SBATCH --output=/speed-scratch/a_s87063/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu20


echo "Preparing training environment"
source /speed-scratch/a_s87063/repos/jit-dp-llm/slurm_scripts/speed/train_prepare.csh

echo "running the inference script ..."

python /speed-scratch/a_s87063/repos/jit-dp-llm/llama/run_inference.py \
  --base_model_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --model_path /speed-scratch/a_s87063/repos/jit-dp-llm/llama/training/run_2025-09-29_22-14-38/output/checkpoint-425 \
  --dataset_path /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/apachejit/apachejit_total_llm_struc_meta.jsonl \
  --mixed_precision bf16 \
  --truncation_len 22000 \
  --quant \
  --debug

echo "inference finished"
