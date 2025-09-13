#!/encs/bin/tcsh

#SBATCH --job-name=run-inference-speed
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=168:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu20

echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/slurm_scripts/speed/train_prepare.csh

echo "running the inference script ..."

python /speed-scratch/a_s87063/repos/perf-pilot/llama/run_inference.py \
  --base_model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --model_path /speed-scratch/a_s87063/repos/perf-pilot/llama/training/run_2025-09-11_15-56-45/output/checkpoint-150 \
  --dataset_path /speed-scratch/a_s87063/repos/perf-pilot/datasets/apachejit/apachejit_small_llm_struc.jsonl \
  --mixed_precision bf16 \
  --truncation_len 22000 \
  --quant \
  # --debug

echo "inference finished"
