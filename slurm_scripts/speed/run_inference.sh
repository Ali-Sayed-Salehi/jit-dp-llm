#!/encs/bin/tcsh

#SBATCH --job-name=inference-mbert-perf
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
  --base_model_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/snapshots/answerdotai/ModernBERT-large \
  --model_path /speed-scratch/a_s87063/repos/jit-dp-llm/llama/training/run_2025-10-31_23-19-02_perf_mbert/output/checkpoint-70 \
  --dataset_path /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf/perf_llm_struc.jsonl \
  --mixed_precision bf16 \
  --eval_ds_as_final_test \
  # --truncation_len 22000 \
  # --quant \
  # --debug

echo "inference finished"
