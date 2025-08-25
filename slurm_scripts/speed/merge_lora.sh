#!/encs/bin/tcsh

#SBATCH --job-name=merge-lora-speed
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

echo "running the script ..."

# CLM
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/merge_lora.py \
#   --base_model /speed-scratch/a_s87063/repos/perf-pilot/LLMs/snapshots/meta-llama/Llama-3.1-8B \
#   --task causal-lm \
#   --adapter_path /speed-scratch/a_s87063/repos/perf-pilot/llama/training/run_2025-08-25_14-40-59/model \
#   --save_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/test \
#   # --dtype fp32

# Sequence classification
python /speed-scratch/a_s87063/repos/perf-pilot/llama/merge_lora.py \
  --task seq-cls \
  --base_model /speed-scratch/a_s87063/repos/perf-pilot/LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --adapter_path /speed-scratch/a_s87063/repos/perf-pilot/llama/training/run_2025-08-25_14-40-59/model \
  --save_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/test \
  # --dtype fp32

echo "merging finished"
