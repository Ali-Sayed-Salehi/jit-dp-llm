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
  --model_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/test \
  --dataset_path /speed-scratch/a_s87063/repos/perf-pilot/datasets/jit_defects4j/jit_defects4j_small_llm_struc.jsonl \
  --mixed_precision bf16 \
  --batch_size 1 \
  --truncation_len 20000 \
  --quant \
  --debug

echo "inference finished"
