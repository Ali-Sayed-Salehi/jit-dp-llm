#!/encs/bin/tcsh

#SBATCH --job-name=cls-head-speed
#SBATCH --output=/speed-scratch/a_s87063/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=168:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu20

echo "Preparing training environment"
source /speed-scratch/a_s87063/repos/jit-dp-llm/slurm_scripts/speed/train_prepare.csh

echo "Running the attach classification head script ..."

# Llama 4 backbone
# python make_classifier_from_lm.py \
#   --base_lm_path /models/my-org/llama4-8b-finetuned \
#   --seq_cls_config_path /configs/seq-cls-templates/llama4 \
#   --save_path /outputs/classifiers \
#   --llama4 \
# #   --dtype bf16

# llama
python /speed-scratch/a_s87063/repos/jit-dp-llm/llama/attach_classification_head.py \
  --base_lm_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/pretrained/causal-lm/llama3.1_jit_defects4j \
  --seq_cls_config_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/snapshots/meta-llama/Llama-3.1-8B \
  --save_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/pretrained/sequence-classification/llama3.1_jit_defects4j \
#   --dtype fp16

echo "script finished"