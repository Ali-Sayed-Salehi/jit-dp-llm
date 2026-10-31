#!/encs/bin/tcsh

#SBATCH --job-name=score-commits
#SBATCH --output=/speed-scratch/a_s87063/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu20

# Usage: sbatch score_commits.sh <checkpoint dir> <scoring set jsonl> <scores jsonl> [batch size] [base model dir]
# The checkpoint is a full sequence classification directory, or a LoRA
# adapter directory with the base model as the fifth argument. Rerunning
# with the same scores file resumes where the previous job stopped.
set REPO = /speed-scratch/a_s87063/repos/jit-dp-llm
if ($#argv < 3) then
    echo "usage: sbatch score_commits.sh <checkpoint dir> <scoring set jsonl> <scores jsonl> [batch size] [base model dir]"
    exit 1
endif
set MODEL = $argv[1]
set INPUT = $argv[2]
set OUTPUT = $argv[3]
set BATCH = 8
if ($#argv >= 4) set BATCH = $argv[4]
set BASE = ""
if ($#argv >= 5) set BASE = "--base_model_path $argv[5]"

echo "Preparing environment"
source $REPO/slurm_scripts/speed/train_prepare.csh
nvidia-smi

echo "scoring $INPUT with $MODEL into $OUTPUT"
python $REPO/llama/score_commits.py \
  --model_path $MODEL \
  --input $INPUT \
  --output $OUTPUT \
  --batch_size $BATCH \
  --mixed_precision bf16 $BASE

echo "scoring finished"
