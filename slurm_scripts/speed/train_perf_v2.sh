#!/encs/bin/tcsh

#SBATCH --job-name=train-perf-v2
#SBATCH --output=/speed-scratch/a_s87063/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu20

# Usage: sbatch train_perf_v2.sh <training config yaml>
# e.g.   sbatch train_perf_v2.sh llama/configs/templates/train_config_speed_mbert_perf_v2.yaml
set REPO = /speed-scratch/a_s87063/repos/jit-dp-llm
if ($#argv < 1) then
    echo "usage: sbatch train_perf_v2.sh <training config yaml>"
    exit 1
endif
set CONFIG = $argv[1]
if ("$CONFIG" !~ /*) set CONFIG = $REPO/$CONFIG

echo "Preparing training environment"
source $REPO/slurm_scripts/speed/train_prepare.csh

nvidia-smi
free -h
setenv DS_SKIP_CUDA_CHECK 1
setenv PYTORCH_CUDA_ALLOC_CONF expandable_segments:True

echo "running the training script with $CONFIG"
accelerate launch \
--config_file=$REPO/llama/configs/accelerate_config_deepspeed.yaml \
--num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID $REPO/llama/train.py \
--config $CONFIG

echo "training finished"
