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
setenv PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
# Each job gets its own datasets cache: concurrent jobs mapping the same
# dataset otherwise race on one cache file and die with SIGBUS.
if (! $?TMPDIR) setenv TMPDIR /tmp
setenv HF_DATASETS_CACHE $TMPDIR/hf_datasets_cache_$SLURM_JOB_ID

# One GPU, no DeepSpeed: the encoder and 1.5B-parameter candidates fit a 20 GB
# slice with gradient checkpointing, and DeepSpeed's CPU offload needs nvcc to
# build its ops, which the compute nodes do not expose. The Llama runs keep
# using train.sh with the DeepSpeed accelerate config.
echo "running the training script with $CONFIG"
python $REPO/llama/train.py --config $CONFIG

echo "training finished"
