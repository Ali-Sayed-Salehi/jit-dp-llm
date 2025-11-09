#!/encs/bin/tcsh

#SBATCH --job-name=train-codebert
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

nvidia-smi
free -h
setenv DS_SKIP_CUDA_CHECK 1
setenv PYTORCH_CUDA_ALLOC_CONF expandable_segments:True

echo "running the training script"
accelerate launch \
--config_file=/speed-scratch/a_s87063/repos/jit-dp-llm/llama/configs/accelerate_config_deepspeed.yaml \
--num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID /speed-scratch/a_s87063/repos/jit-dp-llm/llama/train.py \
--config /speed-scratch/a_s87063/repos/jit-dp-llm/llama/configs/templates/train_config_speed_codebert_perf.yaml

# python /speed-scratch/a_s87063/repos/jit-dp-llm/llama/train.py --config /speed-scratch/a_s87063/repos/jit-dp-llm/llama/configs/train_config.yaml
echo "training finished"
