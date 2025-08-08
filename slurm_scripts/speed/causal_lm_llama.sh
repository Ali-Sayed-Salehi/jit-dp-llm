#!/encs/bin/tcsh

#SBATCH --job-name=causal-lm-llama-speed
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu20

echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/slurm_scripts/speed/train_prepare.csh

nvidia-smi

echo "running the training script"
accelerate launch --config_file=/speed-scratch/a_s87063/repos/perf-pilot/llama/configs/accelerate_config_deepspeed.yaml --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID /speed-scratch/a_s87063/repos/perf-pilot/llama/causal_lm.py --config /speed-scratch/a_s87063/repos/perf-pilot/llama/configs/causal_lm_config.yaml
echo "training finished"
