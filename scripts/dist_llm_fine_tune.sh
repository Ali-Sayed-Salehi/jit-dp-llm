#!/encs/bin/tcsh

#SBATCH --job-name=dist-llm-finetune
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --account=pcr

if ( $?SLURM_CPUS_PER_TASK ) then
   setenv omp_threads $SLURM_CPUS_PER_TASK
else
   setenv omp_threads 1
endif
setenv OMP_NUM_THREADS $omp_threads

setenv RDZV_HOST `hostname -s`
setenv RDZV_PORT 29500
setenv endpoint ${RDZV_HOST}:${RDZV_PORT}
setenv CUDA_LAUNCH_BLOCKING 1
setenv NCCL_BLOCKING_WAIT 1
#setenv NCCL_DEBUG INFO
setenv NCCL_P2P_DISABLE 1
setenv NCCL_IB_DISABLE 1

echo "Using $RDZV_HOST as rendezvous server at port $RDZV_PORT"


echo "Preparing training environment"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_prepare.csh

unsetenv CUDA_VISIBLE_DEVICES

echo "running the training script"
srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$endpoint /speed-scratch/a_s87063/repos/perf-pilot/llama/fine_tune_classification.py
echo "training finished"

echo "Cleaning up"
source /speed-scratch/$USER/repos/perf-pilot/scripts/train_cleanup.csh