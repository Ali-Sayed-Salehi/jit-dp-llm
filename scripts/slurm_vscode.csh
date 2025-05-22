# run interactive job with GPU
# THIS SHOULD BE RUN FIRST AND SEPARATELY FROM OTHER COMMANDS
salloc --partition=pt --constraint=el9 --mem=10G --gpus=2 -A pcr

#prepare environment
cd /speed-scratch/$USER/vscode
setenv XDG_RUNTIME_DIR /speed-scratch/$USER/run-user

# load required modules
module load python/3.12.0/default
module load cuda/11.8/default

# set environment variables
source /speed-scratch/$USER/repos/perf-pilot/scripts/set_env.csh

# run vscode
/speed-scratch/nag-public/code-server-4.22.1/bin/code-server --user-data-dir=$PWD\/projects --config=$PWD\/home/.config/code-server/config.yaml --bind-addr="0.0.0.0:8890" /speed-scratch/$USER/repos/perf-pilot

# create tunnel to slurm node on user machine
ssh -L 8890:speed-17:8890 slurm

# check the password to open vscode in the browser
cat /speed-scratch/$USER/vscode/home/.config/code-server/config.yaml

# all the commands as a single one
cd /speed-scratch/$USER/vscode ; setenv XDG_RUNTIME_DIR /speed-scratch/$USER/run-user ; module load python/3.12.0/default ; module load cuda/11.8/default ; source /nfs/home/a/a_s87063/set_env.csh; /speed-scratch/nag-public/code-server-4.22.1/bin/code-server --user-data-dir=$PWD\/projects --config=$PWD\/home/.config/code-server/config.yaml --bind-addr="0.0.0.0:8890" /speed-scratch/$USER/repos/perf-pilot


# submit a batch job for single GPU
sbatch /speed-scratch/$USER/repos/perf-pilot/scripts/fine_tune_codebert.sh

# submit a batch job for multiple GPUs using multiple nodes
sbatch /speed-scratch/$USER/repos/perf-pilot/scripts/dist_llm_fine_tune.sh

# see slurm job logs
cd /speed-scratch/$USER/repos/perf-pilot/slurm_jobs

# get the status for a job
sacct -j <job_id> --format=JobID,State,ExitCode,Elapsed