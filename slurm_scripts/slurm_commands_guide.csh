# vscode
# run interactive job with GPU
# THIS SHOULD BE RUN FIRST AND SEPARATELY FROM OTHER COMMANDS

# SPEED

# 16 GB GPU
salloc --partition=pg --constraint=gpu16 --cpus-per-task=8 --mem=200G --gpus=1 -A pcr

# 20 GB GPU
salloc --partition=ps,pg,pt --constraint=gpu20 --cpus-per-task=8 --mem=200G --gpus=1 -A pcr

# 32 GB GPU
salloc --partition=ps,pg,pt --constraint=gpu32 --mem=200G --gpus=1 -A pcr

# 80 GB
salloc --partition=ps,pg,pt --gres=gpu:nvidia_a100_7g.80gb:1 --mem=200G -A pcr --time=01:00:00

-gres=gpu:A100.80gb:2
--gpus=4
--nodelist=speed-37,speed-38,speed-39,speed-40,speed-41


# CCDB
module load cuda/default; module load python/default; source /home/$USER/scratch/repos/perf-pilot/slurm_scripts/ccdb/set_env.sh;

# Cedar 
# V100 32gb (1 - 4)
salloc --nodes=1 --time=0-01:00 --mem=64G --account def-pcr --gpus-per-node=v100l:1

# Graham 
# v100 32GB (8)
salloc --nodes=1 --time=0-01:00 --mem=64G --account def-pcr --constraint=cascade,v100 --gpus-per-node=v100:1

# Beluga 
# V100 16gb (1 - 4)
salloc --nodes=1 --time=0-01:00 --mem=64G --account def-pcr --gpus=1

# Narval
# A100 40GB (4)
salloc --nodes=1 --time=0-01:00 --mem=64G --account def-pcr --gpus=1


# VIRYA
source /etc/profile.d/modules.sh; module load python/default; module load cuda/default; module list; cd /home/a_s87063/repos/perf-pilot; source venv/bin/activate


# 20 GB MIGS - Virya[3-6]
salloc --partition=migs --gpus=1 --mem=200G --time=01:00:00

# 32 GB GPU (8) - Virya[1-2]
salloc --partition=phys --gpus=2 --cpus-per-task=4 --mem=90G --time=02:00:00

-w virya2

# start ssh agent and add private key
eval "$(ssh-agent -s)"; ssh-add ~/.ssh/id_ed25519_github;



#Speed - prepare environment
cd /speed-scratch/$USER/vscode
setenv XDG_RUNTIME_DIR /speed-scratch/$USER/run-user

# load required modules
module load python/3.12.0/default
module load cuda/11.8/default

# set environment variables
source /nfs/home/a/a_s87063/set_env.csh

# run vscode
/speed-scratch/nag-public/code-server-4.22.1/bin/code-server --user-data-dir=$PWD\/projects --config=$PWD\/home/.config/code-server/config.yaml --bind-addr="0.0.0.0:8890" /speed-scratch/$USER/repos/perf-pilot

# create tunnel to slurm node on user machine
ssh -L 8890:speed-40:8890 speed

# check the password to open vscode in the browser
cat /speed-scratch/$USER/vscode/home/.config/code-server/config.yaml

# all the commands as a single one
cd /speed-scratch/$USER/vscode ; setenv XDG_RUNTIME_DIR /speed-scratch/$USER/run-user ; module load python/3.12.0/default ; module load cuda/11.8/default ; source /home/a_s87063/repos/perf-pilot/slurm_scripts/speed/set_env.csh; /speed-scratch/nag-public/code-server-4.22.1/bin/code-server --user-data-dir=$PWD\/projects --config=$PWD\/home/.config/code-server/config.yaml --bind-addr="0.0.0.0:8890" /speed-scratch/$USER/repos/perf-pilot


# submit a batch job
sbatch /speed-scratch/a_s87063/repos/perf-pilot/slurm_scripts/speed/seq_class_llama.sh

sbatch /speed-scratch/a_s87063/repos/perf-pilot/slurm_scripts/speed/extract_data.sh

sbatch /speed-scratch/$USER/repos/perf-pilot/slurm_scripts/speed/log_reg.sh

sbatch /speed-scratch/a_s87063/repos/perf-pilot/slurm_scripts/speed/seq_class_codebert.sh

sbatch /home/a_s87063/repos/perf-pilot/slurm_scripts/virya/download_transform_model.sh

# submit a batch job for multiple GPUs using multiple nodes
sbatch /speed-scratch/$USER/repos/perf-pilot/slurm_scripts/speed/dist_llm_fine_tune.sh


squeue -u $USER

scontrol show node virya1

sacct -j 540762 --format=JobID,State,ExitCode,Elapsed
scontrol show job 540762

# Check logs
/speed-scratch/$USER/repos/perf-pilot/slurm_jobs

# Send a file via rsync through windows wsl
rsync -avzP -e ssh a_s87063@speed.encs.concordia.ca:/nfs/speed-scratch/a_s87063/repos/perf-pilot/venv "/mnt/c/Users/alals/Desktop/repos sync"

# add python requirements without version
pip freeze | awk -F '=' '{print $1}' > requirements.txt


# virya
sbatch /home/a_s87063/repos/perf-pilot/slurm_scripts/virya/causal_lm_llama.sh