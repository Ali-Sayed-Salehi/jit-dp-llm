# run interactive job with GPU
# THIS SHOULD BE RUN FIRST AND SEPARATE FROM OTHER COMMANDS
salloc --nodelist=speed-17 --partition=pg --constraint=el9 --mem=10G --gpus=1 -A pcr

#prepare environment
cd /speed-scratch/$USER/vscode
setenv XDG_RUNTIME_DIR /speed-scratch/$USER/run-user

# load required modules
module load python/3.12.0/default
module load cuda/11.8/default

# run vscode
/speed-scratch/nag-public/code-server-4.22.1/bin/code-server --user-data-dir=$PWD\/projects --config=$PWD\/home/.config/code-server/config.yaml --bind-addr="0.0.0.0:8890" /speed-scratch/$USER/repos/perf-pilot

# create tunnel to slurm node on user machine
ssh -L 8890:speed-17:8890 slurm

# check the password to open vscode in the browser
cat /speed-scratch/$USER/vscode/home/.config/code-server/config.yaml

# all the commands as a single one
cd /speed-scratch/$USER/vscode ; setenv XDG_RUNTIME_DIR /speed-scratch/$USER/run-user ; module load python/3.12.0/default ; module load cuda/11.8/default ; /speed-scratch/nag-public/code-server-4.22.1/bin/code-server --user-data-dir=$PWD\/projects --config=$PWD\/home/.config/code-server/config.yaml --bind-addr="0.0.0.0:8890" /speed-scratch/$USER/repos/perf-pilot
