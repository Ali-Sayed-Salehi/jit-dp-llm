#!/encs/bin/tcsh

echo "loading modules"
module load python/3.12.0/default
module load cuda/12.8/default
module list

echo "setting env variables"
source /speed-scratch/a_s87063/repos/perf-pilot/slurm_scripts/speed/set_env.csh

echo "activating venv"
source /speed-scratch/$USER/repos/perf-pilot/venv/bin/activate.csh
# echo "Python interpreter: `which python`"
