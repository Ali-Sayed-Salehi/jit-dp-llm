#!/encs/bin/tcsh

#SBATCH --job-name=extract-data
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=168:00:00
#SBATCH --account=pcr
#SBATCH --constraint=el9

echo "loading modules"
module load python/3.12.0/default
module list

echo "activating venv"
source /speed-scratch/$USER/repos/perf-pilot/venv/bin/activate.csh

echo "running the data extraction script . . ."
# python /speed-scratch/a_s87063/repos/perf-pilot/data_extraction/github_api/get_jit_dataset.py --struc --ast
# python /speed-scratch/a_s87063/repos/perf-pilot/data_extraction/github_api/get_jit_dataset.py --struc --dataset_name apachejit
# python /speed-scratch/a_s87063/repos/perf-pilot/data_extraction/data_preparation.py --mode mozilla_perf_struc --debug
# python /speed-scratch/a_s87063/repos/perf-pilot/data_extraction/github_api/javalang_structured_diff.py apache flink 999baceff36165d950a61dd9cc4342f252e64837 --ast
# python /speed-scratch/a_s87063/repos/perf-pilot/data_extraction/github_api/javalang_structured_diff.py apache hbase 2c799fb70aa47f9109714dc410862af24b4a0321 --ast
python /speed-scratch/a_s87063/repos/perf-pilot/data_extraction/data_preparation.py --mode jit_llm_struc --dataset_name jit_defects4j --dataset_size total
# python /speed-scratch/a_s87063/repos/perf-pilot/data_extraction/conduit/get_bug_diffs.py
echo "extraction finished"
