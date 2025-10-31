#!/encs/bin/tcsh

#SBATCH --job-name=command
#SBATCH --output=/speed-scratch/a_s87063/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=168:00:00
#SBATCH --account=pcr
#SBATCH --constraint=el9

REPO_ROOT="/speed-scratch/a_s87063/repos/jit-dp-llm"

echo "loading modules"
module load python/3.12.0/default
module list

echo "setting env variables"
source /speed-scratch/a_s87063/repos/jit-dp-llm/slurm_scripts/speed/set_env.csh

echo "activating venv"
source /speed-scratch/a_s87063/repos/jit-dp-llm/venv/bin/activate.csh

echo "Running script"

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/eval_thresholds.py \
# --json /speed-scratch/a_s87063/repos/jit-dp-llm/llama/inference/eval_2025-10-09_19-10-15/final_test_results_seq_cls.json \
# --out /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/results/eval_results_3.json \
# --recall-target 0.84 \
# --threshold 0.4 \
# -p 0.10 0.20 \


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/token_length_violin.py \
# --dataset /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf/perf_llm_struc.jsonl \
# --model /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/snapshots/meta-llama/Llama-3.1-8B \
# --configs /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/seq_len_confis.jsonl \
# --field prompt \
# --logy \
# --output /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/results/violin_plot


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/simulate_perf_reg_predictor.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/simulation.py

python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/simulate_predictor.py

echo "Script finished"