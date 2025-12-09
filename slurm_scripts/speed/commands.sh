#!/encs/bin/tcsh

#SBATCH --job-name=command
#SBATCH --output=/speed-scratch/a_s87063/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
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

echo "Running script ..."

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/eval_thresholds.py \
# --json /speed-scratch/a_s87063/repos/jit-dp-llm/llama/inference/eval_2025-10-31_22-57-47/final_test_results_seq_cls.json \
# --out /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/results/eval_thr_results_perf_mbert.json \
# --threshold 0.7 \
# --recall-target 0.84 \
# -p 0.10 0.20 \


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/token_length_violin.py \
# --dataset /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf/perf_llm_struc.jsonl \
# --model /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/snapshots/meta-llama/Llama-3.1-8B \
# --configs /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/seq_len_confis.jsonl \
# --field prompt \
# --logy \
# --output /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/results/violin_plot


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/simulate_perf_reg_predictor.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/simulation.py \
# --input-json-eval /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/final_test_results_perf_mbert_eval.json \
# --input-json-final /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/final_test_results_perf_mbert_final_test.json \
# --output-eval /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/k_test/batch_eval_mopt.json \
# --output-final /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/k_test/batch_test_mopt.json \
# --mopt-trials 2 \
# --num-test-workers 100 \
# --full-suite-sigs-per-run 850 \
# --log-level INFO
# --final-only

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/simulate_predictor.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/plot.py \
# --json_path /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/batch_eval_mopt.json \
# --out /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/scatter_mft.png \
# --xaxis mft

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/conduit/get_all_drevs.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/conduit/get_commit_drevs.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_job_duration.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_num_perf_tests.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_failing_perf_sigs.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/bugzilla/get_perf_bugs.py

python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_num_perf_tests.py --debug

echo "Script finished"