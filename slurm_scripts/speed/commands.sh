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

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/jit_dp/eval_thresholds.py \
# --json /speed-scratch/a_s87063/repos/jit-dp-llm/llama/inference/eval_2025-10-31_22-57-47/final_test_results_seq_cls.json \
# --out /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/results/jit_dp/eval_thr_results_perf_mbert.json \
# --threshold 0.7 \
# --recall-target 0.84 \
# -p 0.10 0.20 \


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/jit_dp/token_length_violin.py \
# --dataset /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf/perf_llm_struc.jsonl \
# --model /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/snapshots/meta-llama/Llama-3.1-8B \
# --configs /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/jit_dp/seq_len_confis.jsonl \
# --field prompt \
# --logy \
# --output /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/results/jit_dp/violin_plot


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/simulate_perf_reg_predictor.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/simulation.py \
# --input-json-eval /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/final_test_results_perf_codebert_eval.json \
# --input-json-final /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/final_test_results_perf_codebert_final_test.json \
# --output-eval /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/results/50t_opt_max_ttc/batch_eval_mopt.json \
# --output-final /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/results/50t_opt_max_ttc/batch_test_mopt.json \
# --build-time-minutes 98.7 \
# --mopt-trials 50 \
# --skip-exhaustive-testing \
# --batching TWSB,TWB,TWB-s,FSB,FSB-s,RASB,RASB-s,RASB-la,RASB-la-s,RAPB,RAPB-s,RAPB-la,RAPB-la-s,RATB,RATB-s,LAB,LAB-s,LARAB,LARAB-s,HATS,RAHATS,RAHATS-la,ARAHATS,ARAHATS-la \
# --bisection PAR,TOB,RWAB,RWAB-LS,TKRB,SWB,SWF \
# --optimize-for-timeliness-metric max_ttc \
# --baseline-opt-metric-multplier 2 \
# --workers-android 60 \
# --workers-windows 120 \
# --workers-linux 100 \
# --workers-mac 250


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

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_num_perf_tests.py --debug

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_job_wait_times.py --dry-run

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_ttc_stats.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/bugzilla/get_all_bugs.py --start-offset 83250

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/mercurial/link_bug_diffs.py --cutoff-date 2025-01-01

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/git_bisect/simulate.py \
# --bugs-path /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_jit/mozilla_jit_2022.jsonl \
# --commits-path /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_jit/all_commits.jsonl \
# --risk-eval /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/git_bisect/risk_predictions_eval.json \
# --risk-final /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/git_bisect/risk_predictions_final_test.json \
# --output-eval /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/git_bisect/results/50t/simulation_optuna_eval.json \
# --output-final /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/git_bisect/results/50t/simulation_optuna_final_test.json \
# --pareto-front-stats-path /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/git_bisect/results/50t/pareto_front_stats.json \
# --pareto-front-plot-path /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/git_bisect/results/50t/pareto_front_distributions.png \
# --penalize-window-start-lookback \
# --window-start-lookback-penalty-tests 4 \
# --mopt-trials 50 \
# --optuna-seed 42 \
# --lookback NBLB,NLB,FSLB,FSLB-AD,FSLB-AI,RATLB,RATLB-AD,RATLB-AI,RWLBS,RWLBS-AD,RWLBS-AI,RWLBLS,RWLBLS-AD,RWLBLS-AI,TWLB,TWLB-AD,TWLB-AI,FSLB-FF,FSLB-AD-FF,FSLB-AI-FF,RATLB-FF,RATLB-AD-FF,RATLB-AI-FF,RWLBS-FF,RWLBS-AD-FF,RWLBS-AI-FF,RWLBLS-FF,RWLBLS-AD-FF,RWLBLS-AI-FF,TWLB-FF,TWLB-AD-FF,TWLB-AI-FF \
# --bisection GB,TKRB,RWBS,RWBLS \
# --multi-objective-opt \
# --log-level INFO \
# --final-only
# # --dry-run


# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_sigs_per_job.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_perf_sigs.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/create_sig_groups.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/rectify_job_count_per_revision.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/filter_perf_llm_struc.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/find_historical_risk_scores.py \
# --eval-preds-json /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/final_test_results_perf_codebert_eval.json


echo "Script finished"
