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
# --batching "TWSB,\
# TWB,TWB-s,TWB-hats,\
# FSB,FSB-s,FSB-hats,\
# RASB,RASB-s,RASB-hats,\
# RASB-la,RASB-la-s,RASB-la-hats,\
# RAPB,RAPB-s,RAPB-hats,\
# RAPB-la,RAPB-la-s,RAPB-la-hats,\
# RATB,RATB-s,RATB-hats,\
# LAB,LAB-s,LAB-hats,\
# LARAB,LARAB-s,LARAB-hats,\
# LARAB-la,LARAB-la-s,LARAB-la-hats,\
# HATS,RAHATS,RAHATS-la,ARAHATS,ARAHATS-la" \
# --bisection PAR,TOB,RWAB,RWAB-LS,TKRB,SWB,SWF \
# --optimize-for-timeliness-metric max_ttc \
# --baseline-opt-metric-multplier 2 \
# --workers-android 60 \
# --workers-windows 120 \
# --workers-linux 100 \
# --workers-mac 250


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/model_machine_count.py \
# --input-json-eval /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/final_test_results_perf_codebert_eval.json \
# --input-json-final /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/final_test_results_perf_codebert_final_test.json \
# --out-json /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/results/plots/test_4/machine_count_sweep.json \
# --plots-dir /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/results/plots/test_4 \
# --build-time-minutes 98.7 \
# --mopt-trials 50 \
# --skip-exhaustive-testing \
# --compare-bisect-strats \
# --batching "TWSB,\
# TWB,TWB-s,TWB-hats,\
# FSB,FSB-s,FSB-hats,\
# RASB,RASB-s,RASB-hats,\
# RASB-la,RASB-la-s,RASB-la-hats,\
# RAPB,RAPB-s,RAPB-hats,\
# RAPB-la,RAPB-la-s,RAPB-la-hats,\
# RATB,RATB-s,RATB-hats,\
# LAB,LAB-s,LAB-hats,\
# LARAB,LARAB-s,LARAB-hats,\
# LARAB-la,LARAB-la-s,LARAB-la-hats,\
# HATS,RAHATS,RAHATS-la,ARAHATS,ARAHATS-la" \
# --bisection PAR,TOB,RWAB,RWAB-LS,TKRB \
# --optimize-for-timeliness-metric max_ttc \
# --baseline-opt-metric-multplier 2 \
# --base-workers-android 60 \
# --base-workers-windows 120 \
# --base-workers-linux 100 \
# --base-workers-mac 250 \
# --multiplier-list "0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,4" \
# --unknown-platform-pool mac \
# --log-level INFO


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/simulate_predictor.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/plot.py \
# --json_path /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/batch_eval_mopt.json \
# --out /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/scatter_mft.png \
# --xaxis mft

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

# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/calculate_oracle_metrics.py

# Flag ownership for this simulation command:
# - --risk-scores is used by ProbabilisticBisection_CumulativeRiskMedian_UniformPrior,
#   ProbabilisticBisection_PosteriorMedian_RiskAwarePrior, RiskWeightedBisection,
#   and RiskWeightedMultisection.
# - --midpoint-retrigger-count-* is tuned for StandardMidpointBisection and
#   RiskWeightedBisection.
# - --multisection-section-count-* is tuned for StandardMidpointMultisection,
#   RiskWeightedMultisection, and
#   ProbabilisticMultiSection_PosteriorQuantile_UniformPrior.
# - --multisection-retrigger-count-* is tuned for StandardMidpointMultisection
#   and RiskWeightedMultisection.
# - --backfill-retrigger-count-* is tuned for Backfill and BackfillWithRepeat.
# - --probe-repeat-count-* is tuned for BackfillWithRepeat.
# - --pba-confidence-threshold-*, --pba-repeat-count-*, and
#   --pba-max-test-runs-* are tuned for probabilistic bisection localizers and
#   ProbabilisticMultiSection_PosteriorQuantile_UniformPrior.
# - --pba-risk-prior-uniform-weight-* would tune only
#   ProbabilisticBisection_PosteriorMedian_RiskAwarePrior if provided.
# - --optuna-trials/--optuna-seed control tuning for every selected tunable
#   localizer/oracle combo; --random-seed controls noisy oracle draws.

python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/simulation.py \
--dataset all \
--regression-dir /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect \
--signature-info /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect/per_sig_perf_data_info.jsonl \
--revision-data /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect/per_revision_perf_data.jsonl \
--oracle-metrics /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/per_regression_oracle_metrics.jsonl \
--risk-scores /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect/per_commit_risk_scores.jsonl \
--output-dir /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/results \
--workers 1 \
--oracles SummaryComparison \
--localizers Backfill BackfillWithRepeat ProbabilisticBisection_CumulativeRiskMedian_UniformPrior ProbabilisticBisection_PosteriorMedian_RiskAwarePrior ProbabilisticBisection_PosteriorMedian_UniformPrior ProbabilisticMultiSection_PosteriorQuantile_UniformPrior RiskWeightedBisection RiskWeightedMultisection StandardMidpointBisection StandardMidpointMultisection \
--random-seed 42 \
--optuna-seed 42 \
--optuna-trials 50 \
--midpoint-retrigger-count-max 30 \
--multisection-section-count-min 3 \
--multisection-section-count-max 20 \
--multisection-retrigger-count-max 30 \
--backfill-retrigger-count-max 30 \
--probe-repeat-count-max 30 \
--pba-confidence-threshold-min 0.6 \
--pba-confidence-threshold-max 0.99 \
--pba-repeat-count-max 30 \
--pba-max-test-runs-min 20 \
--pba-max-test-runs-max 200


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/plot_for_machine_counts.py \
# --worker-counts 1 2 4 8 16 \
# --regression-dir /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect \
# --signature-info /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect/per_sig_perf_data_info.jsonl \
# --revision-data /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect/per_revision_perf_data.jsonl \
# --oracle-metrics /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/per_regression_oracle_metrics.jsonl \
# --risk-scores /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect/per_commit_risk_scores.jsonl \
# --output-dir /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/results \
# --sweep-output-json /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/results/machine_count_sweep_final_test.json \
# --plots-dir /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/results/plots \
# --oracles SummaryComparison \
# --localizers Backfill BackfillWithRepeat ProbabilisticBisection_CumulativeRiskMedian_UniformPrior ProbabilisticBisection_PosteriorMedian_RiskAwarePrior ProbabilisticBisection_PosteriorMedian_UniformPrior ProbabilisticMultiSection_PosteriorQuantile_UniformPrior RiskWeightedBisection RiskWeightedMultisection StandardMidpointBisection StandardMidpointMultisection \
# --random-seed 42 \
# --optuna-seed 42 \
# --optuna-trials 50 \
# --midpoint-retrigger-count-max 200 \
# --multisection-section-count-min 3 \
# --multisection-section-count-max 16 \
# --multisection-retrigger-count-max 200 \
# --backfill-retrigger-count-max 200 \
# --probe-repeat-count-max 200 \
# --pba-confidence-threshold-min 0.6 \
# --pba-confidence-threshold-max 0.99 \
# --pba-repeat-count-max 200 \
# --pba-max-test-runs-min 20 \
# --pba-max-test-runs-max 200



echo "Script finished"
