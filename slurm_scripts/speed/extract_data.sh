#!/encs/bin/tcsh

#SBATCH --job-name=extract-data
#SBATCH --output=/speed-scratch/a_s87063/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=168:00:00
#SBATCH --account=pcr
#SBATCH --constraint=el9

echo "loading modules"
module load python/3.12.0/default
module list

echo "setting env variables"
source /speed-scratch/a_s87063/repos/jit-dp-llm/slurm_scripts/speed/set_env.csh

echo "activating venv"
source /speed-scratch/a_s87063/repos/jit-dp-llm/venv/bin/activate.csh

echo "running the data extraction script . . ."
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/github_api/get_jit_dataset.py --struc --ast
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/github_api/get_jit_dataset.py --struc --dataset_name jit_defects4j
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/data_preparation.py --mode mozilla_perf_struc --debug
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/github_api/javalang_structured_diff.py apache flink 999baceff36165d950a61dd9cc4342f252e64837 --ast
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/github_api/javalang_structured_diff.py apache hbase 2c799fb70aa47f9109714dc410862af24b4a0321 --ast
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/data_preparation.py --mode jit_llm_struc --dataset_name apachejit --dataset_size total --include_metadata
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/data_preparation.py --mode jit_llm_struc --dataset_name apachejit --dataset_size total --clm_mode --include_metadata
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/conduit/get_bug_diffs.py
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_perf_alerts.py
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/bugzilla/get_perf_bugs.py
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/mercurial/get_bug_diffs.py
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/mercurial/fetch_all_commit.py
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/data_preparation.py --mode mozilla_jit_struc

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

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_sigs_per_job.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_perf_sigs.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/create_sig_groups.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/rectify_job_count_per_revision.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/filter_perf_llm_struc.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/find_historical_risk_scores.py \
# --eval-preds-json /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/batch_testing/final_test_results_perf_codebert_eval.json

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/filter_fail_perf_sigs.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_perf_test_data_per_sig.py --debug --overwrite

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_perf_data_info.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/get_perf_test_data_per_revision.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/create_perf_bisect_dataset.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/conduit/get_commit_risk_scores.py

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/conduit/get_all_drevs.py \
# --input-jsonl /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/all_commits.jsonl \
# --eval-predictions-json /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/risk_predictions_eval.json \
# --final-test-predictions-json /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/risk_predictions_final_test.json \
# --output-jsonl /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/all_drevs.jsonl \
# --page-limit 100 \
# --rate-limit-min-interval 0.5
# Default repositories: autoland and mozilla-central.
# Debug-only: add --max-pages 2 for a smoke test. It intentionally truncates output.

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/conduit/get_per_commit_drevs.py \
# --all-drevs-jsonl /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/all_drevs.jsonl \
# --eval-predictions-json /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/risk_predictions_eval.json \
# --final-test-predictions-json /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/risk_predictions_final_test.json \
# --debug

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/conduit/get_drevs_transactions.py \
# --debug \
# --page-limit 100 \
# --rate-limit-min-interval 0.5

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/conduit/get_file_path_timelines.py \
# --skip-repo-update \
# --output-file-path-timeline-jsonl /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/file_path_timeline.jsonl \
# --disable-similarity-inference \
# --similarity-threshold 0.90 \
# --similarity-margin 0.05 \
# --similarity-max-file-bytes 1000000 \
# --similarity-max-pairs 2500 \
# --debug

# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/conduit/create_code_review_dataset.py \
# --skip-repo-update \
# --file-path-timeline-jsonl /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/file_path_timeline.jsonl \
# --output-dataset-jsonl /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/drev_review_data.jsonl \
# --write-select-commit-diffs \
# --select-commit-diffs-jsonl /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_code_review/select_commit_diffs.jsonl \
# --debug-count 20 \
# --debug


# Build expanded perf-bisect v2 data without risk-score split boundaries.
# By default this exports all_commits.jsonl from the existing local Autoland
# checkout but does not run `hg pull -u`; add --pull-commits if you want that.
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/create_perf_bisect_dataset_v2.py \
# --source-dir /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect \
# --output-dir /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect_v2 \
# --autoland-repo /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/mercurial/repos/autoland \
# --eval-fraction 0.30 \
# --job-duration-samples 3 \
# --exclude-framework-ids 2,6,18 \
# --log-level INFO


# Prepare the reduced v2 dataset once before running the simulation below:
# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/reduce_dataset_sizes.py \
# --source-dir /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect_v2 \
# --output-dir /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect_v2/reduced \
# --overwrite


# Recompute oracle accuracies from the reduced summary-only revision data:
# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/perf_bisect/calculate_oracle_metrics.py \
# --skip-plot


# Count summary perf jobs per regression, filtered to oracle-metric rows:
# python /speed-scratch/a_s87063/repos/jit-dp-llm/data_extraction/treeherder/count_per_regression_test_jobs.py \
# --oracle-metrics /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/mozilla_perf_bisect_v2/reduced/per_regression_oracle_metrics_v2.jsonl \
# --one-signature-per-alert \
# --log-level INFO


echo "extraction finished"
