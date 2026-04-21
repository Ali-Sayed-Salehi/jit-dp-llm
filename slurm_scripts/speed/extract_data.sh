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

echo "extraction finished"
