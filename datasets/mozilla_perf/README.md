# `mozilla_perf` dataset

Mozilla performance regression datasets extracted from:
- Treeherder performance alerts (which point to Bugzilla bugs + a regressor revision)
- Bugzilla bug metadata
- Mercurial (`autoland`) code changes and commit metadata

This directory also contains Treeherder “coverage”/metadata artifacts (signatures, jobs, grouping)
used by the perf-job batch testing simulations.

For script-level documentation of the extraction pipeline, see:
- `data_extraction/treeherder/README.md`
- `data_extraction/bugzilla/README.md`
- `data_extraction/mercurial/README.md`

## Files and schema

### `alert_summaries.csv` (CSV)

**What it is:** Raw Treeherder `performance/alertsummary` results over a time window.

**Produced by:** `data_extraction/treeherder/get_perf_alerts.py`

**Used by:** `data_extraction/treeherder/get_failing_perf_sigs.py`, `data_extraction/treeherder/get_ttc_stats.py`

**Columns (25):**
`id`, `push_id`, `prev_push_id`, `original_revision`, `created`, `first_triaged`,
`triage_due_date`, `repository`, `framework`, `alerts`, `related_alerts`, `status`,
`bug_number`, `bug_due_date`, `bug_updated`, `issue_tracker`, `notes`, `revision`,
`push_timestamp`, `prev_push_revision`, `original_prev_push_revision`,
`assignee_username`, `assignee_email`, `performance_tags`, `duplicated_summaries_ids`.

**Notes:**
- `alerts` / `related_alerts` are stringified list/dict structures (the scripts parse them with
  `ast.literal_eval`).
- `push_timestamp` is the push time (epoch seconds) serialized into CSV.
- `get_perf_alerts.py` appends to `alert_summaries.csv` if it already exists; delete the file to
  avoid duplicated pages when regenerating.

### `alerts_with_bug_and_test_info.csv` (CSV)

**What it is:** A compact “join” table derived from alert summaries, used to drive Bugzilla pulls.

**Produced by:** `data_extraction/treeherder/get_perf_alerts.py`

**Used by:** `data_extraction/bugzilla/get_perf_bugs.py`, `data_extraction/treeherder/get_failing_perf_sigs.py`

**Columns (5):**
- `regression_bug_id` (int): Bugzilla bug number linked from the alert summary
- `reg_perf_tests_list` (stringified list[dict]): each element is `{"test": <str>, "platform": <str>}`
- `perf_reg_alert_summary_id` (int): Treeherder alert summary id
- `regressor_push_head_revision` (str): revision/hash Treeherder associates as the regressor push head
- `alert_creation_date` (str): timestamp from Treeherder (`created`)

### `alert_summary_fail_perf_sigs.csv` (CSV)

**What it is:** For each Treeherder alert summary ID, the list of *regressing* performance signature
IDs extracted from the `alerts` payload.

**Produced by:** `data_extraction/treeherder/get_failing_perf_sigs.py`

**Used by:** `analysis/batch_testing/` (maps failing signatures → signature-groups via `sig_groups.jsonl`)

**Columns (4):**
- `alert_summary_id` (int)
- `revision` (str): Treeherder `revision` field from the alert summary
- `fail_perf_sig_ids` (JSON string): list[int] of `series_signature.id` where `is_regression == True`
- `num_fail_perf_sig_ids` (int)

### `time_to_culprit_stats.json` (JSON)

**What it is:** Summary stats for Treeherder “time to culprit” (TTC), defined as:
`created_time - push_timestamp`.

**Produced by:** `data_extraction/treeherder/get_ttc_stats.py`

### `all_bugs.csv` (CSV)

**What it is:** Raw Bugzilla bugs pulled for a time window.

**Produced by:** `data_extraction/bugzilla/get_perf_bugs.py`

**Used by:** `data_extraction/bugzilla/get_perf_bugs.py` (intermediate for constructing `perf_bugs.csv`)

**Columns (6):**
- `id` (int)
- `summary` (str)
- `regressed_by` (stringified list[int])
- `product` (str)
- `component` (str)
- `creation_time` (str / ISO timestamp)

### `perf_bugs.csv` (CSV)

**What it is:** Filtered Bugzilla bugs limited to “relevant” perf areas, labeled as perf regressor
or perf regression and enriched with Treeherder info.

**Produced by:** `data_extraction/bugzilla/get_perf_bugs.py`

**Used by:** `data_extraction/mercurial/get_bug_diffs.py` (diff extraction)

**Columns (10):**
- `bug_id` (int)
- `bug_summary` (str)
- `bug_is_perf_regressor` (bool)
- `bug_is_perf_regression` (bool)
- `regressed_perf_tests` (stringified list[dict]): same shape as `reg_perf_tests_list`
- `regressor_revision` (str | empty): revision/hash for regressor bugs
- `perf_reg_alert_summary_id` (int)
- `product` (str)
- `component` (str)
- `creation_time` (str / ISO timestamp)

### `all_commits.jsonl` (JSONL)

**What it is:** Mercurial `autoland` commit metadata exported from `hg log -Tjson -r all()`.

**Produced by:** `data_extraction/mercurial/fetch_all_commit.py`

**Used by:** `data_extraction/treeherder/get_num_perf_tests.py`, `data_extraction/mercurial/get_bug_diffs.py`

**Per-line object fields (4):**
- `node` (str): full changeset hash
- `desc` (str): commit description/message
- `date` (list): Mercurial date field `[epoch_seconds, tz_offset_seconds]`
- `parents` (list[str]): parent changeset hashes (first parent is `parents[0]`)

### `perf_bugs_with_diff.jsonl` (JSONL)

**What it is:** Bug-level dataset that attaches a single *net diff* to each bug based on a contiguous
block of `Bug <id>` commits in `autoland`.

**Produced by:** `data_extraction/mercurial/get_bug_diffs.py`

**Used by:** `data_extraction/data_preparation.py` (mode `mozilla_perf_struc`)

**Per-line object fields (6):**
- `bug_id` (str)
- `revision` (str): newest commit hash in the contiguous block
- `last_commit_date` (str / ISO timestamp)
- `regressor` (bool): from `perf_bugs.csv.bug_is_perf_regressor`
- `commit_message` (str): concatenated/cleaned commit messages for the block
- `diff` (str): unified diff text for the net change (`hg diff -r <p1(oldest)> -r <newest>`)

### `perf_llm_struc.jsonl` (JSONL)

**What it is:** LLM-ready classification dataset derived from `perf_bugs_with_diff.jsonl`.

**Produced by:** `data_extraction/data_preparation.py` (mode `mozilla_perf_struc`)

**Per-line object fields (3):**
- `commit_id` (str): same value as `revision`
- `prompt` (str): `<COMMIT_MESSAGE>…</COMMIT_MESSAGE>` + structured diff markup
- `response` (str): `"1"` if `regressor` else `"0"`

**Structured diff markup:** generated by `data_extraction/utils.py:diff_to_structured_xml()`.
It emits repeated blocks like:
- `<FILE>` then a file path line
- nested `<ADDED>` / `<REMOVED>` blocks with the changed lines

### `all_signatures.jsonl` (JSONL)

**What it is:** Treeherder performance signature metadata (one signature record per line).

**Produced by:** `data_extraction/treeherder/get_perf_sigs.py`

**Used by:** `data_extraction/treeherder/get_sigs_per_job.py`, `data_extraction/treeherder/rectify_job_count_per_revision.py`,
`analysis/batch_testing/` (maps signature metadata → simulation pools)

**Per-line object fields:** Treeherder response objects. Common keys include:
`id`, `signature_hash`, `framework_id`, `suite`, `test`, `machine_platform`,
`extra_options`, `measurement_unit`, `lower_is_better`, `has_subtests`, `tags`, …

### `perf_jobs_by_signature.jsonl` (JSONL)

**What it is:** Treeherder job summaries cached per signature (one line per signature).

**Produced by:** `data_extraction/treeherder/get_num_perf_tests.py`

**Used by:** `data_extraction/treeherder/get_sigs_per_job.py`, `data_extraction/treeherder/get_job_duration.py`,
`data_extraction/treeherder/get_job_wait_times.py`

**Per-line object fields (2):**
- `signature_id` (int)
- `jobs` (list[dict]): Treeherder “summary” job dicts. Observed keys include:
  `id`, `job_id` (may be null), `push_id`, `push_timestamp`, `revision`, `submit_time` (may be null), `value`

### `perf_jobs_per_revision.csv` (CSV)

**What it is:** Count of perf signature IDs observed per revision in a date window.

**Produced by:** `data_extraction/treeherder/get_num_perf_tests.py`

**Used by:** `data_extraction/treeherder/get_job_wait_times.py`

**Columns (3):** `revision` (str), `submit_time_iso` (str / ISO timestamp), `total_jobs` (int)

### `perf_jobs_per_revision_details.jsonl` (JSONL)

**What it is:** Per-revision perf signature coverage with the explicit `signature_ids` list attached.

**Produced by:** `data_extraction/treeherder/get_num_perf_tests.py`

**Used by:** `data_extraction/treeherder/rectify_job_count_per_revision.py`

**Per-line object fields (4):**
- `revision` (str)
- `submit_time_iso` (str / ISO timestamp)
- `total_jobs` (int): number of unique `signature_ids` for that revision (after timing filters)
- `signature_ids` (list[int])

### `perf_jobs_stats.json` (JSON)

**What it is:** Summary statistics over `perf_jobs_per_revision.csv.total_jobs` (mean/median,
percentiles, trimmed mean, geometric mean).

**Produced by:** `data_extraction/treeherder/get_num_perf_tests.py`

### `perf_jobs_per_revision_dist.png` (PNG)

**What it is:** Histogram of `perf_jobs_per_revision.csv.total_jobs`.

**Produced by:** `data_extraction/treeherder/get_num_perf_tests.py`

### `sigs_by_job_id.jsonl` (JSONL)

**What it is:** Inverted index from Treeherder job id → list of perf signature ids.

**Produced by:** `data_extraction/treeherder/get_sigs_per_job.py`

**Used by:** `data_extraction/treeherder/get_sigs_per_job.py` (as input when building the detailed file)

**Per-line object fields (2):** `job_id` (int), `signature_ids` (list[int])

### `sigs_by_job_id_detailed.jsonl` (JSONL)

**What it is:** Adds a per-job “common signature metadata fields” summary.

**Produced by:** `data_extraction/treeherder/get_sigs_per_job.py`

**Used by:** `data_extraction/treeherder/create_sig_groups.py`

**Per-line object fields (3):**
- `job_id` (int)
- `signature_ids` (list[int])
- `common_sig_features` (list[str]): signature metadata keys that are identical across all signatures in that job

### `sig_groups.jsonl` (JSONL)

**What it is:** Signature co-occurrence groups. Two signatures are grouped iff they appear in exactly
the same set of jobs.

**Produced by:** `data_extraction/treeherder/create_sig_groups.py`

**Used by:** `data_extraction/treeherder/get_job_duration.py`, `data_extraction/treeherder/rectify_job_count_per_revision.py`,
`analysis/batch_testing/` (maps signatures → signature-groups)

**Per-line object fields (2):** `Sig_group_id` (int), `signatures` (list[int])

### `job_durations.csv` (CSV)

**What it is:** Intermediate per-signature mean job duration estimates (minutes). This file is used
to build `sig_group_job_durations.csv`.

**Produced by:** `data_extraction/treeherder/get_job_duration.py`

**Columns (2):** `signature_id` (int), `duration_minutes` (float)

### `sig_group_job_durations.csv` (CSV)

**What it is:** Mean duration (minutes) per signature-group.

**Produced by:** `data_extraction/treeherder/get_job_duration.py`

**Used by:** `analysis/batch_testing/` (simulation + bisection strategy timing)

**Columns (2):** `signature_group_id` (int), `duration_minutes` (float)

### `sig_group_job_duration_stats.json` (JSON)

**What it is:** Summary statistics over `sig_group_job_durations.csv.duration_minutes`.

**Produced by:** `data_extraction/treeherder/get_job_duration.py`

### `job_durations.png` (PNG)

**What it is:** Histogram plot of `sig_group_job_durations.csv.duration_minutes`.

**Produced by:** `data_extraction/treeherder/get_job_duration.py`

### `max_revision_jobs.jsonl` (JSONL)

**What it is:** Append-only cache of detailed Treeherder job objects for the single revision with
the highest `perf_jobs_per_revision.csv.total_jobs`.

**Produced by:** `data_extraction/treeherder/get_job_wait_times.py`

**Per-line object fields:** Treeherder job objects (the script uses `id`/`job_id`,
`submit_timestamp`, `start_timestamp`, and `revision`).

### `job_wait_time_stats.json` (JSON)

**What it is:** Summary stats for job “wait time” (seconds/minutes), defined as:
`start_timestamp - submit_timestamp`, computed from `max_revision_jobs.jsonl`.

**Produced by:** `data_extraction/treeherder/get_job_wait_times.py`

### `perf_jobs_per_revision_details_rectified.jsonl` (JSONL)

**What it is:** Per-revision signature coverage, but “rectified” by signature-grouping and filtered
to exclude specific signature frameworks.

**Produced by:** `data_extraction/treeherder/rectify_job_count_per_revision.py`
(`perf_jobs_per_revision_details.jsonl` is the unrectified input produced by `get_num_perf_tests.py`)

**Used by:** `analysis/batch_testing/` (derives the set of signature-groups Mozilla actually ran per revision)

**Per-line object fields (4):**
- `revision` (str)
- `submit_time_iso` (str / ISO timestamp)
- `total_jobs` (int): `len(unique(signature_group_ids))` after exclusions
- `signature_group_ids` (list[int])

## Typical regeneration order

1. `python data_extraction/treeherder/get_perf_alerts.py`
2. `python data_extraction/treeherder/get_ttc_stats.py` (optional)
3. `python data_extraction/treeherder/get_failing_perf_sigs.py` (optional)
4. `python data_extraction/bugzilla/get_perf_bugs.py`
5. `python data_extraction/mercurial/fetch_all_commit.py`
6. `python data_extraction/mercurial/get_bug_diffs.py`
7. `python data_extraction/data_preparation.py --mode mozilla_perf_struc`
8. `python data_extraction/treeherder/get_num_perf_tests.py`
9. `python data_extraction/treeherder/get_perf_sigs.py`
10. `python data_extraction/treeherder/get_sigs_per_job.py`
11. `python data_extraction/treeherder/create_sig_groups.py`
12. `python data_extraction/treeherder/get_job_duration.py` (optional; hits Treeherder for timestamps)
13. `python data_extraction/treeherder/get_job_wait_times.py` (optional; hits Treeherder for timestamps)
14. `python data_extraction/treeherder/rectify_job_count_per_revision.py`
