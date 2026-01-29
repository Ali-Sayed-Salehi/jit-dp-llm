# `mozilla_perf` dataset

Mozilla performance regression datasets extracted from:
- Treeherder performance alerts (which point to Bugzilla bugs + a regressor revision)
- Bugzilla bug metadata
- Mercurial (`autoland`) code changes and commit metadata

This directory also contains Treeherder “coverage”/metadata artifacts (signatures, jobs, grouping)
used by the perf-job batch testing simulations.

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

**Used by:** `data_extraction/treeherder/get_sigs_per_job.py`, `data_extraction/treeherder/rectify_job_count_per_revision.py`

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

**Used by:** `data_extraction/treeherder/get_job_duration.py`, `data_extraction/treeherder/rectify_job_count_per_revision.py`

**Per-line object fields (2):** `Sig_group_id` (int), `signatures` (list[int])

### `sig_group_job_durations.csv` (CSV)

**What it is:** Mean duration (minutes) per signature-group.

**Produced by:** `data_extraction/treeherder/get_job_duration.py`

**Columns (2):** `signature_group_id` (int), `duration_minutes` (float)

### `perf_jobs_per_revision_details_rectified.jsonl` (JSONL)

**What it is:** Per-revision signature coverage, but “rectified” by signature-grouping and filtered
to exclude specific signature frameworks.

**Produced by:** `data_extraction/treeherder/rectify_job_count_per_revision.py`
(`perf_jobs_per_revision_details.jsonl` is the unrectified input produced by `get_num_perf_tests.py`)

**Per-line object fields (4):**
- `revision` (str)
- `submit_time_iso` (str / ISO timestamp)
- `total_jobs` (int): `len(unique(signature_group_ids))` after exclusions
- `signature_group_ids` (list[int])

## Typical regeneration order

1. `python data_extraction/treeherder/get_perf_alerts.py`
2. `python data_extraction/bugzilla/get_perf_bugs.py`
3. `python data_extraction/mercurial/fetch_all_commit.py`
4. `python data_extraction/mercurial/get_bug_diffs.py`
5. `python data_extraction/data_preparation.py --mode mozilla_perf_struc`
6. `python data_extraction/treeherder/get_num_perf_tests.py`
7. `python data_extraction/treeherder/get_perf_sigs.py`
8. `python data_extraction/treeherder/get_sigs_per_job.py`
9. `python data_extraction/treeherder/create_sig_groups.py`
10. `python data_extraction/treeherder/get_job_duration.py`
11. `python data_extraction/treeherder/rectify_job_count_per_revision.py`

