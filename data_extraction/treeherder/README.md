# Treeherder extraction (`data_extraction/treeherder`)

This folder contains scripts that pull performance data from Mozilla Treeherder and write
dataset artifacts under `datasets/mozilla_perf/`.

For file-level schemas and how these artifacts relate to the rest of the project, see:
`datasets/mozilla_perf/README.md`.

## Prerequisites

- Python environment with `requirements.txt` installed (notably `treeherder-client`, `requests`,
  `pandas`, `matplotlib`).
- Network access for scripts that hit the Treeherder API.

Most scripts infer the repo root from `__file__` and default their inputs/outputs to
`datasets/mozilla_perf/`. Many outputs are written in an append-only way so interrupted runs can
resume; if you want a clean regeneration, delete the relevant output files first.

## Scripts

### `get_perf_alerts.py`

Fetch Treeherder performance alert summaries and derive a compact “join table” for Bugzilla.

- **Inputs**
  - Treeherder API: `performance/alertsummary` (paged).
  - Script constants: `TIMESPAN_IN_YEARS` (default 3 years).
- **Outputs** (written to `datasets/mozilla_perf/`)
  - `alert_summaries.csv`: raw `performance/alertsummary` results over the time window.
  - `alerts_with_bug_and_test_info.csv`: filtered rows (only regression statuses) with bug id,
    regressed tests/platforms, alert summary id, and regressor revision.
- **Notes**
  - The script appends to `alert_summaries.csv` if it already exists; delete the file to avoid
    duplication when re-running.

### `get_failing_perf_sigs.py`

Extract the *regressing* performance signature IDs per alert summary.

- **Inputs** (from `datasets/mozilla_perf/`)
  - `alerts_with_bug_and_test_info.csv`: provides the set of `perf_reg_alert_summary_id` values.
  - `alert_summaries.csv`: provides the `alerts` payload for each summary id.
- **Outputs** (to `datasets/mozilla_perf/`)
  - `alert_summary_fail_perf_sigs.csv`: one row per alert summary id with:
    - `revision` (Treeherder “regressor” revision field from the alert summary)
    - `fail_perf_sig_ids` (JSON-encoded list[int] of `series_signature.id` where `is_regression`)
    - `num_fail_perf_sig_ids` (count)

### `get_ttc_stats.py`

Compute “time to culprit” (TTC) stats from alert summaries:

`TTC = created_time - push_timestamp`

- **Inputs** (from `datasets/mozilla_perf/`)
  - `alert_summaries.csv` (columns used: `created`, `push_timestamp`)
- **Outputs** (to `datasets/mozilla_perf/`)
  - `time_to_culprit_stats.json`: summary statistics in seconds and hours.

### `get_num_perf_tests.py`

Compute a per-revision perf “coverage” signal on `autoland`: for each revision in a configured UTC
date window, count the number of **unique perf signature IDs** that have at least one job
associated with that revision.

- **Inputs**
  - `datasets/mozilla_perf/all_commits.jsonl`: provides the canonical set of autoland revisions and
    their commit timestamps (used to filter jobs to the target date window).
  - Treeherder API:
    - `performance/signatures`: enumerates signature IDs.
    - `performance/summary`: job summaries per signature over a lookback interval.
  - Script constants: `REVISION_START_DATE`, `REVISION_END_DATE`.
- **Outputs** (to `datasets/mozilla_perf/`)
  - `perf_jobs_by_signature.jsonl`: append-only cache of per-signature job summaries (one line per
    signature) to make runs resumable.
  - `perf_jobs_per_revision.csv`: per-revision counts.
  - `perf_jobs_per_revision_details.jsonl`: same rows as the CSV, but includes `signature_ids`.
  - `perf_jobs_stats.json`: summary statistics over per-revision counts.
  - `perf_jobs_per_revision_dist.png`: histogram of per-revision counts.
- **Notes**
  - `--debug` samples a small set of signatures to keep runtime short.

### `get_perf_sigs.py`

Fetch full Treeherder performance signature metadata and write one record per line.

- **Inputs**
  - Treeherder API: `performance/signatures` (enumerate IDs + fetch details per ID).
- **Outputs** (to `datasets/mozilla_perf/`)
  - `all_signatures.jsonl`: append-only signature metadata cache.
- **Notes**
  - Supports `--repository`, `--debug N`, and resumable appending.

### `get_sigs_per_job.py`

Invert per-signature job caches into per-job signature lists, then add per-job “shared signature
metadata fields” summaries.

- **Inputs** (from `datasets/mozilla_perf/`)
  - `perf_jobs_by_signature.jsonl` (optional): used to build `sigs_by_job_id.jsonl` if missing.
  - `sigs_by_job_id.jsonl`: per-job signature list (created if missing).
  - `all_signatures.jsonl`: signature metadata used to compute shared fields.
- **Outputs** (to `datasets/mozilla_perf/`)
  - `sigs_by_job_id.jsonl`: `{job_id, signature_ids}` (one line per job).
  - `sigs_by_job_id_detailed.jsonl`: adds `common_sig_features` per job.

### `create_sig_groups.py`

Create “signature groups” from per-job signature lists.

Two signatures are in the same group iff they appear in *exactly the same set of jobs* (i.e., they
always co-occur).

- **Inputs** (from `datasets/mozilla_perf/`)
  - `sigs_by_job_id_detailed.jsonl`
- **Outputs** (to `datasets/mozilla_perf/`)
  - `sig_groups.jsonl`: `{Sig_group_id, signatures}` (one line per group).

### `get_job_duration.py`

Estimate job durations (minutes) by sampling job instances per signature and fetching timing
details from Treeherder. Also aggregates mean durations at the signature-group level.

- **Inputs** (from `datasets/mozilla_perf/`)
  - `perf_jobs_by_signature.jsonl`: job summaries (provides `job_id` values to look up).
  - `sig_groups.jsonl`: maps signature ID → signature group id.
  - Treeherder API: job details via `get_jobs(..., id=<job_id>)` for timestamps.
- **Outputs** (to `datasets/mozilla_perf/`)
  - `job_durations.csv`: mean duration per signature (`signature_id`, `duration_minutes`).
  - `sig_group_job_durations.csv`: mean duration per signature-group (`signature_group_id`, minutes).
  - `sig_group_job_duration_stats.json`: summary stats over signature-group durations.
  - `job_durations.png`: histogram plot of signature-group durations.
- **Notes**
  - If `job_durations.csv` already exists, the script skips Treeherder fetch and reuses it.

### `get_job_wait_times.py`

Compute queue wait times (submit → start) for the *busiest* revision in the configured window.

- **Inputs** (from `datasets/mozilla_perf/`)
  - `perf_jobs_per_revision.csv`: used to choose the revision with maximum `total_jobs`.
  - `perf_jobs_by_signature.jsonl`: used to enumerate job_ids for that revision.
  - Treeherder API: job details via `get_jobs(..., id=<job_id>)`.
- **Outputs** (to `datasets/mozilla_perf/`)
  - `max_revision_jobs.jsonl`: append-only cache of detailed job objects for the target revision.
  - `job_wait_time_stats.json`: summary stats for `start_timestamp - submit_timestamp`.
- **Notes**
  - `--dry-run` limits Treeherder fetches to 10 new jobs for quick experimentation.

### `rectify_job_count_per_revision.py`

Convert per-revision perf coverage from signature IDs to **signature-group IDs** to avoid
double-counting signatures that always co-occur. Optionally excludes entire signature-groups if
all signatures in the group belong to specific `framework_id` values.

- **Inputs** (from `datasets/mozilla_perf/`)
  - `perf_jobs_per_revision_details.jsonl`: per-revision signature ID lists from `get_num_perf_tests.py`.
  - `all_signatures.jsonl`: provides `framework_id` per signature.
  - `sig_groups.jsonl`: maps signature ID → signature group id.
- **Outputs** (to `datasets/mozilla_perf/`)
  - `perf_jobs_per_revision_details_rectified.jsonl`: replaces `signature_ids` with
    `signature_group_ids` and updates `total_jobs`.
- **Notes**
  - Default excluded frameworks: `{2, 6, 18}` (override with `--exclude-framework-ids`).

### `filter_perf_llm_struc.py` (post-processing)

Relabel `perf_llm_struc.jsonl` by excluding failing signatures in specific Treeherder frameworks.

- **Inputs**
  - `datasets/mozilla_perf/perf_llm_struc.jsonl` (LLM dataset produced by `data_extraction/data_preparation.py`)
  - `datasets/mozilla_perf/alert_summary_fail_perf_sigs.csv` (from `get_failing_perf_sigs.py`)
  - `datasets/mozilla_perf/alert_summaries.csv` (from `get_perf_alerts.py`; used to print alert summary `framework`)
  - `datasets/mozilla_perf/all_signatures.jsonl` (from `get_perf_sigs.py`; provides `framework_id`)
- **Outputs**
  - A rewritten JSONL with updated `response` labels (default:
    `datasets/mozilla_perf/perf_llm_struc_no_fw_2_6_18.jsonl`)
  - Optional JSON report (`--report-json`) with sanity-check counters.

## Typical Treeherder-only run order

This is a common order for (re)generating the Treeherder-derived artifacts under
`datasets/mozilla_perf/`:

1. `python data_extraction/treeherder/get_perf_alerts.py`
2. `python data_extraction/treeherder/get_ttc_stats.py` (optional)
3. `python data_extraction/treeherder/get_failing_perf_sigs.py` (optional)
4. `python data_extraction/treeherder/get_num_perf_tests.py`
5. `python data_extraction/treeherder/get_perf_sigs.py`
6. `python data_extraction/treeherder/get_sigs_per_job.py`
7. `python data_extraction/treeherder/create_sig_groups.py`
8. `python data_extraction/treeherder/get_job_duration.py` (optional; hits API for timestamps)
9. `python data_extraction/treeherder/get_job_wait_times.py` (optional; hits API for timestamps)
10. `python data_extraction/treeherder/rectify_job_count_per_revision.py`
