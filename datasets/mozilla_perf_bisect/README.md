# Mozilla Performance Bisect Dataset

This directory contains the data used by `analysis/perf_bisect/` to replay
historical Mozilla performance regressions and simulate culprit localization
between a known-good and known-bad revision.

Most files here are generated artifacts. Several upstream extraction scripts
still default to `datasets/mozilla_perf/`; for this bisect dataset, run those
scripts there first and copy or symlink the needed outputs into
`datasets/mozilla_perf_bisect/`, or pass explicit input/output paths where the
script supports them.

## Main Consumers

- `analysis/perf_bisect/simulation.py`
  - reads `perf_bisect_regressions_eval.jsonl`
  - reads `perf_bisect_regressions_final_test.jsonl`
  - reads `per_revision_perf_data.jsonl`
  - reads `per_sig_perf_data_info.jsonl`
- `data_extraction/treeherder/create_perf_bisect_dataset.py`
  - creates the eval/final-test regression JSONL files from the upstream
    alert, commit, signature, and model-prediction artifacts.

## Files

### `alert_summaries.csv`

Raw Treeherder `performance/alertsummary` rows. Important columns for this
dataset are:

- `id`: Treeherder alert summary id.
- `alerts`: Python-literal payload containing individual alert records.
- `original_prev_push_revision`: known-good revision before the regressing push.
- `original_revision`: known-bad revision for the regressing push.
- `revision`: Treeherder's culprit/regressor revision.
- `framework`, `created`: used by the filter step.

Produced by:

```bash
python data_extraction/treeherder/get_perf_alerts.py
```

That script writes to `datasets/mozilla_perf/alert_summaries.csv` by default.

### `alert_summary_fail_perf_sigs.csv`

One row per alert summary with the regressing performance signature IDs from
the alert payload.

Columns:

- `alert_summary_id`
- `revision`
- `fail_perf_sig_ids`: JSON-encoded list of failing signature ids.
- `num_fail_perf_sig_ids`

Produced by:

```bash
python data_extraction/treeherder/get_failing_perf_sigs.py
```

That script writes to `datasets/mozilla_perf/alert_summary_fail_perf_sigs.csv`
by default.

### `alert_summary_fail_perf_sigs_no_fw_2_6_18.csv`

Filtered allowlist of alert summaries used when building the bisect regression
splits. It removes rows whose matching `alert_summaries.csv` row belongs to
framework ids `2`, `6`, or `18`, and rows whose `created` timestamp is before
the cutoff encoded in `filter_fail_perf_sigs.py`. The current script constant
is `2025-06-20T00:00:00Z`.

Produced by:

```bash
python data_extraction/treeherder/filter_fail_perf_sigs.py
```

Downstream, `create_perf_bisect_dataset.py` uses this file as an
`alert_summary_id` allowlist. Extra columns, such as
`regressor_push_head_revision`, are informational.

### `all_commits.jsonl`

Autoland Mercurial commit metadata, one JSON object per line. Each row contains
at least:

- `node`
- `desc`
- `date`
- `parents`

The bisect scripts require this file to be parent-before-child ordered and use
it to:

- map revisions to linear indices,
- validate ancestry relationships,
- compute `num_candidate_revisions`, and
- select the revision window for `per_revision_perf_data.jsonl`.

Produced by:

```bash
python data_extraction/mercurial/fetch_all_commit.py
```

That script writes to `datasets/mozilla_perf/all_commits.jsonl` by default.

### `sig_groups.jsonl`

Signature co-occurrence groups. Each row has:

- `Sig_group_id`
- `signatures`: list of Treeherder performance signature ids.

Two signatures are put in the same group when they appear in exactly the same
set of jobs.

Produced by this upstream flow:

```bash
python data_extraction/treeherder/get_num_perf_tests.py
python data_extraction/treeherder/get_perf_sigs.py
python data_extraction/treeherder/get_sigs_per_job.py
python data_extraction/treeherder/create_sig_groups.py
```

The final script writes `datasets/mozilla_perf/sig_groups.jsonl` by default.

### `sig_group_job_durations.csv`

Mean job duration, in minutes, for each signature group.

Columns:

- `signature_group_id`
- `duration_minutes`

Produced by:

```bash
python data_extraction/treeherder/get_job_duration.py
```

That script uses `perf_jobs_by_signature.jsonl` and `sig_groups.jsonl`, then
writes `datasets/mozilla_perf/sig_group_job_durations.csv` by default.

### `per_sig_perf_data_summary.jsonl`

Per-signature Treeherder performance measurements fetched with
`replicates=False`.

Each row contains:

- `signature_id`
- `filter_stats`
- `perf_measurement_data`

Produced by:

```bash
python data_extraction/treeherder/get_perf_test_data_per_sig.py
```

The default input is
`alert_summary_fail_perf_sigs_no_fw_2_6_18.csv`. The script fetches the
`performance/summary` endpoint for each failing signature and filters
measurements to the configured UTC time window. The current window is
`2025-06-01T00:00:00Z` inclusive through `2025-11-01T00:00:00Z` exclusive.

### `per_sig_perf_data_replicates.jsonl`

Per-signature Treeherder performance measurements fetched with
`replicates=True`.

The schema matches `per_sig_perf_data_summary.jsonl`:

- `signature_id`
- `filter_stats`
- `perf_measurement_data`

Produced by the same command:

```bash
python data_extraction/treeherder/get_perf_test_data_per_sig.py
```

This file is later used to infer replicate counts per signature.

### `per_sig_perf_data_info.jsonl`

Compact per-signature metadata for the perf-bisect simulator.

Each row contains:

- `signature_id`
- `replicate_counts`
- `job_duration`
- `lower_is_better`
- `alert_threshold`
- `platform`

Produced by:

```bash
python data_extraction/treeherder/get_perf_data_info.py
```

Inputs:

- `per_sig_perf_data_replicates.jsonl`
- `sig_groups.jsonl`
- `sig_group_job_durations.csv`
- Treeherder `performance/summary` metadata for `lower_is_better`,
  `alert_threshold`, and `platform`

### `per_revision_perf_data.jsonl`

Commit-window view of performance measurements, one row per selected Autoland
revision.

Each row contains the commit fields from `all_commits.jsonl`, with `desc`
removed, plus:

- `perf_measurement_data`: measurements whose `revision` equals the commit
  `node`.

Each matched measurement is augmented with:

- `signature_id`
- `replicate`: `true` for samples from `per_sig_perf_data_replicates.jsonl`,
  `false` for samples from `per_sig_perf_data_summary.jsonl`.

Produced by:

```bash
python data_extraction/treeherder/get_perf_test_data_per_revision.py
```

Inputs:

- `all_commits.jsonl`
- `per_sig_perf_data_replicates.jsonl`
- `per_sig_perf_data_summary.jsonl`

The current revision window is `2025-06-01T00:00:00Z` inclusive through
`2025-11-01T00:00:00Z` exclusive.

### `final_test_results_perf_codebert_eval.json`

Model prediction file whose `samples` array defines the eval split boundary for
the bisect regression dataset.

Important fields:

- `samples[].commit_id`
- `samples[].true_label`
- `samples[].prediction`
- `samples[].confidence`

`create_perf_bisect_dataset.py` uses only the `samples[].commit_id` values to
find the min/max commit indices for the eval window. The predictions themselves
are not used as labels for the perf-bisect regression rows.

Produced outside this dataset directory by the model inference/evaluation
pipeline, typically `llama/run_inference.py`, then copied or renamed here.

### `final_test_results_perf_codebert_final_test.json`

Model prediction file whose `samples` array defines the final-test split
boundary. The schema and use are the same as
`final_test_results_perf_codebert_eval.json`.

Produced outside this dataset directory by the model inference/evaluation
pipeline, typically `llama/run_inference.py`, then copied or renamed here.

### `perf_bisect_regressions_eval.jsonl`

Final eval split consumed by `analysis/perf_bisect/simulation.py`.

Each row represents one alert-summary/signature pair and contains:

- `alert_summary_id`
- `good_revision`
- `bad_revision`
- `num_candidate_revisions`
- `culprit_revision`
- `failing_sig`

`failing_sig` contains:

- `signature_id`
- `Good_value`
- `bad_value`
- `alert_threshold`
- `platform`

Produced by:

```bash
python data_extraction/treeherder/create_perf_bisect_dataset.py
```

### `perf_bisect_regressions_final_test.jsonl`

Final test split consumed by `analysis/perf_bisect/simulation.py`.

The schema matches `perf_bisect_regressions_eval.jsonl`.

Produced by:

```bash
python data_extraction/treeherder/create_perf_bisect_dataset.py
```

### `perf_bisect_regressions.jsonl`

Legacy unsplit regression file. It uses the older `failing_sigs` mapping shape,
where one regression row can contain multiple failing signatures:

```json
{
  "alert_summary_id": 47207,
  "good_revision": "...",
  "bad_revision": "...",
  "num_candidate_revisions": 30,
  "culprit_revision": "...",
  "failing_sigs": {
    "5352629": {
      "Good_value": 24.62,
      "bad_value": 25.31,
      "alert_threshold": 2.0,
      "platform": "macosx1500-aarch64-shippable"
    }
  }
}
```

The current `create_perf_bisect_dataset.py` writes only the split files with the
singular `failing_sig` shape. The simulator still has compatibility code for
the old `failing_sigs` shape, but this file is not used by the default
perf-bisect simulation entrypoint.

## Generation Order

A complete rebuild is roughly:

1. Fetch Autoland commit metadata.

   ```bash
   python data_extraction/mercurial/fetch_all_commit.py
   ```

2. Fetch Treeherder alert summaries and failing performance signatures.

   ```bash
   python data_extraction/treeherder/get_perf_alerts.py
   python data_extraction/treeherder/get_failing_perf_sigs.py
   ```

3. Copy or symlink the needed upstream artifacts from `datasets/mozilla_perf/`
   into `datasets/mozilla_perf_bisect/`:

   - `alert_summaries.csv`
   - `alert_summary_fail_perf_sigs.csv`
   - `all_commits.jsonl`

4. Filter the failing-signature alert summaries for the bisect dataset.

   ```bash
   python data_extraction/treeherder/filter_fail_perf_sigs.py
   ```

5. Build signature groups and duration metadata, then copy or symlink the
   outputs into this directory.

   ```bash
   python data_extraction/treeherder/get_num_perf_tests.py
   python data_extraction/treeherder/get_perf_sigs.py
   python data_extraction/treeherder/get_sigs_per_job.py
   python data_extraction/treeherder/create_sig_groups.py
   python data_extraction/treeherder/get_job_duration.py
   ```

   Required outputs:

   - `sig_groups.jsonl`
   - `sig_group_job_durations.csv`

6. Fetch per-signature performance measurements for the filtered failing
   signatures.

   ```bash
   python data_extraction/treeherder/get_perf_test_data_per_sig.py
   ```

7. Build per-signature simulator metadata.

   ```bash
   python data_extraction/treeherder/get_perf_data_info.py
   ```

8. Join measurements back onto the revision stream.

   ```bash
   python data_extraction/treeherder/get_perf_test_data_per_revision.py
   ```

9. Copy or generate the model prediction JSON files that define the eval and
   final-test commit windows:

   - `final_test_results_perf_codebert_eval.json`
   - `final_test_results_perf_codebert_final_test.json`

10. Create the final perf-bisect regression split files.

    ```bash
    python data_extraction/treeherder/create_perf_bisect_dataset.py
    ```

## Notes

- `create_perf_bisect_dataset.py` validates that
  `all_commits.jsonl` is parent-before-child ordered.
- `create_perf_bisect_dataset.py` excludes rows whose culprit revision is not
  inside the Mercurial DAG range `(good_revision, bad_revision]`.
- `num_candidate_revisions` is computed from commit indices as
  `bad_index - good_index - 1`.
- The eval and final-test split boundaries are inferred from the min/max commit
  indices in the prediction JSON `samples` arrays.
- `per_sig_perf_data_info.jsonl` is optional for
  `create_perf_bisect_dataset.py`, but when present it supplies better
  `alert_threshold` and `platform` values for `failing_sig`.
