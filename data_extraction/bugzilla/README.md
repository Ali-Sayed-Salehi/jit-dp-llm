# Bugzilla extraction (`data_extraction/bugzilla`)

This folder contains scripts that pull bug metadata from Bugzilla (bugzilla.mozilla.org) and write
dataset artifacts under:

- `datasets/mozilla_perf/` (perf regression dataset; downstream analysis/simulation)
- `datasets/mozilla_jit/` (JIT-style bug+diff dataset)

For file-level schemas and how these artifacts relate to the rest of the project, see:
- `datasets/mozilla_perf/README.md`
- `datasets/mozilla_jit/README.md`

## Prerequisites

- Network access (Bugzilla REST API).
- `BUGZILLA_API_KEY` in `secrets/.env` (recommended; some scripts can run without it but may be
  rate-limited).

## Scripts

### `get_all_bugs.py`

Fetch a large Bugzilla bug corpus (used by the `mozilla_jit` pipeline).

- **Flow**
  1. Page through Bugzilla `/rest/bug` results using `limit`/`offset`.
  2. Filter server-side to bugs created after `--since`, with `resolution in {FIXED, WONTFIX}`,
     and `classification != Graveyard`.
  3. Write one normalized JSON object per bug to JSONL (optionally resume appending).
- **Inputs**
  - Bugzilla API: `/rest/bug`
  - `secrets/.env` (optional): `BUGZILLA_API_KEY`
- **Outputs** (default)
  - `datasets/mozilla_jit/all_bugs.jsonl`

### `get_perf_bugs.py`

Fetch Bugzilla bugs for a recent window and join them against Treeherder regressions to produce
`perf_bugs.csv`.

- **Flow**
  1. Fetch Bugzilla bugs created within the configured lookback window.
  2. Write the raw bug table to `datasets/mozilla_perf/all_bugs.csv`.
  3. Load `datasets/mozilla_perf/alerts_with_bug_and_test_info.csv` (from Treeherder extraction).
  4. For each regression bug, use Bugzilla’s `regressed_by` field to identify regressor bug ids.
  5. Restrict to a “relevant” product/component set observed among regressor-linked bugs and a
     configured `START_DATE_ISO` cutoff.
  6. Write `datasets/mozilla_perf/perf_bugs.csv` with labels + joined metadata.
- **Inputs**
  - Bugzilla API: `/rest/bug`
  - `datasets/mozilla_perf/alerts_with_bug_and_test_info.csv`
  - `secrets/.env`: `BUGZILLA_API_KEY`
- **Outputs**
  - `datasets/mozilla_perf/all_bugs.csv`
  - `datasets/mozilla_perf/perf_bugs.csv`

### `get_perf_reg_bugs.py`

Collect bugs filed by the “treeherder Bug Filer” account and attempt to extract a regressor
revision (40-hex SHA1) from the filing comment text.

- **Flow**
  1. Page through Bugzilla bugs where `creator == "treeherder Bug Filer"`.
  2. For each bug id, fetch comments and search for a Treeherder regression template prefix.
  3. Extract the first SHA1-like value from the qualifying comment.
  4. Write results to CSV.
- **Inputs**
  - Bugzilla API: `/rest/bug` and `/rest/bug/<id>/comment`
  - `secrets/.env` (optional): `BUGZILLA_API_KEY`
- **Outputs**
  - `datasets/mozilla_perf/perf_reg_bugs.csv`
- **Notes**
  - The script contains a `DEBUG` flag (in-code) that limits results to a single page.

## Typical usage

- `mozilla_perf` (Bugzilla portion):
  1. `python data_extraction/treeherder/get_perf_alerts.py` (produces `alerts_with_bug_and_test_info.csv`)
  2. `python data_extraction/bugzilla/get_perf_bugs.py` (produces `perf_bugs.csv`)
- `mozilla_jit`:
  1. `python data_extraction/bugzilla/get_all_bugs.py` (produces `datasets/mozilla_jit/all_bugs.jsonl`)

