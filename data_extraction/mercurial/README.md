# Mercurial extraction (`data_extraction/mercurial`)

This folder contains scripts that work with Mozilla’s `autoland` Mercurial repository to:

- Cache commit metadata (`hg log -Tjson`) as JSONL.
- Compute a single “net diff” per bug by finding a contiguous block of `Bug <id>` commits and
  running `hg diff` between the block base and head.

Artifacts are written under:
- `datasets/mozilla_perf/` (perf regression dataset)
- `datasets/mozilla_jit/` (JIT-style bug+diff dataset)

For file-level schemas and how these artifacts relate to the rest of the project, see:
- `datasets/mozilla_perf/README.md`
- `datasets/mozilla_jit/README.md`

## Prerequisites

- Mercurial installed and available as `hg` on `PATH`.
- Network access (for cloning/pulling autoland).
- Disk space: the autoland clone and JSONL exports can be large.

Most scripts expect a local checkout at `data_extraction/mercurial/repos/autoland`.

## Scripts

### `fetch_all_commit.py`

Ensure a local autoland clone exists, pull updates, and export all commits to JSONL.

- **Flow**
  1. Clone `https://hg-edge.mozilla.org/integration/autoland` into `data_extraction/mercurial/repos/autoland`
     if missing; otherwise run `hg pull -u`.
  2. Run `hg log -Tjson -r all()` and write one JSON object per line.
- **Inputs**
  - Mercurial remote: `https://hg-edge.mozilla.org/integration/autoland`
  - Local repo path: `data_extraction/mercurial/repos/autoland`
- **Outputs**
  - `datasets/mozilla_perf/all_commits.jsonl`
- **Notes**
  - `data_extraction/mercurial/link_bug_diffs.py` defaults to reading
    `datasets/mozilla_jit/all_commits.jsonl`; you can copy/symlink the exported file if desired.

### `get_bug_diffs.py`

Build `perf_bugs_with_diff.jsonl` by attaching a net diff to each Bugzilla bug (perf dataset).

- **Flow**
  1. Load `perf_bugs.csv` (bug ids + optional `regressor_revision`).
  2. Load commit metadata from `all_commits.jsonl` to locate `Bug <id>` commits and first parents.
  3. For each bug id, find the newest contiguous block of matching commits (skipping backouts).
  4. Compute one unified “net diff” for the block:
     `hg diff -r <p1(oldest)> -r <newest>`.
  5. Write one JSONL record per bug with non-empty diff.
- **Inputs**
  - `datasets/mozilla_perf/perf_bugs.csv`
  - `datasets/mozilla_perf/all_commits.jsonl`
  - Local repo: `data_extraction/mercurial/repos/autoland`
- **Outputs**
  - `datasets/mozilla_perf/perf_bugs_with_diff.jsonl`

### `link_bug_diffs.py`

Build `mozilla_jit.jsonl` by joining Bugzilla bugs to autoland commit blocks and diffs (JIT dataset).

- **Flow**
  1. Load Bugzilla bugs (`all_bugs.jsonl`) and autoland commits (`all_commits.jsonl`).
  2. For each bug id, find the newest contiguous block of `Bug <id>` commits (skipping backouts).
  3. Compute one unified “net diff” for the block:
     `hg diff -r <p1(oldest)> -r <newest>`.
  4. Write one JSONL record per bug with non-empty diff (supports `--dry-run`/`--limit`/`--cutoff-date`).
- **Inputs** (defaults; override with CLI flags)
  - `datasets/mozilla_jit/all_bugs.jsonl`
  - `datasets/mozilla_jit/all_commits.jsonl`
  - Local repo: `data_extraction/mercurial/repos/autoland`
- **Outputs**
  - `datasets/mozilla_jit/mozilla_jit.jsonl`
- **Notes**
  - The script selects the *newest* contiguous commit block per bug; older blocks (if they exist)
    are ignored rather than treated as an error.

## Typical usage

- `mozilla_perf` (Mercurial portion):
  1. `python data_extraction/mercurial/fetch_all_commit.py`
  2. `python data_extraction/mercurial/get_bug_diffs.py`
- `mozilla_jit`:
  1. `python data_extraction/bugzilla/get_all_bugs.py`
  2. `python data_extraction/mercurial/fetch_all_commit.py` (copy/symlink into `datasets/mozilla_jit/all_commits.jsonl`)
  3. `python data_extraction/mercurial/link_bug_diffs.py`

