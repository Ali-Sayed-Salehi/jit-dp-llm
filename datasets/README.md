# Datasets

This repository stores extracted datasets under `datasets/`.

Each subdirectory contains its own `README.md` describing:
- The files present in that dataset directory and their schema (CSV columns / JSONL fields)
- Which scripts under `data_extraction/` produce and/or consume each file

## Subdirectories

- `datasets/mozilla_perf/`: Mozilla performance regression dataset (Treeherder + Bugzilla + Mercurial).
- `datasets/mozilla_jit/`: Mozilla JIT-style bug/regressor dataset (Bugzilla + Mercurial).
- `datasets/apachejit/`: Apache JIT defect prediction dataset (GitHub + structured diffs).

## Formats

- `*.jsonl` is JSON Lines: one JSON object per line.
- `*.csv` is comma-separated values. Some CSV columns contain Python-literal strings
  (e.g. a stringified list/dict) because the extraction scripts use `ast.literal_eval`
  when re-loading them.

