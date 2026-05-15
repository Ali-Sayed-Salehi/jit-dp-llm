#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" analysis/perf_bisect/simulation.py \
  --dataset all \
  --regression-dir datasets/mozilla_perf_bisect \
  --signature-info datasets/mozilla_perf_bisect/per_sig_perf_data_info.jsonl \
  --revision-data datasets/mozilla_perf_bisect/per_revision_perf_data.jsonl \
  --output-dir analysis/perf_bisect/results \
  --workers 1 \
  --test-duration-minutes 1.0 \
  --oracles SummaryComparison \
  --localizers Backfill \
  --backfill-non-monotonic-retrigger-count 2 \
  --random-seed 0 \
  "$@"
