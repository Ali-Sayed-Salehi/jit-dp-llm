# Conduit Extraction Scripts

This directory contains scripts and exploratory notes for Mozilla Phabricator
Conduit API extraction.

## Setup

Install the repo requirements and create `secrets/.env` with:

```bash
CONDUIT_API_TOKEN=...
```

Optional:

```bash
PHABRICATOR_API_URL=https://phabricator.services.mozilla.com/api/
```

The scripts use the pinned `python-phabricator` client from `requirements.txt`.

## Current Mozilla Code Review Pipeline

Run these from the repo root.

1. Fetch per-commit DREVs:

```bash
python data_extraction/conduit/get_per_commit_drevs.py \
  --input-jsonl datasets/mozilla_code_review/all_commits.jsonl \
  --output-jsonl datasets/mozilla_code_review/per_commit_drevs.jsonl \
  --eval-predictions-json datasets/mozilla_code_review/risk_predictions_eval.json \
  --final-test-predictions-json datasets/mozilla_code_review/risk_predictions_final_test.json \
  --debug-count 10 \
  --api-url https://phabricator.services.mozilla.com/api/ \
  --rate-limit-min-interval 0.5 \
  --max-retries 5 \
  --retry-base-sleep 5.0
```

This script:

- loads Mercurial commits from `all_commits.jsonl`;
- derives eval and final-test commit boundaries from the prediction JSON files;
- scans only commits from the eval start boundary through the final-test end
  boundary;
- looks for a trailing `https://phabricator.services.mozilla.com/D...` URL in
  the commit message;
- fetches each matching DREV with `differential.revision.search`;
- converts prediction rows into a `risk_score` probability for label `1`;
- writes only published, closed DREVs to `per_commit_drevs.jsonl`.

Output row shape:

```json
{"commit_id": "...", "dataset_split": "eval", "risk_score": 0.42, "drev": {...}}
```

2. Fetch transactions for those DREVs:

```bash
python data_extraction/conduit/get_drevs_transactions.py \
  --input-jsonl datasets/mozilla_code_review/per_commit_drevs.jsonl \
  --output-jsonl datasets/mozilla_code_review/per_commit_drev_transactions.jsonl \
  --debug-count 10 \
  --api-url https://phabricator.services.mozilla.com/api/ \
  --page-limit 100 \
  --rate-limit-min-interval 0.5 \
  --max-retries 5 \
  --retry-base-sleep 5.0
```

This script:

- loads `per_commit_drevs.jsonl`;
- fetches all transaction pages for each DREV with `transaction.search`;
- preserves `dataset_split` and `risk_score`;
- writes one row per commit/DREV with all transactions nested in a list.

Output row shape:

```json
{
  "commit_id": "...",
  "drev_id": 123456,
  "dataset_split": "final test",
  "risk_score": 0.42,
  "transactions": [...]
}
```

Add `--debug` to either command for a bounded smoke test. The debug mode is
selected before Conduit calls, so it avoids unnecessary API requests.

Cluster-specific commented commands are also available in
`slurm_scripts/speed/extract_data.sh`.

## Other Files

- `explore.ipynb`
  - Notebook used to explore Conduit endpoints and raw API responses.

- `get_all_drevs.py`
  - Legacy/export script for recent DREVs by repository PHID and creation time.
  - Writes a CSV under `datasets/mozilla_perf/`.

- `get_commit_drevs.py`
  - Earlier commit-to-DREV extraction script for `datasets/mozilla_perf`.
  - Computes review-derived metrics such as change-inducing review and PR lead
    time.

- `archive/`
  - Older extraction experiments kept for reference.
