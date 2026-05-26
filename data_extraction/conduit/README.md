# Conduit Extraction Scripts

This directory contains the current Mozilla Phabricator Conduit extraction
pipeline for `datasets/mozilla_code_review/`, plus archived older experiments.

## Setup

Install the repo requirements and create `secrets/.env` with:

```bash
CONDUIT_API_TOKEN=...
```

Optional:

```bash
PHABRICATOR_API_URL=https://phabricator.services.mozilla.com/api/
```

The Conduit stages use the pinned `python-phabricator` client from
`requirements.txt`. The final dataset assembly stage also requires the
Mercurial CLI, `hg`, because it reads changed-file lists from a local autoland
clone.

## Current Mozilla Code Review Pipeline

Run these from the repo root.

1. Expand prediction scores to per-commit risk scores:

```bash
python data_extraction/conduit/get_commit_risk_scores.py \
  --input-jsonl datasets/mozilla_code_review/all_commits.jsonl \
  --eval-predictions-json datasets/mozilla_code_review/risk_predictions_eval.json \
  --final-test-predictions-json datasets/mozilla_code_review/risk_predictions_final_test.json \
  --output-jsonl datasets/mozilla_code_review/per_commit_risk_scores.jsonl
```

This script:

- validates that `all_commits.jsonl` is parent-before-child ordered;
- derives the valid eval-through-final-test Mercurial interval from the
  prediction files;
- normalizes eval and final-test prediction confidence formats into
  `risk_score = P(label 1)`;
- walks backward from each directly scored commit across contiguous commits with
  the same leading `Bug <id>` marker;
- writes one risk-score row for every directly scored or inherited commit in
  the valid interval.

Output row shape:

```json
{"commit_id": "...", "risk_score": 0.42, "desc": "Bug ..."}
```

2. Fetch published, closed DREVs for scored commits:

```bash
python data_extraction/conduit/get_per_commit_drevs.py \
  --input-jsonl datasets/mozilla_code_review/all_commits.jsonl \
  --output-jsonl datasets/mozilla_code_review/per_commit_drevs.jsonl \
  --eval-predictions-json datasets/mozilla_code_review/risk_predictions_eval.json \
  --final-test-predictions-json datasets/mozilla_code_review/risk_predictions_final_test.json \
  --risk-scores-jsonl datasets/mozilla_code_review/per_commit_risk_scores.jsonl \
  --api-url https://phabricator.services.mozilla.com/api/ \
  --rate-limit-min-interval 0.5 \
  --max-retries 5 \
  --retry-base-sleep 5.0
```

This script:

- loads risk scores from `per_commit_risk_scores.jsonl`;
- scans only the eval-through-final-test Mercurial interval;
- keeps only commits with a leading bug id and a risk score;
- looks for a trailing `https://phabricator.services.mozilla.com/D...` URL in
  the commit message;
- fetches each matching DREV with `differential.revision.search`;
- writes only published, closed DREVs to `per_commit_drevs.jsonl`.

Output row shape:

```json
{"commit_id": "...", "dataset_split": "eval", "risk_score": 0.42, "drev": {...}}
```

3. Fetch transactions for those DREVs:

```bash
python data_extraction/conduit/get_drevs_transactions.py \
  --input-jsonl datasets/mozilla_code_review/per_commit_drevs.jsonl \
  --output-jsonl datasets/mozilla_code_review/per_commit_drev_transactions.jsonl \
  --api-url https://phabricator.services.mozilla.com/api/ \
  --page-limit 100 \
  --rate-limit-min-interval 0.5 \
  --max-retries 5 \
  --retry-base-sleep 5.0
```

This script:

- loads `per_commit_drevs.jsonl`;
- validates `dataset_split`, DREV id/PHID, and `risk_score`;
- fetches all transaction pages for each unique DREV PHID with
  `transaction.search`;
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

4. Build the compact review dataset:

```bash
python data_extraction/conduit/create_code_review_dataset.py \
  --drev-transactions-jsonl datasets/mozilla_code_review/per_commit_drev_transactions.jsonl \
  --output-jsonl datasets/mozilla_code_review/drev_review_data.jsonl \
  --autoland-repo data_extraction/mercurial/repos/autoland \
  --autoland-url https://hg-edge.mozilla.org/integration/autoland \
  --hg-pull-max-attempts 5 \
  --hg-pull-retry-base-sleep 5.0
```

This script:

- clones autoland if missing, or runs `hg pull -u URL` if it already exists;
- retries failed pulls to tolerate intermittent Mercurial HTTP failures;
- supports `--skip-repo-update` when the local clone is already sufficient;
- reads changed files with `hg status --change <commit_id>`;
- extracts non-empty Phabricator transaction comments into chronological
  `reviews`;
- omits Phabricator application comments by default unless
  `--include-app-comments` is passed.

Output row shape:

```json
{
  "commit_id": "...",
  "dataset_split": "eval",
  "risk_score": 0.42,
  "drev_submission_date": "2024-09-24T14:59:23Z",
  "drev_closed_merged_date": "2024-09-30T15:08:41Z",
  "drev_author": "PHID-USER-...",
  "files_changed": ["path/to/file.cpp"],
  "reviews": [{"author": "PHID-USER-...", "submission_date": "...", "comment": "..."}]
}
```

Add `--debug` to stages 2, 3, or 4 for a split-balanced smoke test. Debug mode
selects the bounded subset before API calls or dataset assembly.

Cluster-specific commented commands are also available in
`slurm_scripts/speed/extract_data.sh`.

## Current Files

- `get_commit_risk_scores.py`
  - Converts prediction JSON confidence formats into per-commit `risk_score`
    rows and expands scores across contiguous same-bug commit blocks.

- `get_per_commit_drevs.py`
  - Fetches published, closed Differential Revisions linked by trailing DREV
    URLs in eligible Mercurial commit messages.

- `get_drevs_transactions.py`
  - Fetches full Conduit transaction history for each DREV row.

- `create_code_review_dataset.py`
  - Builds the final compact code-review JSONL with changed files and extracted
    review comments.

- `explore.ipynb`
  - Notebook used to explore Conduit endpoints and raw API responses.

- `archive/`
  - Older extraction experiments kept for reference, including the previous
    `get_all_drevs.py` and `get_commit_drevs.py` scripts.
