# `apachejit` dataset

Apache JIT defect prediction dataset (commit-level) plus extracted diffs and LLM-ready prompt files.

## Files and schema

### `apachejit_total.csv` (CSV)

**What it is:** Base commit-level JIT dataset (features + labels).

**Used by:** `data_extraction/github_api/get_jit_dataset.py`, `data_extraction/data_preparation.py`

**Columns (18):**
- `commit_id` (str): git SHA
- `project` (str): `owner/repo` (e.g. `apache/groovy`)
- `buggy` (bool-like): whether this is a bug-inducing commit
- `fix` (bool-like): whether this is a bug-fixing commit (dataset-provided label)
- `year` (int)
- `author_date` (int): unix timestamp (seconds)
- `la`, `ld` (int): lines added / deleted
- `nf`, `nd`, `ns` (int): files / directories / subsystems touched
- `ent` (float): change entropy
- `ndev` (int): number of developers who previously touched the files
- `age` (float|int): time since last change (dataset-provided unit)
- `nuc` (int): number of unique changes in files
- `aexp`, `arexp`, `asexp` (float|int): author experience metrics

### `apachejit_total_with_struc_diff.jsonl` (JSONL)

**What it is:** `apachejit_total.csv` rows plus a `diff` field containing a *structured diff string*.

**Produced by:** `data_extraction/github_api/get_jit_dataset.py --dataset_name apachejit --struc`

**Used by:** `data_extraction/data_preparation.py --mode jit_llm_struc`

**Per-line object fields (19):** all CSV columns above plus:
- `diff` (str): structured diff markup starting with `<COMMIT_MESSAGE>…</COMMIT_MESSAGE>` and then
  repeated `<FILE>` blocks containing `<ADDED>` / `<REMOVED>` sections.

**How the structured diff is built:** `data_extraction/github_api/javalang_structured_diff.py:extract_structured_diff()`.
It clones the target repo, then emits:
- `<COMMIT_MESSAGE>…</COMMIT_MESSAGE>`
- For each changed file: `<FILE>` + file path + blocks like `<ADDED>…</ADDED>` / `<REMOVED>…</REMOVED>`
  (optionally with `path="…"` attributes when `--ast` is used).

### `apachejit_total_llm_struc.jsonl` (JSONL)

**What it is:** LLM-ready classification dataset built from structured diffs.

**Produced by:** `data_extraction/data_preparation.py --mode jit_llm_struc --dataset_name apachejit --dataset_size total`

**Per-line object fields (3):**
- `commit_id` (str)
- `prompt` (str): structured diff string (from `apachejit_total_with_struc_diff.jsonl.diff`)
- `response` (str): `"1"` if `buggy` else `"0"`

### `apachejit_total_llm_struc_meta.jsonl` (JSONL)

**What it is:** Same as `apachejit_total_llm_struc.jsonl`, but prefixes bucketized metadata lines.

**Produced by:** `data_extraction/data_preparation.py --mode jit_llm_struc --include_metadata ...`

**Per-line object fields (3):** `commit_id`, `prompt`, `response`

## Related (not always present)

Depending on flags to `data_extraction/github_api/get_jit_dataset.py`, you may also produce:
- `apachejit_total_with_diff.jsonl` (raw `.diff` text from GitHub instead of structured diffs)
- `apachejit_total_with_struc_ast_diff.jsonl` (structured diffs with AST path annotations)
- `apachejit_total_failed*.csv` (rows that failed diff extraction)

