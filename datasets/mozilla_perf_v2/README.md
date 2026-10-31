# `mozilla_perf_v2` dataset

Training and scoring files for the perf regression predictor, built from
bugbug's performance alerts data pipeline rather than from this repo's own
extraction. See `data_extraction/bugbug/README.md` for the scripts, the
windows and the step-by-step workflow. The files are large and not tracked;
copy them to the cluster by hand.

### `perf_bugbug_struc.jsonl`

One row per commit of the training window (October 2024 to December 2025),
in push date order: `commit_id` (changeset hash), `prompt` (bugbug's exact
model input: cleaned subject and structured diff) and `response` (`"1"` for a
pinned culprit of a confirmed regression, `"0"` for a commit of a settled
clean push; five negatives per positive). `perf_bugbug_struc.jsonl.meta.json`
records the source export, its hash and the counts.

### `scoring_<period>_<context>.jsonl`

One row per commit of a scoring period: `node`, `push_id`, `pushlog_id`,
`push_time`, `ignored_reason` (bugbug's reason for not scoring the commit,
null when it is scored) and `prompt` (null for ignored commits), with the
raw patch capped for the named context length before structuring.

### `scores_<period>_<model>.jsonl`

Written on the cluster: `node`, `prob` (both class probabilities), `class`
and `risk_score` for every scored commit of the matching scoring set.
