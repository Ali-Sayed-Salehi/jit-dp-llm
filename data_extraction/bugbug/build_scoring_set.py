#!/usr/bin/env python
"""Prepare the commits of a period for GPU scoring on the cluster.

For every commit of the autoland pushes in ``[--since, --until)`` from the
pipeline's ``perf_pushes.json``, writes one JSON line with the push fields a
``perf_commit_risk.json`` record carries (``node``, ``push_id``,
``pushlog_id``, ``push_time``), bugbug's ``ignored_reason`` (merges, wptsync
imports, ignore-this-changeset; null for commits to score) and, for commits
to score, the ``prompt`` the prediction endpoint would build from the
commit's patch: bugbug's commit message extraction, subject cleaning and diff
structuring, with the raw patch capped at ``--max-length * 64`` characters
first, as bugbug does for a model with that many tokens of context. Several
``--max-length`` values write one file each from a single pass over the
clone; ``--output`` then needs a ``{max_length}`` placeholder.

The output goes to the cluster, where ``llama/score_commits.py`` adds the
scores, and comes back to ``to_commit_risk.py``. Resumable: rows already in
the output are skipped.

Needs the bugbug virtualenv and a Mercurial clone holding the commits:

    ~/repos/bugbug/.venv/bin/python data_extraction/bugbug/build_scoring_set.py \\
        --data-dir ~/perf-alerts-full/data --repo-dir ~/repos/mozilla-unified \\
        --since 2026-01-01 --until 2026-10-28 --max-length 512 8192 16384 \\
        --output datasets/mozilla_perf_v2/scoring_2026_{max_length}.jsonl
"""

import argparse
import collections
import json
import os
import time
from datetime import datetime, timezone


def parse_date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory holding the pipeline's perf_pushes.json")
    parser.add_argument("--repo-dir", required=True, help="Mercurial clone with the commits (mozilla-unified)")
    parser.add_argument("--since", type=parse_date, required=True, help="First push time to include (ISO 8601, UTC)")
    parser.add_argument("--until", type=parse_date, required=True, help="Push time to stop before (ISO 8601, UTC)")
    parser.add_argument("--max-length", type=int, nargs="+", default=[8192], help="Token context of the model(s) the prompts are for; caps the raw patch at 64 characters per token. Default: %(default)s.")
    parser.add_argument("--output", required=True, help="JSON lines file for llama/score_commits.py; with several --max-length values it must contain {max_length}")
    parser.add_argument("--chunk-size", type=int, default=50, help="Commits described per hg log call. Default: %(default)s.")
    args = parser.parse_args()

    import hglib

    from bugbug import repository
    from bugbug.models.perf_regression_predictor import (
        MAX_CHARS_PER_TOKEN,
        PatchCommitMessageExtractor,
        PerfRegressionPredictorModel,
    )
    from bugbug.perf.commit_risk import (
        changesets_from_clone,
        get_ignore_reason,
        select_unscored_commits,
    )

    builder = PerfRegressionPredictorModel()
    extract_message = PatchCommitMessageExtractor()
    if len(args.max_length) > 1 and "{max_length}" not in args.output:
        raise SystemExit("--output needs a {max_length} placeholder with several --max-length values")
    outputs = {length: args.output.format(max_length=length) for length in args.max_length}

    with open(os.path.join(args.data_dir, "perf_pushes.json"), encoding="utf-8") as fh:
        pushes = [json.loads(line) for line in fh]
    # Resume from the commits every output already holds.
    done = None
    for path in outputs.values():
        nodes = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                nodes = {json.loads(line)["node"] for line in fh}
        done = nodes if done is None else done & nodes
    done = done or set()
    if done:
        print(f"resuming: {len(done)} commits already in the outputs")
    commits = select_unscored_commits(pushes, done, args.since, args.until)
    print(f"{len(commits)} commits to prepare between {args.since} and {args.until}")

    stats: collections.Counter = collections.Counter()
    started = time.monotonic()
    for path in outputs.values():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handles = {length: open(path, "a", encoding="utf-8") for length, path in outputs.items()}
    with hglib.open(args.repo_dir) as hg:
        for start in range(0, len(commits), args.chunk_size):
            chunk = commits[start : start + args.chunk_size]
            changesets = changesets_from_clone(hg, (commit["node"] for commit in chunk))
            for commit in chunk:
                changeset = changesets.get(commit["node"])
                if changeset is None:
                    stats["missing from clone"] += 1
                    continue
                row = {**commit, "ignored_reason": get_ignore_reason(changeset), "prompt": None}
                if row["ignored_reason"] is None:
                    patch = repository.get_commit_patch(hg, commit["node"].encode("ascii"))
                    patch_text = patch.decode("utf-8", "replace")
                    message = extract_message(patch_text) or ""
                    stats["to score"] += 1
                    for length, out in handles.items():
                        prompt = builder.build_model_input(message, patch_text[: length * MAX_CHARS_PER_TOKEN])
                        out.write(json.dumps({**row, "prompt": prompt}) + "\n")
                else:
                    stats["ignored"] += 1
                    for out in handles.values():
                        out.write(json.dumps(row) + "\n")
            if (start // args.chunk_size) % 20 == 19:
                for out in handles.values():
                    out.flush()
                elapsed = time.monotonic() - started
                print(f"  {start + len(chunk)} of {len(commits)} commits, {elapsed / 60:.1f} min, {dict(stats)}")
    for out in handles.values():
        out.close()
    print(f"done: {dict(stats)} -> {', '.join(outputs.values())}; a commit missing from the clone is left for a later run after a pull")


if __name__ == "__main__":
    main()
