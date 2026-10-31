#!/usr/bin/env python
"""Build the perf regression predictor training set from bugbug's export.

Reads ``perf_regression_commits.json``, the commit dataset written by
``bugbug-data-perf-alerts export`` (one commit per line with ``node``,
``commit_message``, ``diff``, ``label`` and ``pushdate``), and writes the
``<commit_id, prompt, response>`` JSON lines file ``llama/train.py`` consumes,
in push date order so the chronological split holds.

Prompts are built with bugbug's own ``build_model_input``: the cleaned commit
subject and the structured diff between the eight structural tags. That is
the text the prediction endpoint and the commit scorer feed the model, so the
model trains on exactly what it will see in production. bugbug caps the raw
diff at ``max_length * 64`` characters before structuring; ``--max-chars``
defaults to that cap for a 32k-token model, which covers every candidate, and
the tokenizer truncates the rest at training time.

Run with the bugbug virtualenv, which has bugbug and its dependencies:

    ~/repos/bugbug/.venv/bin/python data_extraction/bugbug/build_training_set.py \\
        --export ~/perf-alerts-full/data/perf_regression_commits.json \\
        --output datasets/mozilla_perf_v2/perf_bugbug_struc.jsonl

A ``<output>.meta.json`` records where the rows came from and how many.
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

DEFAULT_MAX_CHARS = 32768 * 64


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repo_dir: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, help="bugbug's perf_regression_commits.json")
    parser.add_argument("--output", required=True, help="JSON lines file for llama/train.py")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Raw diff characters kept before structuring, as bugbug does for a model of max_length tokens (max_length * 64). Default: %(default)s.",
    )
    args = parser.parse_args()

    from bugbug.models.perf_regression_predictor import PerfRegressionPredictorModel

    import bugbug

    builder = PerfRegressionPredictorModel()

    rows = []
    with open(args.export, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            rows.append(
                (
                    record["pushdate"],
                    record["node"],
                    record.get("commit_message") or "",
                    record["diff"],
                    int(record["label"]),
                )
            )
    rows.sort(key=lambda row: (row[0], row[1]))
    print(f"read {len(rows)} commits from {args.export}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    positives = 0
    prompt_chars = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for index, (pushdate, node, message, diff, label) in enumerate(rows, 1):
            prompt = builder.build_model_input(message, diff[: args.max_chars])
            out.write(
                json.dumps({"commit_id": node, "prompt": prompt, "response": str(label)})
                + "\n"
            )
            positives += label
            prompt_chars += len(prompt)
            if index % 1000 == 0:
                print(f"  {index} of {len(rows)} prompts built")

    meta = {
        "source": os.path.abspath(args.export),
        "source_sha256": sha256_of(args.export),
        "rows": len(rows),
        "positives": positives,
        "negatives": len(rows) - positives,
        "pushdate_from": rows[0][0] if rows else None,
        "pushdate_to": rows[-1][0] if rows else None,
        "max_chars": args.max_chars,
        "mean_prompt_chars": round(prompt_chars / len(rows)) if rows else 0,
        "bugbug_revision": git_revision(os.path.dirname(os.path.dirname(bugbug.__file__))),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.output + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(
        f"wrote {len(rows)} rows ({positives} positive) from {meta['pushdate_from']} to "
        f"{meta['pushdate_to']} to {args.output}; mean prompt {meta['mean_prompt_chars']} chars"
    )


if __name__ == "__main__":
    main()
