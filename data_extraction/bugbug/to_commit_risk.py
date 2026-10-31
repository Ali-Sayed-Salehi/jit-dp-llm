#!/usr/bin/env python
"""Turn cluster scores into perf_commit_risk.json records and merge them.

Joins the scoring set written by ``build_scoring_set.py`` (push fields and
ignore reasons) with the scores written by ``llama/score_commits.py`` on the
cluster, one line per scored node with ``prob``, ``class`` and ``risk_score``,
into records shaped like bugbug's commit risk scorer writes them, under
``--model-version``. Ignored commits get their reason and null scores, as in
the pipeline. The records are merged into ``--archive``, replacing any record
of the same node and model version, so the simulator can pick the model with
``--model-version`` and the pipeline's ``calibrate`` command fits a
calibration for it.

    python data_extraction/bugbug/to_commit_risk.py \\
        --scoring-set datasets/mozilla_perf_v2/scoring_2026_8k.jsonl \\
        --scores datasets/mozilla_perf_v2/scores_2026_mbert.jsonl \\
        --model-dir LLMs/trained/sequence-classification/perf_mbert_v2/perfregressionpredictormodel \\
        --archive ~/perf-alerts-full/data/perf_commit_risk.json
"""

import argparse
import json
import os
from datetime import datetime, timezone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring-set", required=True, help="Output of build_scoring_set.py")
    parser.add_argument("--scores", required=True, help="Output of llama/score_commits.py")
    version = parser.add_mutually_exclusive_group(required=True)
    version.add_argument("--model-version", help="Version recorded with the scores")
    version.add_argument("--model-dir", help="Packaged checkpoint directory; its bugbug_model.json gives the version")
    parser.add_argument("--archive", required=True, help="perf_commit_risk.json to merge into (created when missing)")
    parser.add_argument("--scored-at", help="ISO timestamp recorded with the scores. Default: now.")
    args = parser.parse_args()

    model_version = args.model_version
    if args.model_dir:
        with open(os.path.join(args.model_dir, "bugbug_model.json"), encoding="utf-8") as fh:
            model_version = json.load(fh)["model_version"]
    scored_at = args.scored_at or datetime.now(timezone.utc).isoformat()

    scores = {}
    with open(args.scores, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            scores[row["node"]] = row

    records = []
    missing = 0
    with open(args.scoring_set, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            record = {
                "node": row["node"],
                "push_id": row["push_id"],
                "pushlog_id": row.get("pushlog_id"),
                "push_time": row["push_time"],
                "ignored_reason": row["ignored_reason"],
                "prob": None,
                "class": None,
                "risk_score": None,
                "model_version": model_version,
                "scored_at": scored_at,
            }
            if row["ignored_reason"] is None:
                score = scores.get(row["node"])
                if score is None:
                    missing += 1
                    continue
                record.update(prob=score["prob"], **{"class": score["class"]}, risk_score=score["risk_score"])
            records.append(record)
    if missing:
        print(f"warning: {missing} commits to score have no score yet; they are left out")

    previous = []
    if os.path.exists(args.archive):
        with open(args.archive, encoding="utf-8") as fh:
            previous = [json.loads(line) for line in fh]
    replaced = {(record["node"], record["model_version"]) for record in records}
    kept = [record for record in previous if (record["node"], record["model_version"]) not in replaced]
    merged = kept + records
    merged.sort(key=lambda r: (r["push_time"], r["push_id"], r["node"], r["model_version"]))
    with open(args.archive, "w", encoding="utf-8") as fh:
        for record in merged:
            fh.write(json.dumps(record) + "\n")
    scored = sum(record["risk_score"] is not None for record in records)
    print(
        f"{len(records)} records for model {model_version} ({scored} scored, {len(records) - scored} ignored); "
        f"{len(previous) - len(kept)} replaced; archive now holds {len(merged)} records"
    )


if __name__ == "__main__":
    main()
