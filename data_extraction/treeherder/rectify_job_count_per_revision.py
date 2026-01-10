#!/usr/bin/env python3
"""
Rectify per-revision perf job counts by signature-grouping.

Input
  - datasets/mozilla_perf/perf_jobs_per_revision_details.jsonl
      {"revision": "...", "submit_time_iso": "...", "total_jobs": 5,
       "signature_ids": [2304398, ...]}
  - sig_groups.jsonl (default: /speed-scratch/.../datasets/mozilla_perf/sig_groups.jsonl)
      {"Sig_group_id": 1083, "signatures": [5436251, 5436252]}

Output
  - datasets/mozilla_perf/perf_jobs_per_revision_details_rectified.jsonl
    Same record fields, except:
      - replace `signature_ids` with `signature_group_ids`
      - set `total_jobs` to len(unique(signature_group_ids))
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

DEFAULT_INPUT_JSONL = os.path.join(DATASET_DIR, "perf_jobs_per_revision_details.jsonl")
DEFAULT_OUTPUT_JSONL = os.path.join(
    DATASET_DIR, "perf_jobs_per_revision_details_rectified.jsonl"
)
DEFAULT_SIG_GROUPS_JSONL = os.path.join(
    DATASET_DIR, "sig_groups.jsonl"
)


def _iter_jsonl(path: str) -> Iterable[Tuple[int, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_num, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e


def load_signature_to_group(sig_groups_jsonl: str) -> Dict[int, int]:
    sig_to_group: Dict[int, int] = {}
    for line_num, row in _iter_jsonl(sig_groups_jsonl):
        group_id = row.get("Sig_group_id")
        signatures = row.get("signatures")
        if not isinstance(group_id, int) or not isinstance(signatures, list):
            raise ValueError(
                f"Bad row on line {line_num} of {sig_groups_jsonl}: "
                f"expected {{'Sig_group_id': int, 'signatures': list}}"
            )
        for sig in signatures:
            if not isinstance(sig, int):
                raise ValueError(
                    f"Bad signature id on line {line_num} of {sig_groups_jsonl}: {sig!r}"
                )
            prev = sig_to_group.get(sig)
            if prev is not None and prev != group_id:
                raise ValueError(
                    f"Signature {sig} maps to multiple groups ({prev}, {group_id}) "
                    f"in {sig_groups_jsonl}"
                )
            sig_to_group[sig] = group_id
    return sig_to_group


def stable_unique(values: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def rectify(
    input_jsonl: str,
    output_jsonl: str,
    sig_to_group: Dict[int, int],
    *,
    allow_missing_signatures: bool,
) -> None:
    missing_sigs = set()
    total_rows = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for line_num, row in _iter_jsonl(input_jsonl):
            total_rows += 1

            sig_ids = row.get("signature_ids")
            if not isinstance(sig_ids, list):
                raise ValueError(
                    f"Bad row on line {line_num} of {input_jsonl}: "
                    "expected `signature_ids` to be a list"
                )

            group_ids: List[int] = []
            for sig in sig_ids:
                if not isinstance(sig, int):
                    raise ValueError(
                        f"Bad signature id on line {line_num} of {input_jsonl}: {sig!r}"
                    )
                group_id = sig_to_group.get(sig)
                if group_id is None:
                    missing_sigs.add(sig)
                    if allow_missing_signatures:
                        continue
                    raise KeyError(
                        f"Signature id {sig} (line {line_num} of {input_jsonl}) "
                        "not found in sig_groups mapping input"
                    )
                group_ids.append(group_id)

            group_ids = stable_unique(group_ids)

            row.pop("signature_ids", None)
            row["signature_group_ids"] = group_ids
            row["total_jobs"] = len(group_ids)

            out_f.write(json.dumps(row) + "\n")

    if missing_sigs:
        missing_preview = ", ".join(map(str, sorted(missing_sigs)[:20]))
        more = "" if len(missing_sigs) <= 20 else f" (+{len(missing_sigs) - 20} more)"
        print(
            f"[WARN] {len(missing_sigs)} signature ids missing from sig_groups mapping: "
            f"{missing_preview}{more}",
            file=sys.stderr,
        )

    print(f"Wrote {total_rows} rows to {output_jsonl}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replace signature IDs with signature-group IDs per revision."
    )
    p.add_argument("--input", default=DEFAULT_INPUT_JSONL, help="Input JSONL path.")
    p.add_argument(
        "--sig-groups",
        default=DEFAULT_SIG_GROUPS_JSONL,
        help="sig_groups.jsonl path mapping signatures to Sig_group_id.",
    )
    p.add_argument("--output", default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")
    p.add_argument(
        "--allow-missing-signatures",
        action="store_true",
        help="Skip signature IDs not found in sig_groups mapping (warn at end).",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    if not os.path.exists(args.sig_groups):
        print(f"sig_groups not found: {args.sig_groups}", file=sys.stderr)
        return 2

    sig_to_group = load_signature_to_group(args.sig_groups)
    rectify(
        args.input,
        args.output,
        sig_to_group,
        allow_missing_signatures=args.allow_missing_signatures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
