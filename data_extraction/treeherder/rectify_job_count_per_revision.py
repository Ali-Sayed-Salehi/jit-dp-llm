#!/usr/bin/env python3
"""
Rectify per-revision perf job counts by signature-grouping.

Input
  - datasets/mozilla_perf/perf_jobs_per_revision_details.jsonl
      {"revision": "...", "submit_time_iso": "...", "total_jobs": 5,
       "signature_ids": [2304398, ...]}
  - datasets/mozilla_perf/all_signatures.jsonl
      Each line: a signature metadata object with at least `id` and `framework_id`.
  - sig_groups.jsonl (default: /speed-scratch/.../datasets/mozilla_perf/sig_groups.jsonl)
      {"Sig_group_id": 1083, "signatures": [5436251, 5436252]}

Output
  - datasets/mozilla_perf/perf_jobs_per_revision_details_rectified.jsonl
    Same record fields, except:
      - replace `signature_ids` with `signature_group_ids`
      - set `total_jobs` to len(unique(signature_group_ids))
      - drop signature-groups where all signatures have a `framework_id` in {2, 6, 18}
  - Prints aggregate job-count stats across all revisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Set, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

DEFAULT_INPUT_JSONL = os.path.join(DATASET_DIR, "perf_jobs_per_revision_details.jsonl")
DEFAULT_ALL_SIGNATURES_JSONL = os.path.join(DATASET_DIR, "all_signatures.jsonl")
DEFAULT_OUTPUT_JSONL = os.path.join(
    DATASET_DIR, "perf_jobs_per_revision_details_rectified.jsonl"
)
DEFAULT_SIG_GROUPS_JSONL = os.path.join(
    DATASET_DIR, "sig_groups.jsonl"
)

DEFAULT_EXCLUDED_FRAMEWORK_IDS = (2, 6, 18)


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


def load_sig_groups_mapping(sig_groups_jsonl: str) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    sig_to_group: Dict[int, int] = {}
    group_to_sigs: Dict[int, List[int]] = {}
    for line_num, row in _iter_jsonl(sig_groups_jsonl):
        group_id = row.get("Sig_group_id")
        signatures = row.get("signatures")
        if not isinstance(group_id, int) or not isinstance(signatures, list):
            raise ValueError(
                f"Bad row on line {line_num} of {sig_groups_jsonl}: "
                f"expected {{'Sig_group_id': int, 'signatures': list}}"
            )
        group_sigs = group_to_sigs.setdefault(group_id, [])
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
            group_sigs.append(sig)
    return sig_to_group, group_to_sigs


def load_signature_to_framework_id(all_signatures_jsonl: str) -> Dict[int, int | None]:
    sig_to_framework: Dict[int, int | None] = {}
    for line_num, row in _iter_jsonl(all_signatures_jsonl):
        sig_id = row.get("id")
        try:
            sig_id_int = int(sig_id)
        except (TypeError, ValueError):
            continue

        framework_id = row.get("framework_id")
        framework_id_int: int | None
        if framework_id is None:
            framework_id_int = None
        else:
            try:
                framework_id_int = int(framework_id)
            except (TypeError, ValueError):
                framework_id_int = None

        if sig_id_int not in sig_to_framework:
            sig_to_framework[sig_id_int] = framework_id_int
            continue

        prev = sig_to_framework[sig_id_int]
        if prev is None and framework_id_int is not None:
            sig_to_framework[sig_id_int] = framework_id_int
        elif prev is not None and framework_id_int is None:
            pass
        elif prev != framework_id_int:
            raise ValueError(
                f"Signature id {sig_id_int} has conflicting framework_id values "
                f"({prev}, {framework_id_int}) in {all_signatures_jsonl} "
                f"(latest seen at line {line_num})."
            )
    return sig_to_framework


def get_excluded_group_ids(
    group_to_sigs: Dict[int, List[int]],
    sig_to_framework: Dict[int, int | None],
    excluded_framework_ids: Set[int],
) -> Tuple[Set[int], Set[int]]:
    excluded_group_ids: Set[int] = set()
    missing_framework_sig_ids: Set[int] = set()

    for group_id, sig_ids in group_to_sigs.items():
        if not sig_ids:
            continue

        all_in_excluded_frameworks = True
        for sig_id in sig_ids:
            framework_id = sig_to_framework.get(sig_id)
            if framework_id is None:
                missing_framework_sig_ids.add(sig_id)
                all_in_excluded_frameworks = False
                break
            if framework_id not in excluded_framework_ids:
                all_in_excluded_frameworks = False
                break

        if all_in_excluded_frameworks:
            excluded_group_ids.add(group_id)

    return excluded_group_ids, missing_framework_sig_ids


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
    excluded_group_ids: Set[int],
    *,
    strict: bool,
) -> None:
    missing_sig_first_line: Dict[int, int] = {}
    total_rows = 0
    total_jobs_non_unique = 0
    total_jobs_unique_per_revision = 0
    unique_group_ids_all_revisions: Set[int] = set()

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
                    missing_sig_first_line.setdefault(sig, line_num)
                    if strict:
                        raise KeyError(
                            f"Signature id {sig} (line {line_num} of {input_jsonl}) "
                            "not found in sig_groups mapping input"
                        )
                    continue
                group_ids.append(group_id)

            if excluded_group_ids:
                group_ids = [gid for gid in group_ids if gid not in excluded_group_ids]

            total_jobs_non_unique += len(group_ids)

            group_ids = stable_unique(group_ids)
            total_jobs_unique_per_revision += len(group_ids)
            unique_group_ids_all_revisions.update(group_ids)

            row.pop("signature_ids", None)
            row["signature_group_ids"] = group_ids
            row["total_jobs"] = len(group_ids)

            out_f.write(json.dumps(row) + "\n")

    if missing_sig_first_line:
        missing_sigs_sorted = sorted(missing_sig_first_line.items(), key=lambda kv: kv[0])
        missing_preview = ", ".join(f"{sig}@{ln}" for sig, ln in missing_sigs_sorted[:20])
        more = (
            ""
            if len(missing_sig_first_line) <= 20
            else f" (+{len(missing_sig_first_line) - 20} more)"
        )
        print(
            f"[WARN] {len(missing_sig_first_line)} signature ids missing from sig_groups mapping "
            f"(showing sig@first_line): "
            f"{missing_preview}{more}",
            file=sys.stderr,
        )

    print(f"Wrote {total_rows} rows to {output_jsonl}")
    print(
        f"Total jobs across all revisions (non-unique signature-group IDs): {total_jobs_non_unique}"
    )
    print(
        "Total jobs across all revisions (unique signature-group IDs per revision): "
        f"{total_jobs_unique_per_revision}"
    )
    print(
        "Unique signature-group IDs across all revisions: "
        f"{len(unique_group_ids_all_revisions)}"
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replace signature IDs with signature-group IDs per revision."
    )
    p.add_argument("--input", default=DEFAULT_INPUT_JSONL, help="Input JSONL path.")
    p.add_argument(
        "--all-signatures",
        default=DEFAULT_ALL_SIGNATURES_JSONL,
        help="all_signatures.jsonl path containing signature metadata (including framework_id).",
    )
    p.add_argument(
        "--sig-groups",
        default=DEFAULT_SIG_GROUPS_JSONL,
        help="sig_groups.jsonl path mapping signatures to Sig_group_id.",
    )
    p.add_argument("--output", default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")
    p.add_argument(
        "--exclude-framework-ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_EXCLUDED_FRAMEWORK_IDS),
        help=(
            "Exclude signature-groups where all signatures have a framework_id in this set "
            "(default: 2 6 18)."
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--strict",
        action="store_true",
        help="Error out if a signature ID is not found in sig_groups mapping.",
    )
    g.add_argument(
        "--allow-missing-signatures",
        action="store_true",
        help="(Default) Skip signature IDs not found in sig_groups mapping (warn at end).",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2
    if not os.path.exists(args.all_signatures):
        print(f"all_signatures not found: {args.all_signatures}", file=sys.stderr)
        return 2
    if not os.path.exists(args.sig_groups):
        print(f"sig_groups not found: {args.sig_groups}", file=sys.stderr)
        return 2

    sig_to_group, group_to_sigs = load_sig_groups_mapping(args.sig_groups)
    sig_to_framework = load_signature_to_framework_id(args.all_signatures)

    excluded_framework_ids = set(args.exclude_framework_ids)
    excluded_group_ids, missing_framework_sig_ids = get_excluded_group_ids(
        group_to_sigs, sig_to_framework, excluded_framework_ids
    )
    if excluded_group_ids:
        excluded_preview = ", ".join(str(i) for i in sorted(excluded_framework_ids))
        print(
            f"[INFO] Excluding {len(excluded_group_ids)} signature-groups where all signatures have "
            f"framework_id in {{{excluded_preview}}}.",
            file=sys.stderr,
        )
    if missing_framework_sig_ids:
        preview = ", ".join(str(i) for i in sorted(missing_framework_sig_ids)[:20])
        more = (
            ""
            if len(missing_framework_sig_ids) <= 20
            else f" (+{len(missing_framework_sig_ids) - 20} more)"
        )
        print(
            f"[WARN] {len(missing_framework_sig_ids)} signature ids missing framework_id in "
            f"{args.all_signatures} (showing first 20): {preview}{more}",
            file=sys.stderr,
        )

    rectify(
        args.input,
        args.output,
        sig_to_group,
        excluded_group_ids,
        strict=args.strict,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
