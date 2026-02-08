#!/usr/bin/env python3
"""
Compute "historical risk" scores per perf signature-group.

This script aggregates regressing perf signatures from:
  - datasets/mozilla_perf/alert_summary_fail_perf_sigs.csv
and maps them into signature-groups from:
  - datasets/mozilla_perf/sig_groups.jsonl

To avoid leaking future information during simulation/optimization, only
revisions whose commit timestamp is strictly BEFORE the start of the
simulation's EVAL window are considered. Commit timestamps are read from:
  - datasets/mozilla_perf/all_commits.jsonl

Risk definition:
  - For each revision, collect the set of signature-groups that appear in that
    revision's failing signature list (deduplicated within the revision).
  - A group's "repeat count" is the number of distinct revisions (after cutoff)
    in which it appears.
  - Convert counts into strictly-(0, 1) "probability-like" values:
      * Default (`--risk-method beta`): treat each historical revision as a
        trial and compute a Beta-Bernoulli posterior mean
          p = (count + alpha) / (n_revisions + alpha + beta).
        If `--prior-beta` is omitted, beta is auto-selected from the data so the
        long tail of zero-count groups is not driven to ~0.
      * Optional (`--risk-method max_norm`): normalize by the maximum count.

Output (JSONL):
  - datasets/mozilla_perf/historical_risk_per_signature_group.jsonl
    Each line: {"Sig_group_id": <int>, "historical_risk": <float>}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


# Allow extremely large fields in CSV
csv.field_size_limit(sys.maxsize)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "mozilla_perf")

DEFAULT_ALERT_CSV = os.path.join(DATASET_DIR, "alert_summary_fail_perf_sigs.csv")
DEFAULT_ALL_COMMITS_JSONL = os.path.join(DATASET_DIR, "all_commits.jsonl")
DEFAULT_SIG_GROUPS_JSONL = os.path.join(DATASET_DIR, "sig_groups.jsonl")
DEFAULT_OUTPUT_JSONL = os.path.join(
    DATASET_DIR, "historical_risk_per_signature_group.jsonl"
)

# Matches analysis/batch_testing/simulation.py default cutoff.
DEFAULT_EVAL_WINDOW_START_ISO = "2024-10-10T00:00:00+00:00"


def iter_jsonl(path: str) -> Iterable[Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_iso_datetime(value: str) -> datetime:
    """
    Parse a user-provided ISO-ish datetime.

    Accepts:
      - YYYY-MM-DD (interpreted as midnight UTC),
      - full ISO-8601 strings, with optional trailing 'Z'.
    """
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Empty datetime string")

    if len(raw) == 10 and raw.count("-") == 2:
        # YYYY-MM-DD
        return datetime.fromisoformat(raw + "T00:00:00+00:00")

    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        # Default to UTC if timezone is missing.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_hg_date(date_field: Any) -> datetime:
    """
    Parse the `date` field from the mozilla perf `all_commits.jsonl` dataset.

    Supports:
      - `[unix_ts, offset_seconds]` pairs (Mercurial-style), and
      - ISO-8601 datetime strings.
    """
    if isinstance(date_field, list) and len(date_field) == 2:
        unix_ts, offset_seconds_west = date_field
        # Mercurial stores tz offset as seconds west of UTC (positive values mean UTC-<hours>).
        # Python's timezone offset is seconds east of UTC, so we negate.
        tz = timezone(timedelta(seconds=-int(offset_seconds_west)))
        return datetime.fromtimestamp(float(unix_ts), tz=tz)
    if isinstance(date_field, str):
        dt = datetime.fromisoformat(date_field.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    raise TypeError(
        f"Unsupported hg date_field type {type(date_field)!r}; expected list[ts, offset] or ISO string."
    )


def load_revision_to_fail_sigs(alert_csv: str) -> dict[str, set[int]]:
    """
    Returns revision -> set(failing_signature_id).

    If multiple rows share the same revision, failing signatures are unioned so
    the revision is counted only once downstream.
    """
    revision_to_sigs: dict[str, set[int]] = defaultdict(set)
    with open(alert_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rev = (row.get("revision") or "").strip()
            if not rev:
                continue

            raw_sigs = (row.get("fail_perf_sig_ids") or "").strip()
            if not raw_sigs:
                continue

            try:
                sig_ids = json.loads(raw_sigs)
            except json.JSONDecodeError:
                # Fallback for non-JSON-ish list strings.
                try:
                    import ast

                    sig_ids = ast.literal_eval(raw_sigs)
                except Exception:
                    continue

            if not isinstance(sig_ids, list):
                continue

            for sig in sig_ids:
                try:
                    revision_to_sigs[rev].add(int(sig))
                except Exception:
                    continue

    return dict(revision_to_sigs)


def load_revision_timestamps(
    all_commits_jsonl: str, revisions: set[str]
) -> dict[str, datetime]:
    """
    Scan all_commits.jsonl once and return node -> timestamp for the requested revisions.
    """
    needed = set(revisions)
    found: dict[str, datetime] = {}
    with open(all_commits_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not needed:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            node = obj.get("node")
            if not node or node not in needed:
                continue
            date_field = obj.get("date")
            if date_field is None:
                continue
            try:
                ts = parse_hg_date(date_field)
            except Exception:
                continue
            found[node] = ts
            needed.remove(node)
    return found


def load_sig_group_maps(sig_groups_jsonl: str) -> tuple[list[int], dict[int, int]]:
    """
    Returns:
      - all_group_ids: sorted list of all Sig_group_id values
      - sig_to_group: signature_id -> Sig_group_id
    """
    sig_to_group: dict[int, int] = {}
    group_ids: list[int] = []
    for record in iter_jsonl(sig_groups_jsonl):
        if not isinstance(record, dict):
            continue
        group_id_raw = record.get("Sig_group_id")
        sigs = record.get("signatures")
        if group_id_raw is None or not isinstance(sigs, list):
            continue
        try:
            group_id = int(group_id_raw)
        except Exception:
            continue
        group_ids.append(group_id)
        for s in sigs:
            try:
                sig_id = int(s)
            except Exception:
                continue
            # If duplicates exist, keep the first mapping for determinism.
            sig_to_group.setdefault(sig_id, group_id)

    group_ids_sorted = sorted(set(group_ids))
    return group_ids_sorted, sig_to_group


def _commit_ids_from_predictions_json(path: str) -> set[str]:
    """
    Extract commit IDs from a model predictions JSON.

    Expected format (as used by analysis/batch_testing/simulation.py):
      {"samples": [{"commit_id": "...", ...}, ...], ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    commit_ids: set[str] = set()

    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        for s in payload["samples"]:
            if not isinstance(s, dict):
                continue
            cid = s.get("commit_id")
            if isinstance(cid, str) and cid:
                commit_ids.add(cid)
        return commit_ids

    # Fallback: treat dict keys as commit ids if they look like hashes.
    if isinstance(payload, dict):
        for k in payload.keys():
            if isinstance(k, str) and len(k) >= 12:
                commit_ids.add(k)
        return commit_ids

    raise ValueError(
        f"Unsupported predictions JSON format in {path}; expected dict with 'samples' list."
    )


def compute_eval_window_start_from_preds(
    eval_preds_json: str, all_commits_jsonl: str
) -> datetime | None:
    commit_ids = _commit_ids_from_predictions_json(eval_preds_json)
    if not commit_ids:
        return None
    ts_map = load_revision_timestamps(all_commits_jsonl, commit_ids)
    if not ts_map:
        return None
    return min(ts_map.values())


def compute_historical_risk(
    revision_to_sigs: dict[str, set[int]],
    revision_to_ts: dict[str, datetime],
    sig_to_group: dict[int, int],
    eval_window_start: datetime,
) -> Counter[int]:
    """
    Compute per-signature-group repeat counts over revisions before eval_window_start.
    """
    counts: Counter[int] = Counter()
    for rev, sig_ids in revision_to_sigs.items():
        ts = revision_to_ts.get(rev)
        if ts is None:
            continue
        if not (ts < eval_window_start):
            continue

        group_ids: set[int] = set()
        for sig_id in sig_ids:
            group_id = sig_to_group.get(sig_id)
            if group_id is not None:
                group_ids.add(group_id)
        for gid in group_ids:
            counts[gid] += 1

    return counts


def write_risk_jsonl(
    out_path: str,
    all_group_ids: list[int],
    group_counts: Counter[int],
    *,
    risk_method: str,
    epsilon: float,
    num_revisions_considered: int,
    prior_alpha: float,
    prior_beta: float | None,
) -> None:
    max_count = max(group_counts.values(), default=0)
    if not (0.0 < float(epsilon) < 0.5):
        raise ValueError(f"--epsilon must be in (0, 0.5); got {epsilon}")

    method = str(risk_method or "").strip().lower()
    if method not in {"beta", "max_norm"}:
        raise ValueError(f"--risk-method must be one of: beta, max_norm (got {risk_method!r})")

    n = int(num_revisions_considered)
    if n < 0:
        raise ValueError(f"num_revisions_considered must be >= 0 (got {n})")

    alpha = float(prior_alpha)
    if alpha <= 0.0:
        raise ValueError(f"--prior-alpha must be > 0 (got {prior_alpha})")

    beta = None if prior_beta is None else float(prior_beta)
    if beta is not None and beta <= 0.0:
        raise ValueError(f"--prior-beta must be > 0 (got {prior_beta})")

    # Auto-select beta for the beta method so that:
    #   sum_g risk_g == (total failing group occurrences) / (num revisions)
    # which keeps the tail non-trivial while keeping risk values probability-like.
    if method == "beta" and beta is None:
        total_occ = int(sum(int(v) for v in group_counts.values()))
        g = int(len(all_group_ids))
        if n > 0 and total_occ > 0 and g > 0:
            denom = (total_occ + alpha * g) * (n / float(total_occ))
            beta = denom - n - alpha
        else:
            # Degenerate case: no history. Fall back to a conservative prior.
            beta = 1000.0

        if beta <= 0.0:
            # Extremely small histories can produce non-positive beta; clamp.
            beta = 1.0

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for gid in all_group_ids:
            count = int(group_counts.get(gid, 0))
            if method == "max_norm":
                if max_count > 0:
                    risk = epsilon + (1.0 - 2.0 * epsilon) * (count / float(max_count))
                else:
                    # No historical failures found; assign all groups the lowest risk.
                    risk = float(epsilon)
            else:
                # Beta-Bernoulli posterior mean with a strong "rare failure" prior:
                #   risk = E[p | count successes in n trials]
                # The trial definition here is "a revision in the historical window".
                assert beta is not None
                denom = n + alpha + beta
                if denom <= 0:
                    risk = float(epsilon)
                else:
                    risk = (count + alpha) / float(denom)

            # Guardrails: enforce (0, 1) strictly.
            if risk <= 0.0:
                risk = float(epsilon)
            if risk >= 1.0:
                risk = 1.0 - float(epsilon)

            f.write(
                json.dumps(
                    {"Sig_group_id": int(gid), "historical_risk": float(risk)},
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalized historical risk scores per perf signature-group, "
            "using only revisions before the simulation EVAL window start."
        )
    )
    parser.add_argument(
        "--alert-csv",
        default=DEFAULT_ALERT_CSV,
        help="Path to alert_summary_fail_perf_sigs.csv.",
    )
    parser.add_argument(
        "--all-commits-jsonl",
        default=DEFAULT_ALL_COMMITS_JSONL,
        help="Path to all_commits.jsonl.",
    )
    parser.add_argument(
        "--sig-groups-jsonl",
        default=DEFAULT_SIG_GROUPS_JSONL,
        help="Path to sig_groups.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_JSONL,
        help="Output JSONL path for historical risk per signature-group.",
    )
    parser.add_argument(
        "--eval-window-start",
        default=DEFAULT_EVAL_WINDOW_START_ISO,
        help=(
            "Start timestamp (ISO-8601) of the simulation EVAL window. "
            "Revisions with commit time >= this are excluded from history."
        ),
    )
    parser.add_argument(
        "--eval-preds-json",
        default=None,
        help=(
            "Optional: path to the EVAL predictions JSON used by the simulator. "
            "If provided, the EVAL window start is computed as the oldest commit "
            "timestamp present in that predictions file (falling back to --eval-window-start)."
        ),
    )
    parser.add_argument(
        "--risk-method",
        choices=["beta", "max_norm"],
        default="beta",
        help=(
            "How to convert historical counts into probabilities. "
            "`beta` (default) produces probability-like values using a Beta prior; "
            "`max_norm` reproduces the old max-count normalization."
        ),
    )
    parser.add_argument(
        "--prior-alpha",
        type=float,
        default=1.0,
        help=(
            "Beta prior alpha (>0) used for `--risk-method beta`. "
            "Larger values raise the floor for unseen groups and reduce skew."
        ),
    )
    parser.add_argument(
        "--prior-beta",
        type=float,
        default=None,
        help=(
            "Beta prior beta (>0) used for `--risk-method beta`. "
            "If omitted, the script auto-selects beta from the data to keep the "
            "tail non-trivial while keeping values probability-like."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help=(
            "Small offset ensuring historical_risk is strictly in (0, 1). "
            "Groups with zero count receive epsilon; max-count groups receive 1-epsilon."
        ),
    )
    args = parser.parse_args()

    for p in [args.alert_csv, args.all_commits_jsonl, args.sig_groups_jsonl]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input: {p}")

    eval_window_start = _parse_iso_datetime(args.eval_window_start)
    if args.eval_preds_json:
        if not os.path.exists(args.eval_preds_json):
            raise FileNotFoundError(f"--eval-preds-json not found: {args.eval_preds_json}")
        dynamic = compute_eval_window_start_from_preds(
            args.eval_preds_json, args.all_commits_jsonl
        )
        if dynamic is not None:
            eval_window_start = dynamic

    revision_to_sigs = load_revision_to_fail_sigs(args.alert_csv)
    revisions = set(revision_to_sigs.keys())
    revision_to_ts = load_revision_timestamps(args.all_commits_jsonl, revisions)

    all_group_ids, sig_to_group = load_sig_group_maps(args.sig_groups_jsonl)
    group_counts = compute_historical_risk(
        revision_to_sigs, revision_to_ts, sig_to_group, eval_window_start
    )

    considered_revs = [
        rev
        for rev in revisions
        if (rev in revision_to_ts and revision_to_ts[rev] < eval_window_start)
    ]

    total_occ = int(sum(int(v) for v in group_counts.values()))
    used_beta = args.prior_beta
    if str(args.risk_method).strip().lower() == "beta" and used_beta is None:
        n = len(considered_revs)
        g = len(all_group_ids)
        alpha = float(args.prior_alpha)
        if n > 0 and total_occ > 0 and g > 0:
            denom = (total_occ + alpha * g) * (n / float(total_occ))
            used_beta = denom - n - alpha
        else:
            used_beta = 1000.0
        if used_beta <= 0.0:
            used_beta = 1.0

    write_risk_jsonl(
        args.output,
        all_group_ids,
        group_counts,
        risk_method=args.risk_method,
        epsilon=args.epsilon,
        num_revisions_considered=len(considered_revs),
        prior_alpha=args.prior_alpha,
        prior_beta=used_beta,
    )

    missing_revs = len(revisions) - len(revision_to_ts)
    print(f"eval_window_start: {eval_window_start.isoformat()}")
    print(f"alert_summary revisions (unique): {len(revisions)}")
    print(f"revisions missing timestamp in all_commits.jsonl: {missing_revs}")
    print(f"revisions considered for history (ts < eval_window_start): {len(considered_revs)}")
    print(f"sig_groups loaded: {len(all_group_ids)}")
    print(f"signature->group mappings: {len(sig_to_group)}")
    print(
        f"groups with nonzero historical count: "
        f"{len([g for g in all_group_ids if group_counts.get(g, 0) > 0])}"
    )
    print(f"total failing group occurrences (dedup per revision): {total_occ}")
    print(
        f"risk_method: {args.risk_method} (alpha={args.prior_alpha}, beta={used_beta})"
    )
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
