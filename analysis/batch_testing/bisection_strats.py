"""
Bisection strategies and shared execution/time model for the batch-testing simulator.

This module provides:
  - Perf metadata loading utilities (signature-group IDs, durations, failing signatures).
  - A central test-capacity simulator (`TestExecutor` + `run_test_suite`).
  - Bisection strategies that operate on batches (TOB, PAR, RWAB, RWAB-LS, TKRB, SWB, SWF).

See `analysis/batch_testing/README.md` for a detailed conceptual overview.
"""

from datetime import timedelta
import os
import csv
import json
import logging
import random
import heapq
import math

# ---------- Perf metadata loading ----------

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SIG_GROUP_JOB_DURATIONS_CSV = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "sig_group_job_durations.csv"
)
ALERT_FAIL_SIGS_CSV = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "alert_summary_fail_perf_sigs.csv"
)
PERF_JOBS_PER_REV_JSON = os.path.join(
    REPO_ROOT,
    "datasets",
    "mozilla_perf",
    "perf_jobs_per_revision_details_rectified.jsonl",
)
SIG_GROUPS_JSONL = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "sig_groups.jsonl"
)
ALL_SIGNATURES_JSONL = os.path.join(
    REPO_ROOT, "datasets", "mozilla_perf", "all_signatures.jsonl"
)

# ---------- Central test executor capacity (per-platform worker pools) ----------

# Default pool sizes (override by passing a custom dict to TestExecutor).
ANDROID_WORKERS = 60
WINDOWS_WORKERS = 120
LINUX_WORKERS = 100
MAC_WORKERS = 250

DEFAULT_WORKER_POOLS = {
    "android": ANDROID_WORKERS,
    "windows": WINDOWS_WORKERS,
    "linux": LINUX_WORKERS,
    "mac": MAC_WORKERS,
}

# Fallbacks and knobs (configured by the main simulation script).
_default_test_duration_min = 20.0
# -1 means "use all available signature-groups" (no downsampling).
_full_suite_signatures_per_run = -1
# Constant build-time overhead applied once per suite run (minutes).
_build_time_minutes = 90.0  # 1.5 hours
# When a signature-group cannot be mapped to a platform (or platform metadata is
# missing/unrecognized), route it to this pool key. The default is "mac", but
# can be overridden via `configure_bisection_defaults()`.
DEFAULT_UNKNOWN_PLATFORM_POOL = "mac"
_unknown_platform_pool = DEFAULT_UNKNOWN_PLATFORM_POOL

# signature_id -> signature_group_id
SIG_TO_GROUP_ID = {}
# signature_group_id -> list[int signature_id]
SIG_GROUP_TO_SIG_IDS = {}
# signature_id -> pool key (android/windows/linux/mac). iOS signatures are routed to "mac".
SIG_ID_TO_POOL = {}
# signature_group_id -> pool key (cached)
SIG_GROUP_ID_TO_POOL = {}
# signature_group_id -> duration_minutes
SIG_GROUP_DURATIONS = {}
# revision (commit_id) -> list[int signature_id] (raw failing signatures)
REVISION_FAIL_SIG_IDS = {}
# revision (commit_id) -> list[int signature_group_id] actually tested on that revision
REVISION_TESTED_SIG_GROUP_IDS = {}
# list[int] of signature_group_ids for the "full" batch test suite
BATCH_SIG_GROUP_IDS = []

_warned_missing_sig_group_duration_ids = set()
_warned_missing_sig_group_pool_ids = set()
_warned_unknown_machine_platform_values = set()

TKRB_TOP_K = 1

logger = logging.getLogger(__name__)


def configure_bisection_defaults(
    default_test_duration_min=None,
    full_suite_signatures_per_run=None,
    build_time_minutes=None,
    unknown_platform_pool=None,
):
    """
    Configure global defaults used by the simulator's time/cost model.

    Parameters
    ----------
    default_test_duration_min:
        Fallback duration (minutes) used when a signature-group ID has no
        recorded duration in `sig_group_job_durations.csv`.
    full_suite_signatures_per_run:
        Caps the number of signature-groups used for a "full suite" run:
          - `-1` => use all available signature-groups
          - `N>0` => randomly sample N signature-groups (models partial suite runs)
    build_time_minutes:
        Constant build-time overhead (minutes) added once per suite run (both
        batch root runs and bisection-step runs).
    unknown_platform_pool:
        Pool key to route signature-groups/jobs to when platform routing cannot
        be determined (missing signature-group id, missing signature metadata,
        or unrecognized `machine_platform`).

    These knobs are typically set by `simulation.py` so all strategies share
    the same configuration.
    """
    global _default_test_duration_min, _full_suite_signatures_per_run, _build_time_minutes, _unknown_platform_pool

    if default_test_duration_min is not None:
        _default_test_duration_min = float(default_test_duration_min)

    if full_suite_signatures_per_run is not None:
        val = int(full_suite_signatures_per_run)
        if val < 0:
            # -1 => use all signature-groups (no cap)
            _full_suite_signatures_per_run = -1
        elif val == 0:
            raise ValueError(
                "full_suite_signatures_per_run must be a positive integer "
                "or -1 to indicate 'use all signature-groups'; got 0."
            )
        else:
            _full_suite_signatures_per_run = val

    if build_time_minutes is not None:
        val = float(build_time_minutes)
        if val < 0:
            raise ValueError(
                f"build_time_minutes must be non-negative; got {build_time_minutes!r}"
            )
        _build_time_minutes = val

    if unknown_platform_pool is not None:
        pool = str(unknown_platform_pool).strip()
        if not pool:
            raise ValueError(
                f"unknown_platform_pool must be a non-empty string; got {unknown_platform_pool!r}"
            )
        _unknown_platform_pool = pool.lower()

    logger.info(
        "Configured bisection defaults: default_test_duration_min=%.2f, "
        "full_suite_signatures_per_run=%s, build_time_minutes=%.2f, "
        "unknown_platform_pool=%s",
        _default_test_duration_min,
        str(_full_suite_signatures_per_run),
        float(_build_time_minutes),
        str(_unknown_platform_pool),
    )


def configure_full_suite_signatures_union(revisions):
    """
    Given an iterable of revision ids that fall within the simulation's
    cutoff windows, compute the union of all perf signature-groups that were
    actually tested on at least one of those revisions and update the
    full-suite batch durations accordingly.

    This is used when we want each initial batch test run to execute all
    tests that appear at least once within the cutoff window, instead of
    all signature-groups from sig_group_job_durations.csv.

    This function mutates the module-level `BATCH_SIG_GROUP_IDS`, which
    is consumed by `get_batch_signature_durations()`.
    """
    global BATCH_SIG_GROUP_IDS

    _load_perf_metadata()
    _load_perf_jobs_per_revision()

    rev_set = set(revisions or [])
    if not rev_set:
        raise RuntimeError(
            "configure_full_suite_signatures_union: empty revision set; "
            "cannot construct a full-suite signature union."
        )

    sig_group_ids = set()
    for rev in rev_set:
        for sig_group in REVISION_TESTED_SIG_GROUP_IDS.get(rev, []):
            try:
                sig_group_ids.add(int(sig_group))
            except (TypeError, ValueError):
                continue

    if not sig_group_ids:
        raise RuntimeError(
            "configure_full_suite_signatures_union: no tested signature-groups "
            "found for the provided revisions; cannot build full suite."
        )

    # Use the union of signature-group IDs as the "full suite" definition.
    # Durations are resolved lazily at execution time (with default fallback)
    # to ensure missing sig_group_job_durations.csv entries are handled
    # consistently via `_get_sig_group_duration_minutes`.
    BATCH_SIG_GROUP_IDS = sorted(sig_group_ids)


def _load_perf_metadata():
    """
    Load:
      - sig_group_job_durations.csv => SIG_GROUP_DURATIONS, BATCH_SIG_GROUP_IDS
      - alert_summary_fail_perf_sigs.csv => REVISION_FAIL_SIG_IDS
    """
    global SIG_GROUP_DURATIONS, REVISION_FAIL_SIG_IDS, BATCH_SIG_GROUP_IDS

    if SIG_GROUP_DURATIONS and BATCH_SIG_GROUP_IDS and REVISION_FAIL_SIG_IDS:
        # Already loaded
        logger.debug("Perf metadata already loaded; skipping reload.")
        return

    # ----- sig_group_job_durations.csv -----
    sig_group_durations = {}
    try:
        logger.info("Loading signature-group job durations from %s", SIG_GROUP_JOB_DURATIONS_CSV)
        with open(SIG_GROUP_JOB_DURATIONS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sig_group = row.get("signature_group_id")
                dur = row.get("duration_minutes")
                if not sig_group or not dur:
                    continue
                try:
                    sig_group_id = int(sig_group)
                    duration = float(dur)
                except ValueError:
                    continue
                sig_group_durations[sig_group_id] = duration
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"sig_group_job_durations.csv not found at {SIG_GROUP_JOB_DURATIONS_CSV}"
        ) from exc

    SIG_GROUP_DURATIONS = sig_group_durations
    if not SIG_GROUP_DURATIONS:
        raise RuntimeError(
            f"No valid signature-group duration rows loaded from {SIG_GROUP_JOB_DURATIONS_CSV}"
        )
    BATCH_SIG_GROUP_IDS = sorted(SIG_GROUP_DURATIONS.keys())
    logger.info(
        "Loaded %d signature-group durations; BATCH_SIG_GROUP_IDS length=%d",
        len(SIG_GROUP_DURATIONS),
        len(BATCH_SIG_GROUP_IDS),
    )

    # ----- alert_summary_fail_perf_sigs.csv -----
    rev_fail = {}
    try:
        logger.info("Loading failing perf signatures from %s", ALERT_FAIL_SIGS_CSV)
        with open(ALERT_FAIL_SIGS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rev = row.get("revision")
                raw = row.get("fail_perf_sig_ids") or ""
                if not rev or not raw:
                    continue

                sig_ids = []
                # Example: "[5094909, 5095143, 5095203]"
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        sig_ids = [int(x) for x in parsed]
                except json.JSONDecodeError:
                    raw_stripped = raw.strip().strip("[]")
                    if raw_stripped:
                        parts = [p.strip() for p in raw_stripped.split(",") if p.strip()]
                        sig_ids = []
                        for p in parts:
                            try:
                                sig_ids.append(int(p))
                            except ValueError:
                                continue

                rev_fail[rev] = sig_ids
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"alert_summary_fail_perf_sigs.csv not found at {ALERT_FAIL_SIGS_CSV}"
        ) from exc

    REVISION_FAIL_SIG_IDS = rev_fail
    if not REVISION_FAIL_SIG_IDS:
        raise RuntimeError(
            f"No failing perf signatures loaded from {ALERT_FAIL_SIGS_CSV}"
        )
    logger.info(
        "Loaded failing signature mapping for %d revisions",
        len(REVISION_FAIL_SIG_IDS),
    )


def _load_perf_jobs_per_revision():
    """
    Load:
      - perf_jobs_per_revision_details_rectified.jsonl => REVISION_TESTED_SIG_GROUP_IDS

    The JSON file can be either:
      * a JSON-lines file (one JSON object per line), or
      * a single JSON list of objects.
    Each object is expected to have:
      - 'revision': str
      - 'signature_group_ids': list[int]
    """
    global REVISION_TESTED_SIG_GROUP_IDS

    if REVISION_TESTED_SIG_GROUP_IDS:
        return

    mapping = {}
    try:
        logger.info(
            "Loading tested perf signature-groups per revision from %s (JSONL)",
            PERF_JOBS_PER_REV_JSON,
        )
        with open(PERF_JOBS_PER_REV_JSON, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSONL line in %s: %s",
                        PERF_JOBS_PER_REV_JSON,
                        line[:200],
                    )
                    continue
                if not isinstance(obj, dict):
                    continue
                rev = obj.get("revision")
                sig_group_ids = obj.get("signature_group_ids") or []
                if not rev:
                    continue
                try:
                    sig_group_ids_int = [int(s) for s in sig_group_ids]
                except (TypeError, ValueError):
                    sig_group_ids_int = []
                mapping[rev] = sig_group_ids_int
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "perf_jobs_per_revision_details_rectified.jsonl not found at "
            f"{PERF_JOBS_PER_REV_JSON}"
        ) from exc

    REVISION_TESTED_SIG_GROUP_IDS = mapping
    if not REVISION_TESTED_SIG_GROUP_IDS:
        raise RuntimeError(
            f"No tested perf signature-groups loaded from {PERF_JOBS_PER_REV_JSON}"
        )
    logger.info(
        "Loaded tested signature-groups for %d revisions",
        len(REVISION_TESTED_SIG_GROUP_IDS),
    )

def _load_sig_groups_mapping():
    """
    Load:
      - sig_groups.jsonl => SIG_TO_GROUP_ID, SIG_GROUP_TO_SIG_IDS

    Expected JSONL rows:
      {"Sig_group_id": 1040, "signatures": [72111, 72112]}
    """
    global SIG_TO_GROUP_ID, SIG_GROUP_TO_SIG_IDS

    if SIG_TO_GROUP_ID and SIG_GROUP_TO_SIG_IDS:
        return

    sig_to_group = {}
    group_to_sigs = {}
    try:
        logger.info("Loading signature->signature-group mapping from %s (JSONL)", SIG_GROUPS_JSONL)
        with open(SIG_GROUPS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSONL line in %s: %s",
                        SIG_GROUPS_JSONL,
                        line[:200],
                    )
                    continue
                if not isinstance(obj, dict):
                    continue
                group_id = obj.get("Sig_group_id")
                signatures = obj.get("signatures") or []
                try:
                    group_id_int = int(group_id)
                except (TypeError, ValueError):
                    continue
                if not isinstance(signatures, list):
                    continue
                group_sigs = group_to_sigs.setdefault(group_id_int, [])
                for sig in signatures:
                    try:
                        sig_id_int = int(sig)
                    except (TypeError, ValueError):
                        continue
                    prev = sig_to_group.get(sig_id_int)
                    if prev is not None and prev != group_id_int:
                        raise ValueError(
                            f"Signature {sig_id_int} maps to multiple groups ({prev}, {group_id_int}) "
                            f"in {SIG_GROUPS_JSONL}"
                        )
                    sig_to_group[sig_id_int] = group_id_int
                    group_sigs.append(sig_id_int)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"sig_groups.jsonl not found at {SIG_GROUPS_JSONL}"
        ) from exc

    # Normalize group signature lists (unique + sorted for determinism).
    normalized_group_to_sigs = {}
    for gid, sigs in group_to_sigs.items():
        if not sigs:
            continue
        normalized_group_to_sigs[int(gid)] = sorted(set(sigs))

    SIG_TO_GROUP_ID = sig_to_group
    SIG_GROUP_TO_SIG_IDS = normalized_group_to_sigs
    if not SIG_TO_GROUP_ID or not SIG_GROUP_TO_SIG_IDS:
        raise RuntimeError(
            f"No valid signature->group mappings loaded from {SIG_GROUPS_JSONL}"
        )
    logger.info(
        "Loaded %d signature->group mappings across %d signature-groups",
        len(SIG_TO_GROUP_ID),
        len(SIG_GROUP_TO_SIG_IDS),
    )

def _pool_from_machine_platform(machine_platform: str):
    """
    Convert a `machine_platform` string into an executor pool key.

    Expected values contain substrings like:
      - linux
      - android
      - ac-ui-test (treated as android)
      - osx (e.g. "macosx")
      - ios (routed to mac pool)
      - windows

    """
    if not machine_platform:
        return None
    s = str(machine_platform).strip().lower()
    if not s:
        return None
    # Some Android Components tasks report a non-standard platform string.
    # Route these to the android worker pool.
    if "ac-ui-test" in s:
        return "android"
    if "android" in s:
        return "android"
    if "ios" in s:
        return "mac"
    if "osx" in s or "mac" in s or "darwin" in s:
        return "mac"
    if "win" in s or "windows" in s:
        return "windows"
    if "linux" in s:
        return "linux"
    return None


def _load_signature_platforms():
    """
    Load:
      - all_signatures.jsonl => SIG_ID_TO_POOL

    Each JSONL row is expected to include:
      - "id": int (signature id)
      - "machine_platform": str

    If the platform value is unrecognized, we log a warning (once per distinct
    platform string) and default that signature to the configured fallback pool
    (`_unknown_platform_pool`).
    """
    global SIG_ID_TO_POOL

    if SIG_ID_TO_POOL:
        return

    mapping = {}
    try:
        logger.info("Loading signature platform metadata from %s (JSONL)", ALL_SIGNATURES_JSONL)
        with open(ALL_SIGNATURES_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSONL line in %s: %s",
                        ALL_SIGNATURES_JSONL,
                        line[:200],
                    )
                    continue
                if not isinstance(obj, dict):
                    continue
                sig_id = obj.get("id")
                machine_platform = obj.get("machine_platform")
                try:
                    sig_id_int = int(sig_id)
                except (TypeError, ValueError):
                    continue
                pool = _pool_from_machine_platform(machine_platform)
                if pool is None:
                    mp = str(machine_platform) if machine_platform is not None else ""
                    if mp and mp not in _warned_unknown_machine_platform_values:
                        _warned_unknown_machine_platform_values.add(mp)
                        logger.warning(
                            "Unrecognized machine_platform value %r in %s; defaulting to %s pool.",
                            mp,
                            ALL_SIGNATURES_JSONL,
                            _unknown_platform_pool,
                        )
                    pool = _unknown_platform_pool
                mapping[sig_id_int] = pool
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"all_signatures.jsonl not found at {ALL_SIGNATURES_JSONL}"
        ) from exc

    SIG_ID_TO_POOL = mapping
    if not SIG_ID_TO_POOL:
        raise RuntimeError(
            f"No valid signature platform rows loaded from {ALL_SIGNATURES_JSONL}"
        )
    logger.info("Loaded platform metadata for %d signatures", len(SIG_ID_TO_POOL))


def _get_worker_pool_for_signature_group(sig_group_id):
    """
    Determine the executor pool key for a signature-group.

    Rule:
      - Pick a representative *random* signature from the group (deterministic
        per group id) and use its machine_platform to choose the pool.
    """
    if sig_group_id is None:
        return _unknown_platform_pool

    try:
        gid = int(sig_group_id)
    except (TypeError, ValueError):
        return _unknown_platform_pool

    cached = SIG_GROUP_ID_TO_POOL.get(gid)
    if cached is not None:
        return cached

    _load_sig_groups_mapping()
    _load_signature_platforms()

    sig_ids = SIG_GROUP_TO_SIG_IDS.get(gid) or []
    if not sig_ids:
        if gid not in _warned_missing_sig_group_pool_ids:
            _warned_missing_sig_group_pool_ids.add(gid)
            logger.warning(
                "No signatures found for signature_group_id=%s in %s; routing to default pool.",
                str(gid),
                SIG_GROUPS_JSONL,
            )
        pool = _unknown_platform_pool
        SIG_GROUP_ID_TO_POOL[gid] = pool
        return pool

    # Deterministic per-group "random" selection order to avoid coupling pool
    # assignment to global RNG state across Optuna trials.
    rng = random.Random(gid)
    shuffled = list(sig_ids)
    rng.shuffle(shuffled)

    pool = None
    for sig in shuffled:
        pool = SIG_ID_TO_POOL.get(int(sig))
        if pool is not None:
            break

    if pool is None:
        if gid not in _warned_missing_sig_group_pool_ids:
            _warned_missing_sig_group_pool_ids.add(gid)
            logger.warning(
                "No signature in signature_group_id=%s could be mapped to a platform via %s; "
                "routing to default pool.",
                str(gid),
                ALL_SIGNATURES_JSONL,
            )
        pool = _unknown_platform_pool

    SIG_GROUP_ID_TO_POOL[gid] = pool
    return pool


def _get_sig_group_duration_minutes(sig_group_id: int) -> float:
    """
    Map a signature-group ID to its duration (in minutes).

    If the group is missing from sig_group_job_durations.csv, log a warning
    (once per group id) and fall back to `_default_test_duration_min`.
    """
    _load_perf_metadata()
    try:
        gid = int(sig_group_id)
    except (TypeError, ValueError):
        return float(_default_test_duration_min)

    dur = SIG_GROUP_DURATIONS.get(gid)
    if dur is None:
        if gid not in _warned_missing_sig_group_duration_ids:
            _warned_missing_sig_group_duration_ids.add(gid)
            logger.warning(
                "Missing duration for signature_group_id=%s in %s; using default %.2f min",
                str(gid),
                SIG_GROUP_JOB_DURATIONS_CSV,
                float(_default_test_duration_min),
            )
        return float(_default_test_duration_min)

    return float(dur)


def validate_failing_signatures_coverage(failing_revisions=None):
    """
    Ensure that failing perf signature-groups are covered by the
    perf_jobs_per_revision_details_rectified.jsonl dataset, restricted to a
    specific set of failing revisions.

    Args:
        failing_revisions: iterable of revision ids that are considered
            buggy and must be present in alert_summary_fail_perf_sigs.csv.

    Raises:
        RuntimeError:
            - if failing_revisions is None or empty,
            - if any of the failing_revisions is missing from
              alert_summary_fail_perf_sigs.csv,
            - if any failing signature cannot be mapped to a signature-group
              via sig_groups.jsonl,
            - or if any relevant failing signature-group is not covered by
              the perf_jobs_per_revision_details_rectified.jsonl dataset.
    """
    _load_perf_metadata()
    _load_sig_groups_mapping()
    _load_perf_jobs_per_revision()

    if not REVISION_FAIL_SIG_IDS:
        # Nothing to validate; either there are no failing signatures
        # or the CSV is missing (already logged in _load_perf_metadata).
        logger.warning(
            "No failing perf signatures loaded from %s; "
            "skipping coverage validation.",
            ALERT_FAIL_SIGS_CSV,
        )
        return

    if not REVISION_TESTED_SIG_GROUP_IDS:
        raise RuntimeError(
            "perf_jobs_per_revision_details_rectified.jsonl did not yield any "
            "tested signature-group data. Cannot validate that failing perf "
            "signature-groups are covered. Please regenerate the dataset at: "
            f"{PERF_JOBS_PER_REV_JSON}"
        )

    if failing_revisions is None:
        raise RuntimeError(
            "failing_revisions must be provided to "
            "validate_failing_signatures_coverage; got None."
        )

    failing_revisions = set(failing_revisions)
    if not failing_revisions:
        raise RuntimeError(
            "failing_revisions is empty; cannot validate perf "
            "signature-group coverage without any failing revisions."
        )

    # Ensure every failing revision is present in the alert CSV.
    alert_revisions = set(REVISION_FAIL_SIG_IDS.keys())
    missing_revisions = sorted(failing_revisions - alert_revisions)
    if missing_revisions:
        sample = ", ".join(missing_revisions[:20])
        extra = (
            "" if len(missing_revisions) <= 20 else f" (and {len(missing_revisions) - 20} more...)"
        )
        raise RuntimeError(
            "Some failing revisions have no failing perf signature "
            "entries in alert_summary_fail_perf_sigs.csv. Every failing "
            "revision must be present in that file.\n"
            f"First missing revisions: {sample}{extra}\n"
            f"Alert CSV: {ALERT_FAIL_SIGS_CSV}"
        )

    # At this point all failing_revisions are present in the alert CSV.
    relevant_revisions = failing_revisions

    # Collect the set of failing signature IDs for the selected revisions.
    failing_sig_ids = set()
    for rev in relevant_revisions:
        for sig in REVISION_FAIL_SIG_IDS.get(rev, []):
            try:
                failing_sig_ids.add(int(sig))
            except (TypeError, ValueError):
                continue

    # Map failing signatures -> failing signature-groups (must be possible).
    failing_sig_ids = set()
    for rev in failing_revisions:
        for sig in REVISION_FAIL_SIG_IDS.get(rev, []):
            try:
                failing_sig_ids.add(int(sig))
            except (TypeError, ValueError):
                continue

    failing_sig_group_ids = set()
    missing_sig_to_group = sorted(sig for sig in failing_sig_ids if sig not in SIG_TO_GROUP_ID)
    if missing_sig_to_group:
        sample = ", ".join(str(sig) for sig in missing_sig_to_group[:20])
        extra = "" if len(missing_sig_to_group) <= 20 else f" (and {len(missing_sig_to_group) - 20} more...)"
        raise RuntimeError(
            "Found failing perf signatures that are missing a sig_groups.jsonl mapping "
            "to a signature-group. This means the simulation cannot translate failures "
            "into signature-group jobs for bisection.\n"
            f"First missing signature_ids: {sample}{extra}\n"
            f"sig_groups JSONL: {SIG_GROUPS_JSONL}\n"
            f"Alert CSV: {ALERT_FAIL_SIGS_CSV}"
        )

    for sig in failing_sig_ids:
        failing_sig_group_ids.add(SIG_TO_GROUP_ID[int(sig)])

    # Collect the set of all signature-group IDs that appear anywhere in the
    # perf_jobs_per_revision_details_rectified.jsonl dataset.
    tested_sig_group_ids = set()
    for group_ids in REVISION_TESTED_SIG_GROUP_IDS.values():
        for gid in group_ids:
            try:
                tested_sig_group_ids.add(int(gid))
            except (TypeError, ValueError):
                continue

    missing_groups = sorted(failing_sig_group_ids - tested_sig_group_ids)
    if missing_groups:
        sample = ", ".join(str(gid) for gid in missing_groups[:20])
        extra = "" if len(missing_groups) <= 20 else f" (and {len(missing_groups) - 20} more...)"
        raise RuntimeError(
            "Found failing perf signature-groups that are not covered by "
            "perf_jobs_per_revision_details_rectified.jsonl at all. "
            "Each relevant failing signature-group (for the revisions under "
            "consideration) should appear at least once somewhere in the "
            "perf jobs dataset. This means the simulation cannot "
            "exercise all required failing signature-groups.\n"
            f"First missing signature_group_ids: {sample}{extra}\n"
            f"Alert CSV: {ALERT_FAIL_SIGS_CSV}\n"
            f"sig_groups JSONL: {SIG_GROUPS_JSONL}\n"
            f"Perf jobs JSONL: {PERF_JOBS_PER_REV_JSON}"
        )


def get_batch_signature_durations():
    """
    Suite for a "full suite" perf run (signature-groups).

    By default this is all signature-groups from `sig_group_job_durations.csv`,
    but it can be
    capped to a fixed-size random subset via `_full_suite_signatures_per_run`.

    This suite is used for the *first* run of a batch when bisection strategies
    are invoked with `is_batch_root=True`.
    """
    _load_perf_metadata()

    if not BATCH_SIG_GROUP_IDS:
        return [(None, float(_default_test_duration_min))]

    limit = _full_suite_signatures_per_run
    # Non-negative limit => cap via random subset when smaller than the
    # available suite size. Negative (e.g., -1) means "use all".
    if isinstance(limit, int) and limit >= 0 and len(BATCH_SIG_GROUP_IDS) > limit:
        sig_group_ids = random.sample(BATCH_SIG_GROUP_IDS, limit)
    else:
        sig_group_ids = BATCH_SIG_GROUP_IDS

    return [(gid, _get_sig_group_duration_minutes(gid)) for gid in sig_group_ids]


def get_failing_signature_durations_for_batch(batch_sorted):
    """
    Suite for the failing signature-groups revealed by the initial
    full batch run for this batch.

    We take the union of failing perf signatures for all *regressor* commits
    in this batch (true_label == True) using alert_summary_fail_perf_sigs.csv,
    map them to signature-groups via sig_groups.jsonl, then map to durations
    via sig_group_job_durations.csv (with a default fallback).

    This matches the workflow:
      - First full batch run fails on some signature-groups.
      - All bisection steps then only run those failing signature-groups.
    """
    _load_perf_metadata()
    _load_sig_groups_mapping()

    has_regressor = any(c.get("true_label") for c in (batch_sorted or []))
    sig_group_ids = set()

    for c in batch_sorted:
        if not c.get("true_label"):
            continue
        cid = c["commit_id"]
        for sig in REVISION_FAIL_SIG_IDS.get(cid, []):
            try:
                sig_id_int = int(sig)
            except (TypeError, ValueError):
                continue
            gid = SIG_TO_GROUP_ID.get(sig_id_int)
            if gid is None:
                continue
            sig_group_ids.add(int(gid))

    if has_regressor and not sig_group_ids:
        logger.warning(
            "No failing signature-groups could be derived for a batch containing regressors; "
            "falling back to default duration %.2f min for bisection steps.",
            float(_default_test_duration_min),
        )
        return [(None, float(_default_test_duration_min))]

    durations = [(gid, _get_sig_group_duration_minutes(gid)) for gid in sorted(sig_group_ids)]
    logger.debug(
        "Computed failing signature-group durations for batch of size %d: %d signature-groups",
        len(batch_sorted),
        len(durations),
    )
    return durations


def get_tested_signatures_for_revision(revision):
    """
    Return the list of signature-group IDs that were actually tested for the given
    revision according to perf_jobs_per_revision_details_rectified.jsonl.
    """
    _load_perf_jobs_per_revision()
    return REVISION_TESTED_SIG_GROUP_IDS.get(revision, [])


def get_signature_durations_for_ids(signature_ids):
    """
    Map a collection of signature-group IDs to a suite of (signature_group_id, duration_minutes)
    entries using
    sig_group_job_durations.csv. For any unknown signature-group, we fall back to
    _default_test_duration_min (and log a warning once per missing group id).
    """
    _load_perf_metadata()
    suite = []
    for sig in signature_ids:
        try:
            sig_group_id = int(sig)
        except (TypeError, ValueError):
            continue
        dur = _get_sig_group_duration_minutes(sig_group_id)
        suite.append((sig_group_id, dur))
    if not suite:
        suite = [(None, float(_default_test_duration_min))]
    return suite


def get_failing_signatures_for_revision(revision):
    """
    Return the list of failing signature IDs for a given revision according to
    alert_summary_fail_perf_sigs.csv.
    """
    _load_perf_metadata()
    return REVISION_FAIL_SIG_IDS.get(revision, [])


def get_signature_group_id_for_signature(signature_id):
    """
    Map a single perf signature ID to its signature-group ID via sig_groups.jsonl.

    Returns:
        int signature_group_id, or None if the signature is unmapped.
    """
    _load_sig_groups_mapping()
    try:
        sig_id_int = int(signature_id)
    except (TypeError, ValueError):
        return None
    return SIG_TO_GROUP_ID.get(sig_id_int)


def get_failing_signature_groups_for_revision(revision):
    """
    Return the list of failing signature-group IDs for a given revision.

    This derives group IDs from alert_summary_fail_perf_sigs.csv (signature IDs)
    by mapping each signature via sig_groups.jsonl.
    """
    _load_perf_metadata()
    _load_sig_groups_mapping()

    sig_ids = REVISION_FAIL_SIG_IDS.get(revision, []) or []
    failing_group_ids = set()
    missing = set()
    for sig in sig_ids:
        try:
            sig_id_int = int(sig)
        except (TypeError, ValueError):
            continue
        gid = SIG_TO_GROUP_ID.get(sig_id_int)
        if gid is None:
            missing.add(sig_id_int)
            continue
        failing_group_ids.add(int(gid))

    if missing:
        sample = ", ".join(str(s) for s in sorted(missing)[:20])
        extra = "" if len(missing) <= 20 else f" (and {len(missing) - 20} more...)"
        logger.warning(
            "Missing signature->group mapping for %d failing signatures on revision %s "
            "(first missing signature_ids: %s%s). Those signatures will be ignored for "
            "signature-group based simulation.",
            len(missing),
            str(revision),
            sample,
            extra,
        )

    return sorted(failing_group_ids)

class TestExecutor:
    """
    Central test executor with per-platform worker pools.

    Each scheduled job provides:
      - a duration (minutes)
      - a signature_group_id, used to route the job to a platform pool

    Parameters
    ----------
    worker_pools:
        Either:
          - int: a single shared pool ("default") with that many workers, or
          - dict[str, int]: pool_name -> worker_count mapping.
    """

    def __init__(self, worker_pools):
        if worker_pools is None:
            worker_pools = dict(DEFAULT_WORKER_POOLS)

        if isinstance(worker_pools, int):
            pool_sizes = {"default": int(worker_pools)}
        elif isinstance(worker_pools, dict):
            pool_sizes = {}
            for k, v in worker_pools.items():
                if v is None:
                    continue
                pool = str(k).strip()
                if not pool:
                    continue
                pool_sizes[pool] = int(v)
        else:
            raise TypeError(
                "TestExecutor worker_pools must be an int or dict[str,int]; "
                f"got {type(worker_pools)!r}"
            )

        if not pool_sizes:
            raise ValueError("TestExecutor requires at least one worker pool with workers > 0.")

        bad = {k: v for k, v in pool_sizes.items() if int(v) <= 0}
        if bad:
            raise ValueError(f"All worker pool sizes must be positive; got: {bad}")

        self.pool_sizes = pool_sizes
        # Prefer configured fallback pool when present (default: "mac").
        preferred_defaults = []
        for candidate in (_unknown_platform_pool, "mac", "linux"):
            if candidate and candidate not in preferred_defaults:
                preferred_defaults.append(candidate)
        self.default_pool = next(
            (k for k in preferred_defaults if k in self.pool_sizes), next(iter(self.pool_sizes))
        )

        # Per-pool min-heaps of (free_time, worker_index), lazily initialized.
        self._worker_heaps = {}
        # Cumulative CPU time in minutes across all scheduled tests
        self.total_cpu_minutes = 0.0
        logger.debug("Created TestExecutor with pools=%s", self.pool_sizes)

    def _ensure_initialized(self, pool: str, t0):
        heap = self._worker_heaps.get(pool)
        if heap:
            return
        n = int(self.pool_sizes.get(pool, 0))
        if n <= 0:
            raise ValueError(f"Unknown or empty worker pool {pool!r} (pool_sizes={self.pool_sizes})")
        heap = [(t0, i) for i in range(n)]
        heapq.heapify(heap)
        self._worker_heaps[pool] = heap

    def schedule(self, requested_start_time, duration_minutes: float, signature_group_id=None):
        """
        Submit a single job that becomes 'ready' at requested_start_time,
        with its own duration (in minutes) and a signature_group_id.

        Returns the actual finish time given current queue & workers.
        """
        pool = (
            _get_worker_pool_for_signature_group(signature_group_id)
            if signature_group_id is not None
            else self.default_pool
        )
        if pool not in self.pool_sizes:
            pool = self.default_pool

        self._ensure_initialized(pool, requested_start_time)
        heap = self._worker_heaps[pool]

        # Find worker that becomes free earliest
        earliest_free, idx = heapq.heappop(heap)

        actual_start = max(requested_start_time, earliest_free)
        try:
            duration_float = float(duration_minutes)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Invalid duration_minutes value passed to TestExecutor.schedule: {duration_minutes!r}"
            ) from exc

        finish_time = actual_start + timedelta(minutes=duration_float)
        # Push worker back with updated free time
        heapq.heappush(heap, (finish_time, idx))
        # Accumulate CPU time regardless of parallelism
        self.total_cpu_minutes += duration_float
        # logger.debug(
        #     "Scheduled test on worker %d: start=%s, duration=%.2f min, finish=%s",
        #     idx,
        #     actual_start,
        #     duration_minutes,
        #     finish_time,
        # )
        return finish_time


def run_test_suite(executor: TestExecutor, requested_start_time, durations_minutes):
    """
    Run a suite of tests in parallel as much as the executor allows.

    durations_minutes:
        Iterable of either:
          - float duration_minutes, or
          - (signature_group_id, duration_minutes) tuples.
    All tests become ready at `requested_start_time`.

    Returns the time when the *last* of those tests finishes.
    """
    if not durations_minutes:
        return requested_start_time

    # Build must complete before the suite's jobs become ready.
    ready_time = requested_start_time + timedelta(minutes=float(_build_time_minutes))

    last_finish = ready_time
    count = 0
    for entry in durations_minutes:
        count += 1
        sig_group_id = None
        dur = entry
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            sig_group_id, dur = entry[0], entry[1]
        elif isinstance(entry, dict):
            sig_group_id = entry.get("signature_group_id")
            dur = entry.get("duration_minutes")

        finish_time = executor.schedule(
            ready_time,
            dur,
            signature_group_id=sig_group_id,
        )
        if finish_time > last_finish:
            last_finish = finish_time

    logger.debug(
        "run_test_suite: scheduled %d tests from %s",
        count,
        requested_start_time,
    )
    logger.debug(
        "run_test_suite: last test finished at %s (span=%.2f min)",
        last_finish,
        (last_finish - requested_start_time).total_seconds() / 60.0,
    )
    return last_finish


def batch_has_regressor(batch):
    return any(c["true_label"] for c in batch)


# ------------------ SHARED INTERVAL TEST HELPER ------------------ #

def _run_interval_and_update(
    batch_sorted,
    status,
    lo,
    hi,
    current_time,
    first_test,
    total_tests_run,
    feedback_times,
    executor,
    batch_fail_durations,
):
    """
    Shared helper for 'test_interval' used by multiple bisection strategies.

    Semantics:

      - For the first test in a batch (first_test == True), we model a *full*
        perf batch run: all signature-groups from sig_group_job_durations.csv.

      - For subsequent interval tests (bisection steps), we model a targeted run
        using the failing perf signature-groups for this batch.

    Cost is counted as the number of signature-group jobs run in the suite.
    """
    if lo > hi:
        return False, current_time, first_test, total_tests_run

    if first_test:
        durations = get_batch_signature_durations()
    else:
        durations = batch_fail_durations

    tests_this_run = len(durations)
    first_test = False  # all subsequent calls in this batch are "bisection"

    logger.debug(
        "_run_interval_and_update: interval [%d, %d], first_test=%s, tests_this_run=%d, current_time=%s",
        lo,
        hi,
        first_test,
        tests_this_run,
        current_time,
    )

    # Cost accounting
    total_tests_run += tests_this_run

    # All physical tests for this logical run start at current_time
    finish_time = run_test_suite(executor, current_time, durations)

    has_defect = any(
        batch_sorted[i]["true_label"] and status[i] != "defect_found"
        for i in range(lo, hi + 1)
    )

    if not has_defect:
        # Interval is clean; provide feedback for all unknown commits here.
        for i in range(lo, hi + 1):
            if status[i] == "unknown":
                status[i] = "clean"
                cid = batch_sorted[i]["commit_id"]
                if cid not in feedback_times:
                    fb_min = (finish_time - batch_sorted[i]["ts"]).total_seconds() / 60.0
                    feedback_times[cid] = fb_min

    current_time = finish_time
    logger.debug(
        "_run_interval_and_update: finished interval [%d, %d], has_defect=%s, total_tests_run=%d",
        lo,
        hi,
        has_defect,
        total_tests_run,
    )
    return has_defect, current_time, first_test, total_tests_run


# ------------------ TOB ------------------ #

def time_ordered_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Time-Ordered Bisection (TOB) with central executor.

    Cost/time model:

      - First call within a batch:
          * Full batch run (all signature-groups).
      - Subsequent calls (bisection steps):
          * Only the failing signature-groups for this batch.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    # Batches are constructed in time order by the caller, so we can
    # avoid re-sorting here and treat the incoming list as already
    # sorted by timestamp.
    batch_sorted = batch
    n = len(batch_sorted)
    tests_before = total_tests_run
    culprits_before = len(culprit_times)
    logger.debug("time_ordered_bisect: starting batch_size=%d", n)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the entire unknown region
        still_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted,
            status,
            lo,
            hi,
            current_time,
            first_test,
            total_tests_run,
            feedback_times,
            executor,
            batch_fail_durations,
        )
        if not still_fails:
            break

        left = lo
        right = hi
        # Standard midpoint-based binary search
        while left < right:
            mid = (left + right) // 2
            left_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
                batch_sorted,
                status,
                left,
                mid,
                current_time,
                first_test,
                total_tests_run,
                feedback_times,
                executor,
                batch_fail_durations,
            )
            if left_fails:
                right = mid
            else:
                left = mid + 1

        idx = left
        c = batch_sorted[idx]
        status[idx] = "defect_found"

        fb_min = (current_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "time_ordered_bisect: batch_size=%d, tests_batch_root=%d, "
        "tests_bisection=%d, culprits_this_batch=%d, culprits_total=%d, "
        "total_tests_run=%d",
        n,
        tests_batch_root,
        tests_bisection,
        culprits_batch,
        len(culprit_times),
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times


# ------------------ Exhaustive variants ------------------ #

def exhaustive_parallel(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Parallel 'no-bisection' strategy with realistic durations.
    
      - For each batch:
          * First test: ONE "full" batch run using ALL signature-groups
            from sig_group_job_durations.csv.

          * Subsequent per-commit runs (for all commits EXCEPT the last):
              - Use the SAME failing signature-group suite as bisection for
                this batch, i.e., the union of failing perf signature-groups
                across all regressor revisions in this batch, as derived
                from alert_summary_fail_perf_sigs.csv.

          * The last commit's feedback time is when the last per-commit
            suite finishes (no additional tests for the last one).

    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    # Caller provides commits in time order; reuse directly to avoid
    # an extra sort per batch.
    batch_sorted = batch
    n = len(batch_sorted)
    tests_before = total_tests_run
    culprits_before = len(culprit_times)
    logger.debug("exhaustive_parallel: starting batch_size=%d", n)

    # Precompute batch-level failing signature-groups (same set used everywhere in this batch)
    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

    # Degenerate: single commit
    if n == 1:
        c = batch_sorted[0]
        cid = c["commit_id"]

        if is_batch_root:
            # Single-commit batch: just the full suite (conceptually same as batch test)
            durations = get_batch_signature_durations()
        else:
            # Non-root usage: use batch-level failing signatures (still meaningful)
            durations = batch_fail_durations

        tests_this_run = len(durations)
        total_tests_run += tests_this_run
        finish_time = run_test_suite(executor, start_time, durations)

        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
        feedback_times[cid] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

        return total_tests_run, culprit_times, feedback_times

    # Normal case: at least 2 commits
    if is_batch_root:
        # 1) One "full" batch run: all signature-groups.
        durations_full = get_batch_signature_durations()
        total_tests_run += len(durations_full)
        finish_time_batch = run_test_suite(executor, start_time, durations_full)
        requested_time = finish_time_batch
    else:
        # Non-root usage: no full batch run; per-commit suites start at start_time.
        requested_time = start_time

    # 2) Per-commit suites using batch-level failing signatures
    #    for all commits EXCEPT the last.
    intermediate_finish_times = []
    for idx in range(0, n - 1):
        c = batch_sorted[idx]
        cid = c["commit_id"]

        durations = batch_fail_durations
        tests_this_run = len(durations)
        total_tests_run += tests_this_run

        finish_time = run_test_suite(executor, requested_time, durations)
        intermediate_finish_times.append(finish_time)

        fb_min = (finish_time - c["ts"]).total_seconds() / 60.0
        feedback_times[cid] = fb_min

        if c["true_label"]:
            culprit_times.append(fb_min)

    # 3) Feedback for the last commit comes when the last per-commit suite finishes.
    last_commit = batch_sorted[-1]
    last_cid = last_commit["commit_id"]

    if intermediate_finish_times:
        last_intermediate_finish = max(intermediate_finish_times)
    else:
        # Shouldn't happen for n >= 2, but guard anyway
        last_intermediate_finish = requested_time

    fb_min_last = (
        last_intermediate_finish - last_commit["ts"]
    ).total_seconds() / 60.0
    feedback_times[last_cid] = fb_min_last
    if last_commit["true_label"]:
        culprit_times.append(fb_min_last)

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "exhaustive_parallel: batch_size=%d, tests_batch_root=%d, "
        "tests_bisection=%d, culprits_this_batch=%d, culprits_total=%d, "
        "total_tests_run=%d",
        n,
        tests_batch_root,
        tests_bisection,
        culprits_batch,
        len(culprit_times),
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times


# ------------------ RWAB ------------------ #

def risk_weighted_adaptive_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Risk-Weighted Adaptive Bisection (RWAB).

    Like time_ordered_bisect, but splits failing intervals so that the sum of
    predicted risk on the left and right halves is approximately balanced.

    Cost/time model:

      - First batch test uses the full suite.
      - Subsequent bisection tests use only the failing signatures for
        this batch.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    # Batches from higher-level strategies are already time-ordered.
    batch_sorted = batch
    n = len(batch_sorted)
    tests_before = total_tests_run
    culprits_before = len(culprit_times)
    logger.debug("risk_weighted_adaptive_bisect: starting batch_size=%d", n)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    # Prefix sums of risk
    risk_prefix = [0.0] * (n + 1)
    for i, c in enumerate(batch_sorted):
        if "risk" not in c or c["risk"] is None:
            raise ValueError(
                f"Missing or None 'risk' value in batch for risk_weighted_adaptive_bisect at index {i}: {c!r}"
            )
        r = float(c["risk"])
        risk_prefix[i + 1] = risk_prefix[i] + r

    def risk_sum(lo, hi):
        return risk_prefix[hi + 1] - risk_prefix[lo]

    def choose_risk_balanced_split(lo, hi):
        if lo >= hi:
            return lo
        total_risk = risk_sum(lo, hi)
        if total_risk <= 0.0:
            return (lo + hi) // 2

        target = total_risk / 2.0
        best_idx = lo
        best_diff = float("inf")

        for s in range(lo, hi):
            left_risk = risk_sum(lo, s)
            diff = abs(left_risk - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = s

        return best_idx

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the whole unknown region
        still_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted,
            status,
            lo,
            hi,
            current_time,
            first_test,
            total_tests_run,
            feedback_times,
            executor,
            batch_fail_durations,
        )
        if not still_fails:
            break

        left = lo
        right = hi
        # Risk-weighted "binary" search
        while left < right:
            split = choose_risk_balanced_split(left, right)
            left_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
                batch_sorted,
                status,
                left,
                split,
                current_time,
                first_test,
                total_tests_run,
                feedback_times,
                executor,
                batch_fail_durations,
            )
            if left_fails:
                right = split
            else:
                left = split + 1

        idx = left
        c = batch_sorted[idx]
        status[idx] = "defect_found"

        fb_min = (current_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "risk_weighted_adaptive_bisect: batch_size=%d, tests_batch_root=%d, "
        "tests_bisection=%d, culprits_this_batch=%d, culprits_total=%d, "
        "total_tests_run=%d",
        n,
        tests_batch_root,
        tests_bisection,
        culprits_batch,
        len(culprit_times),
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times


# ------------------ RWAB-LS ------------------ #

def risk_weighted_adaptive_bisect_log_survival(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Risk-Weighted Adaptive Bisection using log-survival combined probability (RWAB-LS).

    Same as RWAB, but balances split points using a combined probability mass over
    an interval:

        1 - Π(1 - p_i)

    where p_i are per-commit risk probabilities. The combined probability is
    computed via log-survival prefix sums for numerical stability.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    # Batches from higher-level strategies are already time-ordered.
    batch_sorted = batch
    n = len(batch_sorted)
    tests_before = total_tests_run
    culprits_before = len(culprit_times)
    logger.debug("risk_weighted_adaptive_bisect_log_survival: starting batch_size=%d", n)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    # Prefix sums of log survival mass:
    #
    #   log_survival_prefix[k] = Σ log(1 - p_i) for i in [0, k)
    #
    # We keep p==1.0 as a separate prefix count to avoid propagating -inf and
    # producing NaNs when subtracting -inf - (-inf) for ranges that do not
    # include the p==1.0 element.
    log_survival_prefix = [0.0] * (n + 1)
    certain_fail_prefix = [0] * (n + 1)
    for i, c in enumerate(batch_sorted):
        if "risk" not in c or c["risk"] is None:
            raise ValueError(
                "Missing or None 'risk' value in batch for risk_weighted_adaptive_bisect_log_survival "
                f"at index {i}: {c!r}"
            )
        p = float(c["risk"])
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Risk probabilities must be in [0,1], got {p} at index {i}: {c!r}")

        log_survival_prefix[i + 1] = log_survival_prefix[i]
        certain_fail_prefix[i + 1] = certain_fail_prefix[i]
        if p >= 1.0:
            certain_fail_prefix[i + 1] += 1
        elif p > 0.0:
            log_survival_prefix[i + 1] += math.log1p(-p)

    def combined_probability(lo, hi):
        """
        Combined probability of at least one defect in the inclusive interval [lo, hi].
        """
        if lo > hi:
            return 0.0
        start = lo
        end = hi + 1
        if certain_fail_prefix[end] - certain_fail_prefix[start] > 0:
            return 1.0
        log_survival = float(log_survival_prefix[end] - log_survival_prefix[start])
        return float(1.0 - math.exp(log_survival))

    def choose_probability_balanced_split(lo, hi):
        """
        Choose split s in [lo, hi-1] such that combined probability mass is
        approximately balanced between [lo..s] and [s+1..hi].
        """
        if lo >= hi:
            return lo

        total_mass = combined_probability(lo, hi)
        if total_mass <= 0.0:
            return (lo + hi) // 2

        search_lo = lo
        search_hi = hi - 1
        best_idx = (lo + hi) // 2
        best_gap = float("inf")

        # gap(mid) = mass_left - mass_right is monotone increasing in mid for contiguous halves.
        while search_lo <= search_hi:
            mid = (search_lo + search_hi) // 2
            mass_left = combined_probability(lo, mid)
            mass_right = combined_probability(mid + 1, hi)
            gap = abs(mass_left - mass_right)
            if gap < best_gap:
                best_gap = gap
                best_idx = mid

            if mass_left < mass_right:
                search_lo = mid + 1
            else:
                search_hi = mid - 1

        if best_idx < lo:
            best_idx = lo
        if best_idx >= hi:
            best_idx = hi - 1
        return best_idx

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the whole unknown region
        still_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted,
            status,
            lo,
            hi,
            current_time,
            first_test,
            total_tests_run,
            feedback_times,
            executor,
            batch_fail_durations,
        )
        if not still_fails:
            break

        left = lo
        right = hi
        # Risk-weighted "binary" search using combined probability mass
        while left < right:
            split = choose_probability_balanced_split(left, right)
            left_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
                batch_sorted,
                status,
                left,
                split,
                current_time,
                first_test,
                total_tests_run,
                feedback_times,
                executor,
                batch_fail_durations,
            )
            if left_fails:
                right = split
            else:
                left = split + 1

        idx = left
        c = batch_sorted[idx]
        status[idx] = "defect_found"

        fb_min = (current_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "risk_weighted_adaptive_bisect_log_survival: batch_size=%d, tests_batch_root=%d, "
        "tests_bisection=%d, culprits_this_batch=%d, culprits_total=%d, "
        "total_tests_run=%d",
        n,
        tests_batch_root,
        tests_bisection,
        culprits_batch,
        len(culprit_times),
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times


# ------------------ TKRB-K: Top-K Risk-First Bisection ------------------ #

def topk_risk_first_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Top-K Risk-First Bisection (TKRB-K).

    Cost/time model:

      - First batch test uses the full suite.
      - All subsequent tests (probes and fallback bisection) use only the
        failing signatures for this batch.

    Step 2 (Top-K probing):
      - `TKRB_TOP_K` controls how many of the highest-risk commits are probed.
      - For `TKRB_TOP_K == 1`, probing is sequential.
      - For `TKRB_TOP_K > 1`, all top-K probes are submitted in parallel but
        processed with a barrier (PAR-style): follow-up work happens only after
        all probes finish.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    # Assume batch is already sorted by timestamp, as produced by the
    # batching strategies; avoid redundant sorting here.
    batch_sorted = batch
    n = len(batch_sorted)
    tests_before = total_tests_run
    culprits_before = len(culprit_times)
    logger.debug("topk_risk_first_bisect: starting batch_size=%d", n)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)
    logger.debug(
        "topk_risk_first_bisect: TKRB_TOP_K=%d, failing_signatures=%d",
        int(TKRB_TOP_K),
        len(batch_fail_durations),
    )

    def run_interval(lo, hi):
        """
        Wrapper around _run_interval_and_update that:
          - Skips empty intervals.
          - Runs the interval test and updates time/cost/status/feedback.

        Note: Unlike TKRB's previous versions, we intentionally do not cache
        interval results. Other bisection algorithms in this file also do not
        cache interval outcomes, and caching is subtle because interval outcomes
        depend on the evolving `status` state.
        """
        nonlocal current_time, first_test, total_tests_run

        if lo > hi:
            return False

        has_defect, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted,
            status,
            lo,
            hi,
            current_time,
            first_test,
            total_tests_run,
            feedback_times,
            executor,
            batch_fail_durations,
        )
        logger.debug(
            "topk_risk_first_bisect: interval [%d, %d] executed, has_defect=%s",
            lo,
            hi,
            has_defect,
        )
        return has_defect

    def tob_bisect_on_range(lo_bound, hi_bound):
        """
        Run a TOB-style bisection restricted to [lo_bound .. hi_bound],
        using run_interval for physical runs.
        """
        nonlocal current_time, first_test, total_tests_run

        while True:
            unknown_indices = [
                i for i in range(lo_bound, hi_bound + 1)
                if status[i] == "unknown"
            ]
            if not unknown_indices:
                break

            lo = unknown_indices[0]
            hi = unknown_indices[-1]

            still_fails = run_interval(lo, hi)
            if not still_fails:
                break

            left = lo
            right = hi
            while left < right:
                mid = (left + right) // 2
                left_fails = run_interval(left, mid)
                if left_fails:
                    right = mid
                else:
                    left = mid + 1

            culprit_idx = left
            c = batch_sorted[culprit_idx]
            if status[culprit_idx] != "defect_found":
                status[culprit_idx] = "defect_found"
                fb_min = (current_time - c["ts"]).total_seconds() / 60.0
                feedback_times[c["commit_id"]] = fb_min
                if c["true_label"]:
                    culprit_times.append(fb_min)

    # ---- Step 1: Initial whole-batch test (like TOB) ----
    whole_fails = run_interval(0, n - 1)
    if not whole_fails:
        # Entire batch clean
        return total_tests_run, culprit_times, feedback_times

    # ---- Step 2: Top-K risk-first probing ----
    risks = []
    for i in range(n):
        c = batch_sorted[i]
        if "risk" not in c or c["risk"] is None:
            raise ValueError(
                f"Missing or None 'risk' value in batch for topk_risk_first_bisect at index {i}: {c!r}"
            )
        risks.append((i, float(c["risk"])))
    risks.sort(key=lambda t: t[1], reverse=True)
    top_indices = [i for (i, _) in risks[:TKRB_TOP_K]]

    def mark_culprit(idx, at_time):
        c = batch_sorted[idx]
        if status[idx] == "defect_found":
            return
        status[idx] = "defect_found"
        fb_min = (at_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    def interval_has_defect(lo, hi):
        return any(
            batch_sorted[i]["true_label"] and status[i] != "defect_found"
            for i in range(lo, hi + 1)
        )

    def apply_clean_interval(lo, hi, finish_time):
        for i in range(lo, hi + 1):
            if status[i] != "unknown":
                continue
            status[i] = "clean"
            cid = batch_sorted[i]["commit_id"]
            if cid not in feedback_times:
                fb_min = (finish_time - batch_sorted[i]["ts"]).total_seconds() / 60.0
                feedback_times[cid] = fb_min

    if int(TKRB_TOP_K) <= 1 or len(top_indices) <= 1:
        for idx in top_indices:
            if status[idx] != "unknown":
                continue

            curr_fails = run_interval(0, idx)

            prev_fails = False
            if curr_fails and idx > 0:
                prev_fails = run_interval(0, idx - 1)

            if curr_fails and not prev_fails:
                mark_culprit(idx, current_time)
            elif curr_fails and prev_fails and idx > 0:
                tob_bisect_on_range(0, idx - 1)
    else:
        # Parallel top-K probes with a barrier (PAR-style): submit all probe runs
        # at the same requested start time, wait until all of them complete,
        # then process the results.
        probe_start = current_time
        latest_finish = probe_start
        probe_has_defect_by_idx = {}

        for idx in top_indices:
            if status[idx] != "unknown":
                continue

            lo, hi = 0, idx

            durations = batch_fail_durations
            total_tests_run += len(durations)
            finish_time = run_test_suite(executor, probe_start, durations)
            latest_finish = max(latest_finish, finish_time)

            has_defect = interval_has_defect(lo, hi)
            probe_has_defect_by_idx[idx] = has_defect

        # Barrier: do not react to individual probe results until all probes finish.
        current_time = max(current_time, latest_finish)

        # Process probe outcomes after the barrier.
        for idx in top_indices:
            if status[idx] != "unknown":
                continue

            curr_fails = bool(probe_has_defect_by_idx.get(idx, False))
            if not curr_fails:
                apply_clean_interval(0, idx, current_time)
                continue

            prev_fails = False
            if idx > 0:
                prev_fails = run_interval(0, idx - 1)

            if curr_fails and not prev_fails:
                mark_culprit(idx, current_time)
            elif curr_fails and prev_fails and idx > 0:
                tob_bisect_on_range(0, idx - 1)

    # ---- Step 3: Fall back to TOB-style search over remaining unknowns ----
    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        still_fails = run_interval(lo, hi)
        if not still_fails:
            break

        left = lo
        right = hi
        while left < right:
            mid = (left + right) // 2
            left_fails = run_interval(left, mid)
            if left_fails:
                right = mid
            else:
                left = mid + 1

        culprit_idx = left
        c = batch_sorted[culprit_idx]
        if status[culprit_idx] != "defect_found":
            status[culprit_idx] = "defect_found"
            fb_min = (current_time - c["ts"]).total_seconds() / 60.0
            feedback_times[c["commit_id"]] = fb_min
            if c["true_label"]:
                culprit_times.append(fb_min)

    tests_added = total_tests_run - tests_before
    tests_batch_root = len(get_batch_signature_durations()) if is_batch_root else 0
    tests_bisection = max(0, tests_added - tests_batch_root)
    culprits_batch = len(culprit_times) - culprits_before
    logger.debug(
        "topk_risk_first_bisect: batch_size=%d, tests_batch_root=%d, "
        "tests_bisection=%d, culprits_this_batch=%d, culprits_total=%d, "
        "total_tests_run=%d",
        n,
        tests_batch_root,
        tests_bisection,
        culprits_batch,
        len(culprit_times),
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times


def sequential_walk_backward_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Sequential Walk-Backward Bisection (SWB).

    When a batch is known to contain at least one defect, SWB finds culprits by
    walking backwards from the end of the currently-unknown region and testing
    each prior boundary sequentially.

    Concretely, for an unknown interval [lo, hi] that fails:
      - Treat 'hi' as a known-buggy boundary (the interval [lo, hi] contains a regressor).
      - Test [lo, hi-1], then [lo, hi-2], ... sequentially (waiting for each result).
      - If [lo, k] is clean and [lo, k+1] is buggy, the culprit is k+1.

    Cost/time model matches other strategies:
      - First run in a batch (is_batch_root=True) uses the configured full suite.
      - Subsequent runs use the failing-signature suite for this batch.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = batch
    n = len(batch_sorted)
    logger.debug("sequential_walk_backward_bisect: starting batch_size=%d", n)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the entire unknown region first.
        still_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted,
            status,
            lo,
            hi,
            current_time,
            first_test,
            total_tests_run,
            feedback_times,
            executor,
            batch_fail_durations,
        )
        if not still_fails:
            break

        # Walk backwards from hi, testing [lo, k] for k=hi-1..lo.
        if lo == hi:
            culprit_idx = lo
        else:
            culprit_idx = lo
            for k in range(hi - 1, lo - 1, -1):
                fails_k, current_time, first_test, total_tests_run = _run_interval_and_update(
                    batch_sorted,
                    status,
                    lo,
                    k,
                    current_time,
                    first_test,
                    total_tests_run,
                    feedback_times,
                    executor,
                    batch_fail_durations,
                )
                if not fails_k:
                    culprit_idx = k + 1
                    break
                if k == lo:
                    culprit_idx = lo

        c = batch_sorted[culprit_idx]
        status[culprit_idx] = "defect_found"

        fb_min = (current_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    logger.debug(
        "sequential_walk_backward_bisect: finished batch_size=%d total_tests_run=%d",
        n,
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times


def sequential_walk_forward_bisect(
    batch,
    start_time,
    total_tests_run,
    culprit_times,
    feedback_times,
    executor,
    is_batch_root=True,
):
    """
    Sequential Walk-Forward Bisection (SWF).

    When a batch is known to contain at least one defect, SWF finds culprits by
    walking forwards in time from the beginning of the currently-unknown
    region, testing increasingly larger prefixes sequentially.

    For an unknown interval [lo, hi] that fails:
      - Test [lo, lo], then [lo, lo+1], ... sequentially (waiting for each result).
      - The first index k for which [lo, k] fails is the culprit (since [lo, k-1]
        would have been clean).

    Cost/time model matches other strategies:
      - First run in a batch (is_batch_root=True) uses the configured full suite.
      - Subsequent runs use the failing-signature suite for this batch.
    """
    if not batch:
        return total_tests_run, culprit_times, feedback_times

    batch_sorted = batch
    n = len(batch_sorted)
    logger.debug("sequential_walk_forward_bisect: starting batch_size=%d", n)

    status = ["unknown"] * n
    current_time = start_time
    first_test = bool(is_batch_root)

    batch_fail_durations = get_failing_signature_durations_for_batch(batch_sorted)

    while True:
        unknown_indices = [i for i, s in enumerate(status) if s == "unknown"]
        if not unknown_indices:
            break

        lo = unknown_indices[0]
        hi = unknown_indices[-1]

        # Test the entire unknown region first.
        still_fails, current_time, first_test, total_tests_run = _run_interval_and_update(
            batch_sorted,
            status,
            lo,
            hi,
            current_time,
            first_test,
            total_tests_run,
            feedback_times,
            executor,
            batch_fail_durations,
        )
        if not still_fails:
            break

        # Walk forwards from lo, testing [lo, k] for k=lo..hi.
        culprit_idx = hi
        for k in range(lo, hi + 1):
            fails_k, current_time, first_test, total_tests_run = _run_interval_and_update(
                batch_sorted,
                status,
                lo,
                k,
                current_time,
                first_test,
                total_tests_run,
                feedback_times,
                executor,
                batch_fail_durations,
            )
            if fails_k:
                culprit_idx = k
                break

        c = batch_sorted[culprit_idx]
        status[culprit_idx] = "defect_found"

        fb_min = (current_time - c["ts"]).total_seconds() / 60.0
        feedback_times[c["commit_id"]] = fb_min
        if c["true_label"]:
            culprit_times.append(fb_min)

    logger.debug(
        "sequential_walk_forward_bisect: finished batch_size=%d total_tests_run=%d",
        n,
        total_tests_run,
    )
    return total_tests_run, culprit_times, feedback_times
