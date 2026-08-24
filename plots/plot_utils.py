"""Shared utilities for experiment plotting scripts.

Provides constants, data loading with fix-job merging, and per-group metric helpers
used by plot-exp-jct-absolute.py, plot-exp-jct-ratio.py, and future experiment plots.

Fix-job semantics
-----------------
A fix-job (prefix ``dynamic_oom_fix-``, metadata ``"fix_job"``) restarts a failed patched job.
From the user's perspective the two rows represent a single job lifetime, so ``load_and_merge``
folds the fix-job's metrics into its parent before returning:

  turnaround_s  +=  fix.turnaround_s
  wait_s        +=  fix.wait_s
  runtime_s     +=  fix.runtime_s

The equivalence ``turnaround(parent) + turnaround(fix) == end_s(fix) - submit_s(parent)`` is
guaranteed by construction (fix is submitted 1 s after parent; ready_s(fix) = end_s(parent)
always, so the sum telescopes exactly).
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Experiment structure constants
# ---------------------------------------------------------------------------

ITERATION_COUNT = 25
PERCENTAGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEEDS = list(range(1, ITERATION_COUNT + 1))

# ---------------------------------------------------------------------------
# Job-classification constants
# ---------------------------------------------------------------------------

VALID_METADATA = {"baseline", "patched"}
PATCHING_ENTITIES = {"min_gpu_recommender", "avoid_oom_recommender"}
FIX_JOB_PREFIX = "dynamic_oom_fix-"
FIX_JOB_METADATA = "fix_job"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_uid(uid: str) -> str:
    """Strip a known patching-entity prefix from a UID, if present.

    Patching entity names never contain dashes, so splitting on the first dash
    and checking the prefix against PATCHING_ENTITIES is unambiguous.
    """
    parts = uid.split("-", 1)
    if len(parts) == 2 and parts[0] in PATCHING_ENTITIES:
        return parts[1]
    return uid


def load_and_merge(path: Path) -> pd.DataFrame:
    """Load a per-jobs CSV, merge fix-job metrics into their parents, and return the result.

    Steps:
    1. Read CSV.
    2. Validate: every ``job_metadata`` must be in ``VALID_METADATA | {FIX_JOB_METADATA}``.
       Abort on any other value.
    3. For each fix-job row: look up its parent via the ``dependencies`` column (JSON array
       containing the parent's job_id); add the fix-job's ``turnaround_s``, ``wait_s``, and
       ``runtime_s`` onto the parent row.
    4. Drop all fix-job rows.
    5. Return the merged DataFrame — all remaining rows have ``job_metadata`` in
       ``{"baseline", "patched"}``.
    """
    df = pd.read_csv(path)

    # Validate metadata
    allowed = VALID_METADATA | {FIX_JOB_METADATA}
    if "job_metadata" not in df.columns:
        sys.exit(f"ERROR: {path} has no 'job_metadata' column.")
    bad = df[~df["job_metadata"].isin(allowed)]
    if not bad.empty:
        values = bad["job_metadata"].unique().tolist()
        sys.exit(
            f"ERROR: {path} contains invalid job_metadata values: {values}. "
            f"Only {sorted(allowed)} are allowed."
        )

    fix_mask = df["job_metadata"] == FIX_JOB_METADATA
    if not fix_mask.any():
        return df.copy()

    # Index the main rows by job_id for fast parent lookup
    df = df.reset_index(drop=True)
    job_id_to_idx = {job_id: i for i, job_id in enumerate(df["job_id"])}

    additive_cols = ["turnaround_s", "wait_s", "runtime_s"]

    for fix_row in df[fix_mask].itertuples(index=True):
        raw_deps = fix_row.dependencies
        try:
            deps = json.loads(raw_deps) if isinstance(raw_deps, str) else []
        except (json.JSONDecodeError, TypeError):
            sys.exit(
                f"ERROR: {path}: fix-job '{fix_row.job_id}' has unparseable dependencies: {raw_deps!r}"
            )
        if len(deps) != 1:
            sys.exit(
                f"ERROR: {path}: fix-job '{fix_row.job_id}' must have exactly 1 dependency, "
                f"got {len(deps)}: {deps}"
            )
        parent_id = deps[0]
        if parent_id not in job_id_to_idx:
            sys.exit(
                f"ERROR: {path}: fix-job '{fix_row.job_id}' references unknown parent '{parent_id}'."
            )
        parent_idx = job_id_to_idx[parent_id]
        for col in additive_cols:
            df.at[parent_idx, col] += getattr(fix_row, col)

    return df[~fix_mask].reset_index(drop=True)


def group_mean_jct(df: pd.DataFrame) -> dict[str, float]:
    """Return mean JCT in **hours** for each group: ``"baseline"``, ``"patched"``, ``"all"``.

    Returns ``NaN`` for a group that has no rows (e.g. no patched jobs at X=0).
    """
    result: dict[str, float] = {"all": float(df["turnaround_s"].mean()) / 3600}
    for group in ("baseline", "patched"):
        subset = df.loc[df["job_metadata"] == group, "turnaround_s"]
        result[group] = float(subset.mean()) / 3600 if not subset.empty else float("nan")
    return result
