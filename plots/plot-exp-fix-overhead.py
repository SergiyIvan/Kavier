#!/usr/bin/env python3
"""Plot fix-job overhead: wasted compute fraction due to OOM restarts.

For each failing (patched) job that has a fix-job dependency, compute the per-job
overhead ratio:

  overhead(job) = runtime(failing) / runtime(fix)

This measures how much wasted compute was incurred relative to the useful compute
of the fix-job.  A value of 0.10 means 10% extra runtime was wasted on the failed
attempt before the restart completed.

For each (X, seed), compute the mean overhead across all (failing, fix) pairs in
that file.  Then plot mean ± 1 std of those per-seed means across seeds, plus thin
min/max bars.

X=0 has no failing jobs → the overhead is defined as 0 (no wasted compute).

NOTE: this is mean-of-ratios (per-job), not ratio-of-means. This means that each
job, regardless of its duration, contributes equally to the metric. If fix-job is
super-short (e.g., baseline job also crashes), then it will inflate the metric.

Usage:
    python plot-exp-fix-overhead.py <results_dir>

Output:
    exp-fix-overhead.pdf  (written to the current working directory)
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import PERCENTAGES, SEEDS, FIX_JOB_METADATA, VALID_METADATA  # noqa: E402

ALLOWED_METADATA = VALID_METADATA | {FIX_JOB_METADATA}


def load_fix_pairs(path: Path) -> Optional[pd.DataFrame]:
    """Return a DataFrame of (failing_runtime_s, fix_runtime_s) pairs from one results file.

    Each row is one (failing job, fix-job) pair.  Returns None when the file has no
    fix-jobs (e.g. X=0).
    """
    df = pd.read_csv(path)

    if "metadata.source" not in df.columns:
        sys.exit(f"ERROR: {path} has no 'metadata.source' column.")
    bad = df[~df["metadata.source"].isin(ALLOWED_METADATA)]
    if not bad.empty:
        sys.exit(
            f"ERROR: {path} contains invalid metadata.source values: "
            f"{bad['metadata.source'].unique().tolist()}."
        )

    fix_df = df[df["metadata.source"] == FIX_JOB_METADATA]
    if fix_df.empty:
        return None

    job_id_to_runtime: dict[str, float] = dict(zip(df["job_id"], df["runtime_s"]))

    rows = []
    for fix_row in fix_df.itertuples(index=False):
        raw_deps = fix_row.dependencies
        try:
            deps = json.loads(raw_deps) if isinstance(raw_deps, str) else []
        except (json.JSONDecodeError, TypeError):
            sys.exit(
                f"ERROR: {path}: fix-job '{fix_row.job_id}' has unparseable "
                f"dependencies: {raw_deps!r}"
            )
        if len(deps) != 1:
            sys.exit(
                f"ERROR: {path}: fix-job '{fix_row.job_id}' must have exactly 1 "
                f"dependency, got {len(deps)}: {deps}"
            )
        parent_id = deps[0]
        if parent_id not in job_id_to_runtime:
            sys.exit(
                f"ERROR: {path}: fix-job '{fix_row.job_id}' references unknown "
                f"parent '{parent_id}'."
            )
        rows.append({
            "failing_runtime_s": job_id_to_runtime[parent_id],
            "fix_runtime_s": fix_row.runtime_s,
        })

    return pd.DataFrame(rows)


def compute_overhead(results_dir: Path) -> np.ndarray:
    """Return per-seed mean overhead array of shape (len(PERCENTAGES), len(SEEDS)).

    overhead[i, j] = mean(runtime(failing) / runtime(fix)) for all pairs in file (i, j).
    0.0 when no fix-jobs exist (X=0: no wasted compute by definition).
    """
    n_pct = len(PERCENTAGES)
    n_seed = len(SEEDS)
    data = np.zeros((n_pct, n_seed))

    for i, pct in enumerate(PERCENTAGES):
        for j, seed in enumerate(SEEDS):
            path = results_dir / f"kavier-X{pct}-s{seed}_per_jobs.csv"
            if not path.exists():
                sys.exit(f"ERROR: expected results file not found: {path}")
            pairs = load_fix_pairs(path)
            if pairs is not None and not pairs.empty:
                ratios = (pairs["failing_runtime_s"] / pairs["fix_runtime_s"]) * 100
                data[i, j] = float(ratios.mean())

    return data


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {results_dir}")

    data = compute_overhead(results_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.mean(data, axis=1)
        stds  = np.std(data, axis=1, ddof=1)
        mins  = np.min(data, axis=1)
        maxs  = np.max(data, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))

    eb = ax.errorbar(
        PERCENTAGES,
        means,
        yerr=stds,
        linestyle="-",
        marker="o",
        capsize=4,
        label="Fix-job overhead",
    )
    color = eb[0].get_color()
    lo = np.clip(means - mins, 0, None)
    hi = np.clip(maxs - means, 0, None)
    ax.errorbar(
        PERCENTAGES,
        means,
        yerr=[lo, hi],
        linestyle="none",
        capsize=2,
        elinewidth=0.8,
        color=color,
    )

    ax.set_xlabel("Patched jobs (%)", fontsize=16)
    ax.set_ylabel("Mean overhead in %\n(wasted / useful runtime)", fontsize=16)
    ax.set_xlim(min(PERCENTAGES) - 5, max(PERCENTAGES) + 5)
    ax.set_ylim(0, (means + stds).max() * 1.15)
    ax.set_xticks(PERCENTAGES)
    ax.grid(True)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig("exp-fix-overhead.pdf")


if __name__ == "__main__":
    main()
