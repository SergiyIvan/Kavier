#!/usr/bin/env python3
"""Plot mean JCT ratio (± 1 std across seeds) vs. percentage of patched jobs.

For each (X, seed, group), the ratio is:

  mean_jct(group, X, seed) / mean_jct(group, X=0, reference)

where "group" is "baseline" jobs or all jobs, and the reference mean is averaged
across all X=0 seeds.  Using ratio-of-means (not mean-of-ratios) avoids distortion
from jobs with very short reference JCTs inflating per-job ratios.

"Patched jobs" are not shown as a ratio because they do not exist at X=0, so no
well-defined reference exists for that group.

The plot shows mean ± 1 std of these ratios across the SEEDS dimension.

Two lines: "Baseline jobs" (-), "All jobs" (--).
A ratio of 1.0 means no change vs. X=0; < 1.0 means faster (improvement).

Usage:
    python plot-exp-jct.py <results_dir>

Output:
    exp-jct.pdf  (written to the current working directory)
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ITERATION_COUNT = 25
PERCENTAGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEEDS = list(range(1, ITERATION_COUNT + 1))

VALID_METADATA = {"baseline", "patched"}
PATCHING_ENTITIES = {"min_gpu_recommender", "avoid_oom_recommender"}

# Percentage used as the all-baseline reference (denominator of every ratio).
REFERENCE_PCT = 0


def base_uid(uid: str) -> str:
    """Strip a known patching-entity prefix from a UID, if present.

    Patching entity names never contain dashes, so splitting on the first dash
    and checking the prefix against PATCHING_ENTITIES is unambiguous.
    """
    parts = uid.split("-", 1)
    if len(parts) == 2 and parts[0] in PATCHING_ENTITIES:
        return parts[1]
    return uid


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load a per-jobs CSV and abort if any job_metadata value is outside VALID_METADATA."""
    df = pd.read_csv(path)
    if "job_metadata" not in df.columns:
        sys.exit(f"ERROR: {path} has no 'job_metadata' column.")
    bad = df[~df["job_metadata"].isin(VALID_METADATA)]
    if not bad.empty:
        values = bad["job_metadata"].unique().tolist()
        sys.exit(
            f"ERROR: {path} contains invalid job_metadata values: {values}. "
            f"Only {sorted(VALID_METADATA)} are allowed."
        )
    return df


def group_mean_jct(df: pd.DataFrame) -> dict[str, float]:
    """Return mean turnaround_s for each group ("baseline", "patched", "all")."""
    result: dict[str, float] = {"all": df["turnaround_s"].mean()}
    for group in ("baseline", "patched"):
        subset = df.loc[df["job_metadata"] == group, "turnaround_s"]
        result[group] = float(subset.mean()) if not subset.empty else float("nan")
    return result


def build_reference(results_dir: Path) -> dict[str, float]:
    """Compute reference mean JCT (s) per group at REFERENCE_PCT, averaged across all seeds.

    Returns {"baseline": float, "all": float}.
    "patched" is omitted — at X=0 all jobs are baseline, so no patched reference exists.
    """
    accum: dict[str, list[float]] = {"baseline": [], "all": []}
    for seed in SEEDS:
        path = results_dir / f"kavier-X{REFERENCE_PCT}-s{seed}_per_jobs.csv"
        if not path.exists():
            sys.exit(f"ERROR: reference file not found: {path}")
        df = load_and_validate(path)
        for group, val in group_mean_jct(df).items():
            if group in accum and not np.isnan(val):
                accum[group].append(val)

    ref: dict[str, float] = {}
    for group, vals in accum.items():
        if not vals:
            sys.exit(f"ERROR: no data for group '{group}' in X={REFERENCE_PCT} reference files.")
        ref[group] = float(np.mean(vals))
    return ref


def compute_ratios(results_dir: Path, ref: dict[str, float]) -> dict[str, np.ndarray]:
    """Return per-seed JCT-ratio arrays of shape (len(PERCENTAGES), len(SEEDS)).

    ratio[group][i, j] = mean_jct(group, PERCENTAGES[i], SEEDS[j]) / ref[group]

    Only groups present in ref are computed ("baseline" and "all").
    """
    n_pct = len(PERCENTAGES)
    n_seed = len(SEEDS)
    data: dict[str, np.ndarray] = {group: np.full((n_pct, n_seed), np.nan) for group in ref}

    for i, pct in enumerate(PERCENTAGES):
        for j, seed in enumerate(SEEDS):
            path = results_dir / f"kavier-X{pct}-s{seed}_per_jobs.csv"
            if not path.exists():
                sys.exit(f"ERROR: expected results file not found: {path}")
            df = load_and_validate(path)
            for group, mean_jct in group_mean_jct(df).items():
                if group in ref and not np.isnan(mean_jct):
                    data[group][i, j] = mean_jct / ref[group]

    return data


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {results_dir}")

    ref = build_reference(results_dir)
    data = compute_ratios(results_dir, ref)

    fig, ax = plt.subplots(figsize=(10, 4))

    series = [
        ("baseline", "Baseline jobs", "-"),
        ("all",      "All jobs",      "--"),
    ]

    all_values = []
    for key, label, style in series:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # silence all-NaN slice warnings
            means = np.nanmean(data[key], axis=1)
            stds  = np.nanstd(data[key], axis=1, ddof=1)

        valid = ~np.isnan(means)
        all_values.extend((means[valid] + stds[valid]).tolist())

        ax.errorbar(
            np.array(PERCENTAGES)[valid],
            means[valid],
            yerr=stds[valid],
            label=label,
            linestyle=style,
            marker="o",
            capsize=4,
        )

    # Reference line at ratio = 1.0 (no change vs. X=0)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1)

    ax.set_xlabel("Patched jobs (%)", fontsize=16)
    ax.set_ylabel("Mean JCT ratio (vs. X=0)", fontsize=16)
    ax.set_xlim(min(PERCENTAGES) - 5, max(PERCENTAGES) + 5)
    upper = max(all_values) * 1.1 if all_values else 1.5
    ax.set_ylim(0, upper)
    ax.set_xticks(PERCENTAGES)
    ax.grid(True)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig("exp-jct-ratio.pdf")


if __name__ == "__main__":
    main()
