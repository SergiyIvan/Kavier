#!/usr/bin/env python3
"""Plot mean JCT ratio (± 1 std across seeds) vs. percentage of patched jobs.

For each (X, seed, group), the ratio is:

  mean_jct(group, X, seed) / mean_jct(group, X=0, reference)

where "group" is "baseline" jobs or all jobs, and the reference mean is averaged
across all X=0 seeds.  Using ratio-of-means (not mean-of-ratios) avoids distortion
from jobs with very short reference JCTs inflating per-job ratios.

"Patched jobs" are not shown as a ratio because they do not exist at X=0, so no
well-defined reference exists for that group.

Fix-jobs (metadata "fix_job") are merged into their parent's metrics before aggregation.

Two lines: "Baseline jobs" (-), "All jobs" (--).
A ratio of 1.0 means no change vs. X=0; < 1.0 means faster (improvement).

Usage:
    python plot-exp-jct-ratio.py <results_dir>

Output:
    exp-jct-ratio.pdf  (written to the current working directory)
"""

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Import shared utilities (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import PERCENTAGES, SEEDS, load_and_merge, group_mean_jct  # noqa: E402

# Percentage used as the all-baseline reference (denominator of every ratio).
REFERENCE_PCT = 0


def build_reference(results_dir: Path) -> dict[str, float]:
    """Compute reference mean JCT (h) per group at REFERENCE_PCT, averaged across all seeds.

    Returns {"baseline": float, "all": float}.
    "patched" is omitted — at X=0 all jobs are baseline, so no patched reference exists.
    """
    accum: dict[str, list[float]] = {"baseline": [], "all": []}
    for seed in SEEDS:
        path = results_dir / f"kavier-X{REFERENCE_PCT}-s{seed}_per_jobs.csv"
        if not path.exists():
            sys.exit(f"ERROR: reference file not found: {path}")
        df = load_and_merge(path)
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
            df = load_and_merge(path)
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
