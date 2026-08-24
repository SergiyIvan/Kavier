#!/usr/bin/env python3
"""Plot mean JCT (± 1 std across seeds) vs. percentage of patched jobs.

Three lines are shown: metrics for jobs tagged "baseline", jobs tagged "patched",
and all jobs together.  Error bars show ± 1 std of the per-seed means.

Usage:
    python plot-exp-jct.py <results_dir>

Output:
    exp-jct.pdf  (written to the current working directory)
"""

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ITERATION_COUNT = 25
PERCENTAGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEEDS = list(range(1, ITERATION_COUNT + 1))

VALID_METADATA = {"baseline", "patched"}


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load a per-jobs CSV and abort if any job_metadata value is not 'baseline' or 'patched'."""
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


def compute_means(results_dir: Path) -> dict[str, np.ndarray]:
    """Return per-seed mean JCT (h) arrays of shape (len(PERCENTAGES), len(SEEDS)).

    Keys: "baseline", "patched", "all".
    A cell is NaN when the group is empty for that (percentage, seed) — e.g. no patched jobs
    at X=0.  NaN cells are silently skipped when aggregating across seeds.
    """
    n_pct = len(PERCENTAGES)
    n_seed = len(SEEDS)
    data: dict[str, np.ndarray] = {
        "baseline": np.full((n_pct, n_seed), np.nan),
        "patched": np.full((n_pct, n_seed), np.nan),
        "all": np.full((n_pct, n_seed), np.nan),
    }

    for i, pct in enumerate(PERCENTAGES):
        for j, seed in enumerate(SEEDS):
            fname = results_dir / f"kavier-X{pct}-s{seed}_per_jobs.csv"
            if not fname.exists():
                sys.exit(f"ERROR: expected results file not found: {fname}")
            df = load_and_validate(fname)
            jct_h = df["turnaround_s"] / 3600

            data["all"][i, j] = jct_h.mean()

            for group in ("baseline", "patched"):
                subset = jct_h[df["job_metadata"] == group]
                if not subset.empty:
                    data[group][i, j] = subset.mean()
                # else: stays NaN

    return data


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {results_dir}")

    data = compute_means(results_dir)

    fig, ax = plt.subplots(figsize=(10, 4))

    series = [
        ("baseline", "Baseline jobs", "-"),
        ("patched",  "Patched jobs",  "--"),
        ("all",      "All jobs",      ":"),
    ]

    all_means = []
    for key, label, style in series:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # silence all-NaN slice warnings
            means = np.nanmean(data[key], axis=1)   # shape: (n_pct,)
            stds  = np.nanstd(data[key], axis=1, ddof=1)

        valid = ~np.isnan(means)
        all_means.extend(means[valid].tolist())

        ax.errorbar(
            np.array(PERCENTAGES)[valid],
            means[valid],
            yerr=stds[valid],
            label=label,
            linestyle=style,
            marker="o",
            capsize=4,
        )

    ax.set_xlabel("Patched jobs (%)", fontsize=16)
    ax.set_ylabel("Mean JCT (h)", fontsize=16)
    ax.set_xlim(min(PERCENTAGES) - 5, max(PERCENTAGES) + 5)
    ax.set_ylim(0, max(all_means) * 1.15 if all_means else 1)
    ax.set_xticks(PERCENTAGES)
    ax.grid(True)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig("exp-jct-absolute.pdf")


if __name__ == "__main__":
    main()
