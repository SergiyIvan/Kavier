#!/usr/bin/env python3
"""Plot mean queuing time (± 1 std across seeds) vs. percentage of patched jobs.

Three lines: "Baseline jobs" (-), "Patched jobs" (--), "All jobs" (:).
Fix-jobs (metadata "fix_job") are merged into their parent's metrics before aggregation.
Error bars show ± 1 std of per-seed group means across seeds.

Usage:
    python plot-exp-queuing-absolute.py <results_dir>

Output:
    exp-queuing-absolute.pdf  (written to the current working directory)
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
from plot_utils import PERCENTAGES, SEEDS, load_and_merge, group_mean_queuing  # noqa: E402


def compute_means(results_dir: Path) -> dict[str, np.ndarray]:
    """Return per-seed mean queuing time (h) arrays of shape (len(PERCENTAGES), len(SEEDS)).

    Keys: "baseline", "patched", "all".
    NaN when a group is empty for that (percentage, seed) — e.g. no patched jobs at X=0.
    """
    n_pct = len(PERCENTAGES)
    n_seed = len(SEEDS)
    data: dict[str, np.ndarray] = {
        "baseline": np.full((n_pct, n_seed), np.nan),
        "patched":  np.full((n_pct, n_seed), np.nan),
        "all":      np.full((n_pct, n_seed), np.nan),
    }

    for i, pct in enumerate(PERCENTAGES):
        for j, seed in enumerate(SEEDS):
            fname = results_dir / f"kavier-X{pct}-s{seed}_per_jobs.csv"
            if not fname.exists():
                sys.exit(f"ERROR: expected results file not found: {fname}")
            df = load_and_merge(fname)
            for group, mean_h in group_mean_queuing(df).items():
                if not np.isnan(mean_h):
                    data[group][i, j] = mean_h

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
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(data[key], axis=1)
            stds  = np.nanstd(data[key], axis=1, ddof=1)
            mins  = np.nanmin(data[key], axis=1)
            maxs  = np.nanmax(data[key], axis=1)

        valid = ~np.isnan(means)
        all_means.extend(maxs[valid].tolist())

        eb = ax.errorbar(
            np.array(PERCENTAGES)[valid],
            means[valid],
            yerr=stds[valid],
            label=label,
            linestyle=style,
            marker="o",
            capsize=4,
        )
        color = eb[0].get_color()
        lo = np.clip(means[valid] - mins[valid], 0, None)
        hi = np.clip(maxs[valid] - means[valid], 0, None)
        ax.errorbar(
            np.array(PERCENTAGES)[valid],
            means[valid],
            yerr=[lo, hi],
            linestyle="none",
            capsize=2,
            elinewidth=0.8,
            color=color,
        )

    ax.set_xlabel("Patched jobs (%)", fontsize=16)
    ax.set_ylabel("Mean queuing (h)", fontsize=16)
    ax.set_xlim(min(PERCENTAGES) - 5, max(PERCENTAGES) + 5)
    ax.set_ylim(0, max(all_means) * 1.15 if all_means else 1)
    ax.set_xticks(PERCENTAGES)
    ax.grid(True)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig("exp-queuing-absolute.pdf")


if __name__ == "__main__":
    main()
