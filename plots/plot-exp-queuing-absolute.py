#!/usr/bin/env python3
"""Plot mean queuing time vs. participation rate — five plots covering all experiment modes.

Each plot shows mean queuing time (h) on the Y axis and participation rate (%) on the X axis.
Seeded modes show ± 1 std error bars plus thin min/max whiskers (25 seeds).
Oracle modes show bare lines (deterministic, no seed variation).

Plots produced
--------------
1. exp-queuing-absolute-trad-priorities.pdf
   Traditional priorities baseline (all-baseline traces, priorities enabled).
   3 lines: Non-participant / Participant / All jobs.

2. exp-queuing-absolute-random-blend.pdf
   Random blend without priorities (our main approach).
   3 lines: Non-participant / Participant / All jobs.

3. exp-queuing-absolute-random-blend-priorities.pdf
   Random blend with priorities enabled.
   3 lines: Non-participant / Participant / All jobs.

4. exp-queuing-absolute-oracle.pdf
   Oracle blend without priorities (gpu_hours + gpu_number oracles combined).
   6 lines: one Non-participant / Participant / All triple per oracle variant.

5. exp-queuing-absolute-oracle-priorities.pdf
   Oracle blend with priorities enabled.
   6 lines: same structure as plot 4.

Usage:
    python plot-exp-queuing-absolute.py <results_dir>

Output:
    Five PDF files written to the current working directory.
"""

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import PERCENTAGES, SEEDS, load_and_merge, group_mean_queuing  # noqa: E402

# ---------------------------------------------------------------------------
# Series definitions
# ---------------------------------------------------------------------------

# (group_key, label, linestyle)
SEEDED_SERIES: List[Tuple[str, str, str]] = [
    ("non_participant", "Non-participant jobs", "-"),
    ("participant",     "Participant jobs",     "--"),
    ("all",             "All jobs",             ":"),
]


@dataclass
class OracleVariant:
    """One oracle source: a file-stem fragment and a short display tag."""
    stem: str    # e.g. "gpu_hours" — used in the filename pattern
    tag: str     # e.g. "GPU-hours" — appended to legend labels


ORACLE_VARIANTS: List[OracleVariant] = [
    OracleVariant(stem="gpu_hours",  tag="GPU-hours"),
    OracleVariant(stem="gpu_number", tag="GPU-count"),
]

# Line styles cycle: first variant gets solid/dashed/dotted, second gets
# dash-dot / loosely-dashed / densely-dotted so all 6 lines are distinct.
ORACLE_STYLES = ["-", "--", ":", "-.", (0, (5, 2)), (0, (1, 1))]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_seeded(results_dir: Path, pattern: str) -> dict:
    """Load seeded results into arrays of shape (len(PERCENTAGES), len(SEEDS)).

    ``pattern`` is a format string with ``{pct}`` and ``{seed}`` placeholders,
    e.g. ``"kavier-X{pct}-s{seed}_per_jobs.csv"``.

    Returns a dict keyed by group name with np.ndarray values.
    Groups: ``"non_participant"``, ``"participant"``, ``"all"``.
    NaN where a group is empty for that (pct, seed) pair.
    """
    n_pct = len(PERCENTAGES)
    n_seed = len(SEEDS)
    data: dict = {
        "non_participant": np.full((n_pct, n_seed), np.nan),
        "participant":     np.full((n_pct, n_seed), np.nan),
        "all":             np.full((n_pct, n_seed), np.nan),
    }
    for i, pct in enumerate(PERCENTAGES):
        for j, seed in enumerate(SEEDS):
            fname = results_dir / pattern.format(pct=pct, seed=seed)
            if not fname.exists():
                sys.exit(f"ERROR: expected results file not found: {fname}")
            df = load_and_merge(fname)
            for group, mean_h in group_mean_queuing(df).items():
                if not np.isnan(mean_h):
                    data[group][i, j] = mean_h
    return data


def load_oracle(results_dir: Path, pattern: str) -> dict:
    """Load oracle results into arrays of shape (len(PERCENTAGES),) — one value per X.

    ``pattern`` is a format string with a ``{pct}`` placeholder only,
    e.g. ``"kavier-X{pct}-gpu_hours_per_jobs.csv"``.

    Returns a dict keyed by group name with 1-D np.ndarray values (NaN where empty).
    """
    n_pct = len(PERCENTAGES)
    data: dict = {
        "non_participant": np.full(n_pct, np.nan),
        "participant":     np.full(n_pct, np.nan),
        "all":             np.full(n_pct, np.nan),
    }
    for i, pct in enumerate(PERCENTAGES):
        fname = results_dir / pattern.format(pct=pct)
        if not fname.exists():
            sys.exit(f"ERROR: expected results file not found: {fname}")
        df = load_and_merge(fname)
        for group, mean_h in group_mean_queuing(df).items():
            if not np.isnan(mean_h):
                data[group][i] = mean_h
    return data


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _add_seeded_series(
    ax: matplotlib.axes.Axes,
    pcts: np.ndarray,
    data: dict,
    series: List[Tuple[str, str, str]],
    all_maxs: list,
) -> None:
    """Draw mean ± std lines with min/max whiskers for one seeded dataset."""
    for key, label, style in series:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(data[key], axis=1)
            stds  = np.nanstd(data[key], axis=1, ddof=1)
            mins  = np.nanmin(data[key], axis=1)
            maxs  = np.nanmax(data[key], axis=1)

        valid = ~np.isnan(means)
        all_maxs.extend(maxs[valid].tolist())

        eb = ax.errorbar(
            pcts[valid], means[valid], yerr=stds[valid],
            label=label, linestyle=style, marker="o", capsize=4,
        )
        color = eb[0].get_color()
        lo = np.clip(means[valid] - mins[valid], 0, None)
        hi = np.clip(maxs[valid] - means[valid], 0, None)
        ax.errorbar(
            pcts[valid], means[valid], yerr=[lo, hi],
            linestyle="none", capsize=2, elinewidth=0.8, color=color,
        )


def _add_oracle_series(
    ax: matplotlib.axes.Axes,
    pcts: np.ndarray,
    data: dict,
    series: List[Tuple[str, str, str]],
    all_maxs: list,
) -> None:
    """Draw bare lines (no error bars) for one oracle dataset."""
    for key, label, style in series:
        values = data[key]
        valid = ~np.isnan(values)
        all_maxs.extend(values[valid].tolist())
        ax.plot(pcts[valid], values[valid], label=label, linestyle=style, marker="o")


def _finalise(ax: matplotlib.axes.Axes, all_maxs: list, title: Optional[str] = None) -> None:
    pcts = np.array(PERCENTAGES)
    ax.set_xlabel("Participation rate (%)", fontsize=16)
    ax.set_ylabel("Mean queuing time (h)", fontsize=16)
    ax.set_xlim(pcts.min() - 5, pcts.max() + 5)
    ax.set_ylim(0, max(all_maxs) * 1.15 if all_maxs else 1)
    ax.set_xticks(PERCENTAGES)
    ax.tick_params(labelsize=14)
    ax.grid(True)
    ax.legend(fontsize=14)
    if title:
        ax.set_title(title, fontsize=14)


# ---------------------------------------------------------------------------
# The five plot generators
# ---------------------------------------------------------------------------

def plot_trad_priorities(results_dir: Path) -> None:
    """Plot 1: traditional priorities baseline (all-baseline traces, priorities on)."""
    pattern = "kavier-all-baseline-X{pct}-s{seed}-priorities_per_jobs.csv"
    data = load_seeded(results_dir, pattern)

    fig, ax = plt.subplots(figsize=(10, 4))
    all_maxs: list = []
    _add_seeded_series(ax, np.array(PERCENTAGES), data, SEEDED_SERIES, all_maxs)
    _finalise(ax, all_maxs, title="Traditional priorities baseline")
    fig.tight_layout()
    fig.savefig("exp-queuing-absolute-trad-priorities.pdf")
    plt.close(fig)
    print("Written: exp-queuing-absolute-trad-priorities.pdf")


def plot_random_blend(results_dir: Path) -> None:
    """Plot 2: random blend, no priorities."""
    pattern = "kavier-X{pct}-s{seed}_per_jobs.csv"
    data = load_seeded(results_dir, pattern)

    fig, ax = plt.subplots(figsize=(10, 4))
    all_maxs: list = []
    _add_seeded_series(ax, np.array(PERCENTAGES), data, SEEDED_SERIES, all_maxs)
    _finalise(ax, all_maxs, title="Random blend (no priorities)")
    fig.tight_layout()
    fig.savefig("exp-queuing-absolute-random-blend.pdf")
    plt.close(fig)
    print("Written: exp-queuing-absolute-random-blend.pdf")


def plot_random_blend_priorities(results_dir: Path) -> None:
    """Plot 3: random blend with priorities enabled."""
    pattern = "kavier-X{pct}-s{seed}-priorities_per_jobs.csv"
    data = load_seeded(results_dir, pattern)

    fig, ax = plt.subplots(figsize=(10, 4))
    all_maxs: list = []
    _add_seeded_series(ax, np.array(PERCENTAGES), data, SEEDED_SERIES, all_maxs)
    _finalise(ax, all_maxs, title="Random blend + priorities")
    fig.tight_layout()
    fig.savefig("exp-queuing-absolute-random-blend-priorities.pdf")
    plt.close(fig)
    print("Written: exp-queuing-absolute-random-blend-priorities.pdf")


def _plot_oracle(results_dir: Path, priorities: bool) -> None:
    """Shared logic for plots 4 and 5 (oracle, with/without priorities)."""
    suffix = "-priorities" if priorities else ""
    out_name = "exp-queuing-absolute-oracle-priorities.pdf" if priorities else "exp-queuing-absolute-oracle.pdf"
    title = "Oracle blend + priorities" if priorities else "Oracle blend"

    pcts = np.array(PERCENTAGES)
    fig, ax = plt.subplots(figsize=(10, 4))
    all_maxs: list = []

    style_iter = iter(ORACLE_STYLES)
    for variant in ORACLE_VARIANTS:
        pattern = f"kavier-X{{pct}}-{variant.stem}{suffix}_per_jobs.csv"
        data = load_oracle(results_dir, pattern)
        series = [
            (key, f"{label} ({variant.tag})", next(style_iter))
            for key, label, _ in SEEDED_SERIES
        ]
        _add_oracle_series(ax, pcts, data, series, all_maxs)

    _finalise(ax, all_maxs, title=title)
    fig.tight_layout()
    fig.savefig(out_name)
    plt.close(fig)
    print(f"Written: {out_name}")


def plot_oracle(results_dir: Path) -> None:
    """Plot 4: oracle blend, no priorities."""
    _plot_oracle(results_dir, priorities=False)


def plot_oracle_priorities(results_dir: Path) -> None:
    """Plot 5: oracle blend with priorities enabled."""
    _plot_oracle(results_dir, priorities=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {results_dir}")

    plot_trad_priorities(results_dir)
    plot_random_blend(results_dir)
    plot_random_blend_priorities(results_dir)
    plot_oracle(results_dir)
    plot_oracle_priorities(results_dir)


if __name__ == "__main__":
    main()
