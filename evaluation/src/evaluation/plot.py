"""Matplotlib plots for the evaluation report.

All public functions accept the dicts produced by `aggregate.discover_runs`
and write PNG files into `out_dir`. The matplotlib backend is forced to
`Agg` so plots work over SSH / inside CI without a display.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .aggregate import AlgoRuns, moving_average, summarize_runs

DEFAULT_PALETTE = {
    "dqn": "#1f77b4",
    "reinforce": "#ff7f0e",
    "a2c": "#2ca02c",
}
SOLVED_THRESHOLD = 200.0


def _ensure_dir(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _color_for(algo: str, palette: dict[str, str], idx: int) -> str:
    if algo.lower() in palette:
        return palette[algo.lower()]
    fallback = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    return fallback[idx % len(fallback)]


# ---------------------------------------------------------------------------
# Per-algorithm plots.
# ---------------------------------------------------------------------------


def plot_algo_curves(
    algo_runs: AlgoRuns,
    out_dir: str | Path,
    *,
    window: int = 100,
    color: Optional[str] = None,
) -> Path:
    """One PNG with two panels:

    - top    : per-seed raw return + smoothed mean across seeds
    - bottom : per-seed loss (skipped if every cell is NaN)
    """
    out = _ensure_dir(out_dir)
    color = color or DEFAULT_PALETTE.get(algo_runs.algo.lower(), "tab:blue")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_r, ax_l = axes

    # Top panel: rewards.
    for r in algo_runs.runs:
        ax_r.plot(
            r.episode,
            r.reward,
            color=color,
            alpha=0.18,
            linewidth=0.7,
            label=f"seed {r.seed} (raw)" if False else None,
        )
        smoothed = moving_average(r.reward, window=window)
        ax_r.plot(
            r.episode,
            smoothed,
            color=color,
            alpha=0.55,
            linewidth=1.0,
            label=f"seed {r.seed}",
        )

    summary = summarize_runs(algo_runs, window=window)
    if summary["episode"].size > 0:
        ax_r.plot(
            summary["episode"],
            summary["mean"],
            color="black",
            linewidth=2.2,
            label=f"mean of {algo_runs.num_runs} seeds (window={window})",
        )
        ax_r.fill_between(
            summary["episode"],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            color="black",
            alpha=0.12,
        )

    ax_r.axhline(SOLVED_THRESHOLD, color="green", linestyle="--", linewidth=1.0,
                 label=f"solved = {SOLVED_THRESHOLD:.0f}")
    ax_r.set_ylabel("Episode return")
    ax_r.set_title(f"{algo_runs.algo} — training rewards (smoothed window={window})")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="lower right", fontsize=8, ncol=2)

    # Bottom panel: losses (only if any agent logged them).
    any_loss = any(np.isfinite(r.loss).any() for r in algo_runs.runs)
    if any_loss:
        for r in algo_runs.runs:
            mask = np.isfinite(r.loss)
            if not mask.any():
                continue
            smoothed = moving_average(np.where(mask, r.loss, 0.0), window=window)
            ax_l.plot(r.episode, smoothed, color=color, alpha=0.55, linewidth=1.0,
                      label=f"seed {r.seed}")
        ax_l.set_ylabel("Loss (smoothed)")
        ax_l.set_title(f"{algo_runs.algo} — training loss (window={window})")
        ax_l.grid(True, alpha=0.3)
        ax_l.legend(loc="upper right", fontsize=8, ncol=2)
    else:
        ax_l.text(
            0.5, 0.5,
            f"no per-episode loss logged for {algo_runs.algo}",
            ha="center", va="center", transform=ax_l.transAxes, fontsize=10, color="gray",
        )
        ax_l.set_axis_off()

    ax_l.set_xlabel("Episode")

    fig.tight_layout()
    out_path = out / f"curves_{algo_runs.algo}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Comparison plot.
# ---------------------------------------------------------------------------


def plot_comparison(
    grouped: dict[str, AlgoRuns],
    out_dir: str | Path,
    *,
    window: int = 100,
    palette: Optional[dict[str, str]] = None,
    filename: str = "comparison_rewards.png",
) -> Path:
    """One panel: smoothed mean ± std across seeds for every algo, on one axis."""
    out = _ensure_dir(out_dir)
    palette = palette or DEFAULT_PALETTE

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (algo, runs) in enumerate(sorted(grouped.items(), key=lambda kv: kv[0].lower())):
        if runs.num_runs == 0:
            continue
        color = _color_for(algo, palette, idx)
        summary = summarize_runs(runs, window=window)
        if summary["episode"].size == 0:
            continue
        ax.plot(
            summary["episode"],
            summary["mean"],
            color=color,
            linewidth=2.0,
            label=f"{algo} (n={runs.num_runs})",
        )
        ax.fill_between(
            summary["episode"],
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            color=color,
            alpha=0.18,
        )

    ax.axhline(SOLVED_THRESHOLD, color="green", linestyle="--", linewidth=1.0,
               label=f"solved = {SOLVED_THRESHOLD:.0f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Smoothed return (window={window})")
    ax.set_title("Algorithm comparison — Lunar Lander")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path = out / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Bar chart of final performance.
# ---------------------------------------------------------------------------


def plot_final_performance(
    grouped: dict[str, AlgoRuns],
    out_dir: str | Path,
    *,
    last_n: int = 100,
    palette: Optional[dict[str, str]] = None,
    filename: str = "final_performance.png",
) -> Path:
    """Bar chart: mean per-seed return over the last `last_n` episodes,
    error bars = std across seeds."""
    out = _ensure_dir(out_dir)
    palette = palette or DEFAULT_PALETTE

    algos: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    colors: list[str] = []

    for idx, (algo, runs) in enumerate(sorted(grouped.items(), key=lambda kv: kv[0].lower())):
        per_seed = []
        for r in runs.runs:
            if r.reward.size == 0:
                continue
            per_seed.append(float(np.mean(r.reward[-last_n:])))
        if not per_seed:
            continue
        algos.append(algo)
        means.append(float(np.mean(per_seed)))
        stds.append(float(np.std(per_seed, ddof=0)))
        colors.append(_color_for(algo, palette, idx))

    fig, ax = plt.subplots(figsize=(7, 5))
    if algos:
        x = np.arange(len(algos))
        ax.bar(x, means, yerr=stds, color=colors, capsize=6, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.axhline(SOLVED_THRESHOLD, color="green", linestyle="--", linewidth=1.0,
                   label=f"solved = {SOLVED_THRESHOLD:.0f}")
        ax.legend(loc="lower right")
    ax.set_ylabel(f"Mean return over last {last_n} episodes")
    ax.set_title("Final performance (averaged across seeds)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Convenience: plot everything.
# ---------------------------------------------------------------------------


def plot_all(
    grouped: dict[str, AlgoRuns],
    out_dir: str | Path,
    *,
    window: int = 100,
    last_n: int = 100,
    algos: Optional[Iterable[str]] = None,
) -> dict[str, Path]:
    """Make per-algo plots + the comparison + the final-performance bar chart.

    Returns `{label: path}` for everything written.
    """
    out = _ensure_dir(out_dir)
    selected = grouped
    if algos is not None:
        keep = {a.lower() for a in algos}
        selected = {k: v for k, v in grouped.items() if k.lower() in keep}

    written: dict[str, Path] = {}
    for algo, runs in selected.items():
        if runs.num_runs == 0:
            continue
        written[f"curves_{algo}"] = plot_algo_curves(runs, out, window=window)
    if selected:
        written["comparison"] = plot_comparison(selected, out, window=window)
        written["final_performance"] = plot_final_performance(selected, out, last_n=last_n)
    return written


def write_index(written: dict[str, Path], out_dir: str | Path) -> Path:
    """Tiny `plots.txt` index so the CLI / Make targets show what was made."""
    out = _ensure_dir(out_dir)
    index = out / "plots.txt"
    with open(index, "w") as f:
        for label, path in written.items():
            rel = os.path.relpath(path, out)
            f.write(f"{label}: {rel}\n")
    return index
