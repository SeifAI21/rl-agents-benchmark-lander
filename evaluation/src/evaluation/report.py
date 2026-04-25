"""Generate a markdown report comparing all agents.

Pulls everything together: convergence speed, stability, final performance,
plus links to the PNG plots written by `plot.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .aggregate import (
    AlgoRuns,
    convergence_episode,
    final_performance,
    stability,
)


def _fmt(x: float, places: int = 2) -> str:
    if x != x:  # NaN
        return "—"
    return f"{x:.{places}f}"


def _algo_section(algo: str, runs: AlgoRuns, last_n: int, window: int) -> str:
    final = final_performance(runs, last_n=last_n)
    conv = convergence_episode(runs, threshold=200.0, window=window)
    stab = stability(runs, last_n=last_n)

    lines = [f"### {algo}", ""]
    lines.append(f"- seeds: `{runs.seeds}` ({runs.num_runs} runs)")
    lines.append(
        f"- final return (last {last_n} eps, mean over seeds): "
        f"**{_fmt(final['mean'])} ± {_fmt(final['std'])}**"
    )
    if conv["n_solved"] > 0:
        lines.append(
            f"- first episode where smoothed return ≥ {conv['threshold']:.0f}: "
            f"**{_fmt(conv['mean_episode'], 0)} ± {_fmt(conv['std_episode'], 0)}** "
            f"(solved {conv['n_solved']}/{conv['n_solved'] + conv['n_failed']} seeds)"
        )
    else:
        lines.append(
            f"- never reached the {conv['threshold']:.0f} threshold within the recorded episodes "
            f"({conv['n_failed']} seeds)"
        )
    lines.append(
        f"- per-seed std over last {last_n} eps (lower = more stable): "
        f"**{_fmt(stab['mean_std'])}** (max across seeds: {_fmt(stab.get('max_std', float('nan')))})"
    )
    lines.append("")
    return "\n".join(lines)


def _comparison_table(
    grouped: dict[str, AlgoRuns],
    last_n: int,
    window: int,
) -> str:
    header = (
        "| Algorithm | Seeds | Final return (mean ± std) | Convergence ep. (≥200) | "
        "Stability (per-seed std) |"
    )
    sep = "|---|---|---|---|---|"
    rows: list[str] = [header, sep]
    for algo in sorted(grouped, key=str.lower):
        runs = grouped[algo]
        if runs.num_runs == 0:
            continue
        final = final_performance(runs, last_n=last_n)
        conv = convergence_episode(runs, threshold=200.0, window=window)
        stab = stability(runs, last_n=last_n)
        conv_str = (
            f"{_fmt(conv['mean_episode'], 0)} ± {_fmt(conv['std_episode'], 0)} "
            f"({conv['n_solved']}/{conv['n_solved'] + conv['n_failed']})"
            if conv["n_solved"] > 0
            else f"never (0/{conv['n_failed']})"
        )
        rows.append(
            f"| {algo} | {runs.num_runs} | "
            f"{_fmt(final['mean'])} ± {_fmt(final['std'])} | "
            f"{conv_str} | {_fmt(stab['mean_std'])} |"
        )
    return "\n".join(rows)


def write_markdown(
    grouped: dict[str, AlgoRuns],
    out_path: str | Path,
    *,
    last_n: int = 100,
    window: int = 100,
    plots: Optional[dict[str, Path]] = None,
) -> Path:
    """Write a single-file markdown summary at `out_path`."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    parts.append("# RL Agents Benchmark — Evaluation Report\n")
    parts.append(
        "Auto-generated from `results/logs/*.csv`. Each algorithm should be trained "
        "on the same seeds for the same number of episodes through "
        "`src.env_utils.RunLogger` so this comparison stays fair.\n"
    )
    parts.append("## Summary\n")
    parts.append(_comparison_table(grouped, last_n=last_n, window=window))
    parts.append("\n")

    if plots:
        parts.append("## Plots\n")
        for label, path in plots.items():
            try:
                rel = Path(path).resolve().relative_to(out_path.parent.resolve())
            except ValueError:
                rel = Path(path)
            parts.append(f"- **{label}**: `{rel}`")
        parts.append("")

    parts.append("## Per-algorithm details\n")
    if not grouped:
        parts.append("_No runs found under `results/logs/`. Train at least one agent first._\n")
    else:
        for algo in sorted(grouped, key=str.lower):
            runs = grouped[algo]
            if runs.num_runs == 0:
                continue
            parts.append(_algo_section(algo, runs, last_n=last_n, window=window))

    parts.append("## Conclusions (fill in after reviewing plots)\n")
    parts.append("- Convergence speed:\n")
    parts.append("- Stability across seeds:\n")
    parts.append("- Final performance vs the 200 solved-threshold:\n")
    parts.append("- Practical wall-clock cost / sample efficiency:\n")

    out_path.write_text("\n".join(parts))
    return out_path
