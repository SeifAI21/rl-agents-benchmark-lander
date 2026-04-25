"""Smoke tests for the plotting + markdown report pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluation.aggregate import discover_runs
from evaluation.plot import plot_all, write_index
from evaluation.report import write_markdown


def _fake_csv(path: Path, n_episodes: int, slope: float, noise: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rewards = -200.0 + slope * np.arange(n_episodes) + rng.normal(0, noise, size=n_episodes)
    losses = rng.uniform(0.1, 0.5, size=n_episodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "length", "loss"])
        for i, (r, l) in enumerate(zip(rewards, losses)):
            w.writerow([i, float(r), 100, float(l)])


def test_plot_all_writes_pngs(tmp_path: Path) -> None:
    log_dir = tmp_path / "results" / "logs"
    out_dir = tmp_path / "plots"
    _fake_csv(log_dir / "dqn_seed0.csv", 200, slope=2.0, noise=20.0, seed=0)
    _fake_csv(log_dir / "dqn_seed1.csv", 200, slope=1.8, noise=25.0, seed=1)
    _fake_csv(log_dir / "reinforce_seed0.csv", 200, slope=1.2, noise=30.0, seed=2)

    grouped = discover_runs(log_dir)
    plots = plot_all(grouped, out_dir, window=20, last_n=20)

    assert "comparison" in plots
    assert "final_performance" in plots
    assert any(k.startswith("curves_") for k in plots)
    for p in plots.values():
        assert Path(p).is_file()
        assert Path(p).stat().st_size > 0

    index = write_index(plots, out_dir)
    assert index.is_file()
    assert "comparison" in index.read_text()


def test_write_markdown_summary(tmp_path: Path) -> None:
    log_dir = tmp_path / "results" / "logs"
    _fake_csv(log_dir / "dqn_seed0.csv", 100, slope=4.0, noise=10.0, seed=0)
    _fake_csv(log_dir / "reinforce_seed0.csv", 100, slope=2.0, noise=10.0, seed=1)

    grouped = discover_runs(log_dir)
    md = write_markdown(grouped, tmp_path / "report.md", last_n=20, window=10)
    text = md.read_text()
    assert "RL Agents Benchmark" in text
    assert "dqn" in text.lower()
    assert "reinforce" in text.lower()


def test_write_markdown_no_runs(tmp_path: Path) -> None:
    md = write_markdown({}, tmp_path / "empty.md")
    text = md.read_text()
    assert "No runs found" in text
