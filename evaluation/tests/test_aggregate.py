"""Unit tests for the CSV aggregator."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from evaluation.aggregate import (
    convergence_episode,
    discover_runs,
    final_performance,
    load_run,
    moving_average,
    stability,
    summarize_runs,
)


def _write_csv(path: Path, rewards, *, losses=None, lengths=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "length", "loss"])
        for i, r in enumerate(rewards):
            ln = lengths[i] if lengths is not None else 100
            ls = "" if losses is None else losses[i]
            w.writerow([i, r, ln, ls])


def test_load_run_parses_filename(tmp_path: Path) -> None:
    csv_path = tmp_path / "dqn_seed42.csv"
    _write_csv(csv_path, [1.0, 2.0, 3.0], losses=[0.1, "", 0.3])

    run = load_run(csv_path)
    assert run.algo == "dqn"
    assert run.seed == 42
    assert run.num_episodes == 3
    np.testing.assert_array_equal(run.episode, [0, 1, 2])
    np.testing.assert_array_equal(run.reward, [1.0, 2.0, 3.0])
    assert np.isnan(run.loss[1])  # blank cell -> NaN


def test_load_run_rejects_bad_header(tmp_path: Path) -> None:
    bad = tmp_path / "dqn_seed1.csv"
    with open(bad, "w") as f:
        f.write("ep,ret\n0,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_run(bad)


def test_discover_runs_groups_by_algo(tmp_path: Path) -> None:
    _write_csv(tmp_path / "dqn_seed0.csv", [1, 2])
    _write_csv(tmp_path / "dqn_seed1.csv", [3, 4])
    _write_csv(tmp_path / "reinforce_seed42.csv", [5, 6])
    _write_csv(tmp_path / "ignored.csv", [9, 9])  # not matching

    grouped = discover_runs(tmp_path)
    assert set(grouped) == {"dqn", "reinforce"}
    assert grouped["dqn"].seeds == [0, 1]
    assert grouped["reinforce"].seeds == [42]


def test_discover_runs_with_algo_filter(tmp_path: Path) -> None:
    _write_csv(tmp_path / "dqn_seed0.csv", [1, 2])
    _write_csv(tmp_path / "reinforce_seed0.csv", [3, 4])

    grouped = discover_runs(tmp_path, algos=["dqn"])
    assert set(grouped) == {"dqn"}


def test_moving_average_progressive_window() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ma = moving_average(arr, window=3)
    np.testing.assert_allclose(ma, [1.0, 1.5, 2.0, 3.0, 4.0])


def test_moving_average_window_one_is_identity() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(moving_average(arr, window=1), arr)


def test_summarize_runs_aligns_to_min_length(tmp_path: Path) -> None:
    _write_csv(tmp_path / "dqn_seed0.csv", list(range(10)))
    _write_csv(tmp_path / "dqn_seed1.csv", list(range(20)))

    grouped = discover_runs(tmp_path)
    summary = summarize_runs(grouped["dqn"], window=2)
    assert summary["mean"].shape == (10,)
    assert summary["std"].shape == (10,)
    assert summary["episode"].tolist() == list(range(10))


def test_final_performance(tmp_path: Path) -> None:
    _write_csv(tmp_path / "dqn_seed0.csv", [0.0] * 90 + [200.0] * 10)
    _write_csv(tmp_path / "dqn_seed1.csv", [0.0] * 90 + [100.0] * 10)

    grouped = discover_runs(tmp_path)
    final = final_performance(grouped["dqn"], last_n=10)
    assert final["mean"] == pytest.approx(150.0)
    assert final["n_seeds"] == 2


def test_convergence_episode_picks_first_above_threshold(tmp_path: Path) -> None:
    rewards = list(np.linspace(-100, 250, 50))
    _write_csv(tmp_path / "dqn_seed0.csv", rewards)

    grouped = discover_runs(tmp_path)
    conv = convergence_episode(grouped["dqn"], threshold=200.0, window=5)
    assert conv["n_solved"] == 1
    assert conv["mean_episode"] >= 0


def test_convergence_episode_no_solution(tmp_path: Path) -> None:
    _write_csv(tmp_path / "dqn_seed0.csv", [-100.0] * 50)
    grouped = discover_runs(tmp_path)
    conv = convergence_episode(grouped["dqn"], threshold=200.0, window=5)
    assert conv["n_solved"] == 0
    assert conv["n_failed"] == 1


def test_stability_lower_is_more_stable(tmp_path: Path) -> None:
    _write_csv(tmp_path / "dqn_seed0.csv", [100.0] * 100)
    _write_csv(tmp_path / "dqn_seed1.csv", [100.0] * 100)
    grouped = discover_runs(tmp_path)
    s = stability(grouped["dqn"], last_n=20)
    assert s["mean_std"] == pytest.approx(0.0, abs=1e-9)
