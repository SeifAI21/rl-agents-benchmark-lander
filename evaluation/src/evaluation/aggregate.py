"""Load per-episode CSV logs from each agent and compute comparison statistics.

The contract (from `src/env_utils.RunLogger`):

    results/logs/{algo}_seed{seed}.csv

with columns:

    episode, reward, length, loss

`loss` may be blank for algorithms that don't track per-episode loss.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

LOG_FILENAME_RE = re.compile(r"^(?P<algo>[A-Za-z0-9_-]+?)_seed(?P<seed>-?\d+)\.csv$")


# ---------------------------------------------------------------------------
# Data containers.
# ---------------------------------------------------------------------------


@dataclass
class RunData:
    """One training run for a single (algorithm, seed) pair."""

    algo: str
    seed: int
    path: Path
    episode: np.ndarray  # shape (T,)
    reward: np.ndarray   # shape (T,)
    length: np.ndarray   # shape (T,)
    loss: np.ndarray     # shape (T,), NaN where the agent didn't log loss

    @property
    def num_episodes(self) -> int:
        return int(self.reward.shape[0])


@dataclass
class AlgoRuns:
    """All runs we found for one algorithm."""

    algo: str
    runs: list[RunData] = field(default_factory=list)

    @property
    def seeds(self) -> list[int]:
        return [r.seed for r in self.runs]

    @property
    def num_runs(self) -> int:
        return len(self.runs)


# ---------------------------------------------------------------------------
# CSV loading.
# ---------------------------------------------------------------------------


def _coerce_float(value: str) -> float:
    """Empty cell -> NaN, otherwise float."""
    if value is None:
        return float("nan")
    s = value.strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_run(path: str | Path, algo: Optional[str] = None, seed: Optional[int] = None) -> RunData:
    """Load one CSV file into a `RunData`.

    `algo` / `seed` are inferred from the filename if not supplied.
    """
    path = Path(path)
    if algo is None or seed is None:
        match = LOG_FILENAME_RE.match(path.name)
        if match is None:
            raise ValueError(
                f"Cannot infer algo/seed from {path.name!r}; "
                "expected '{algo}_seed{seed}.csv' or pass algo/seed explicitly."
            )
        algo = algo or match.group("algo")
        seed = seed if seed is not None else int(match.group("seed"))

    episodes: list[int] = []
    rewards: list[float] = []
    lengths: list[int] = []
    losses: list[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"episode", "reward", "length", "loss"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            episodes.append(int(float(row["episode"])))
            rewards.append(float(row["reward"]))
            lengths.append(int(float(row["length"])))
            losses.append(_coerce_float(row["loss"]))

    return RunData(
        algo=algo,
        seed=int(seed),
        path=path,
        episode=np.asarray(episodes, dtype=np.int64),
        reward=np.asarray(rewards, dtype=np.float64),
        length=np.asarray(lengths, dtype=np.int64),
        loss=np.asarray(losses, dtype=np.float64),
    )


def discover_runs(
    log_dir: str | Path = "results/logs",
    algos: Optional[Iterable[str]] = None,
) -> dict[str, AlgoRuns]:
    """Find every `{algo}_seed{seed}.csv` under `log_dir` and group by algo.

    Pass `algos` to keep only specific algorithms (case-insensitive match on
    the inferred algo name).
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return {}

    keep = {a.lower() for a in algos} if algos else None
    grouped: dict[str, AlgoRuns] = {}

    for csv_file in sorted(log_dir.glob("*_seed*.csv")):
        try:
            run = load_run(csv_file)
        except ValueError:
            continue
        if keep is not None and run.algo.lower() not in keep:
            continue
        grouped.setdefault(run.algo, AlgoRuns(algo=run.algo)).runs.append(run)

    for ar in grouped.values():
        ar.runs.sort(key=lambda r: r.seed)
    return grouped


# ---------------------------------------------------------------------------
# Smoothing and summary statistics.
# ---------------------------------------------------------------------------


def moving_average(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Causal moving average. For the first `window-1` points the average
    is over however many are available, so the curve starts at episode 0.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    if window <= 1:
        return arr.copy()
    cumulative = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.empty_like(arr)
    for i in range(arr.size):
        start = max(0, i - window + 1)
        n = (i + 1) - start
        out[i] = (cumulative[i + 1] - cumulative[start]) / n
    return out


def _stack_to_min_length(arrays: list[np.ndarray]) -> np.ndarray:
    """Truncate every array to the shortest length and stack as (R, T)."""
    if not arrays:
        return np.empty((0, 0))
    min_len = min(a.shape[0] for a in arrays)
    if min_len == 0:
        return np.empty((len(arrays), 0))
    return np.stack([a[:min_len] for a in arrays], axis=0)


def summarize_runs(algo_runs: AlgoRuns, window: int = 100) -> dict[str, np.ndarray]:
    """Return mean / std / min / max per episode across seeds.

    Each curve is first smoothed per-seed with a `window`-episode moving
    average, then aggregated across seeds. Runs are truncated to the
    shortest seed's episode count so all curves align.
    """
    smoothed = [moving_average(r.reward, window=window) for r in algo_runs.runs]
    stack = _stack_to_min_length(smoothed)
    if stack.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return {
            "episode": empty,
            "mean": empty,
            "std": empty,
            "min": empty,
            "max": empty,
        }
    return {
        "episode": np.arange(stack.shape[1], dtype=np.int64),
        "mean": stack.mean(axis=0),
        "std": stack.std(axis=0, ddof=0),
        "min": stack.min(axis=0),
        "max": stack.max(axis=0),
    }


def final_performance(algo_runs: AlgoRuns, last_n: int = 100) -> dict[str, float]:
    """Mean / std of the per-seed *raw* return averaged over the last `last_n`
    episodes. Reported in the report as 'final performance'.
    """
    per_seed_means: list[float] = []
    for r in algo_runs.runs:
        if r.reward.size == 0:
            continue
        tail = r.reward[-last_n:]
        per_seed_means.append(float(np.mean(tail)))
    if not per_seed_means:
        return {"mean": float("nan"), "std": float("nan"), "n_seeds": 0}
    arr = np.asarray(per_seed_means, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n_seeds": len(per_seed_means),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def convergence_episode(
    algo_runs: AlgoRuns,
    threshold: float = 200.0,
    window: int = 100,
) -> dict[str, float]:
    """First episode where the moving-average return >= `threshold`,
    averaged over seeds. Seeds that never reach the threshold are excluded
    from the mean and counted in `n_failed`.
    """
    convs: list[int] = []
    n_failed = 0
    for r in algo_runs.runs:
        if r.reward.size == 0:
            n_failed += 1
            continue
        smoothed = moving_average(r.reward, window=window)
        idx = np.argmax(smoothed >= threshold)
        if smoothed[idx] >= threshold:
            convs.append(int(r.episode[idx]) if idx < r.episode.size else int(idx))
        else:
            n_failed += 1
    if not convs:
        return {
            "mean_episode": float("nan"),
            "std_episode": float("nan"),
            "n_solved": 0,
            "n_failed": n_failed,
            "threshold": float(threshold),
        }
    arr = np.asarray(convs, dtype=np.float64)
    return {
        "mean_episode": float(arr.mean()),
        "std_episode": float(arr.std(ddof=0)),
        "n_solved": len(convs),
        "n_failed": n_failed,
        "threshold": float(threshold),
    }


def stability(algo_runs: AlgoRuns, last_n: int = 100) -> dict[str, float]:
    """Per-seed standard deviation of the last `last_n` episode returns,
    averaged across seeds. Lower = more stable.
    """
    stds: list[float] = []
    for r in algo_runs.runs:
        if r.reward.size == 0:
            continue
        tail = r.reward[-last_n:]
        if tail.size > 1:
            stds.append(float(np.std(tail, ddof=1)))
    if not stds:
        return {"mean_std": float("nan"), "n_seeds": 0}
    arr = np.asarray(stds, dtype=np.float64)
    return {"mean_std": float(arr.mean()), "max_std": float(arr.max()), "n_seeds": len(stds)}
