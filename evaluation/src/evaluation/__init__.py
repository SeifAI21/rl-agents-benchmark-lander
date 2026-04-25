"""Evaluation & visualization package (Person 5).

Reads the per-episode CSV logs each agent writes via `src.env_utils.RunLogger`
and the saved checkpoints (DQN, REINFORCE, A2C), then produces:

- combined reward-vs-episode plots with moving averages
- per-agent training curves and loss plots
- a markdown summary comparing convergence speed, stability, final return
- a "game" viewer: load any agent's checkpoint and watch it play live,
  optionally recording a GIF

The data contract is fixed by `src/env_utils.py`:

    results/logs/{algo}_seed{seed}.csv -> columns: episode,reward,length,loss

Anything that follows that contract slots into this module without changes.
"""

from .aggregate import (
    AlgoRuns,
    RunData,
    convergence_episode,
    discover_runs,
    final_performance,
    load_run,
    moving_average,
    summarize_runs,
)
from .adapters import AGENT_REGISTRY, BaseAdapter, build_adapter

__all__ = [
    "AGENT_REGISTRY",
    "AlgoRuns",
    "BaseAdapter",
    "RunData",
    "build_adapter",
    "convergence_episode",
    "discover_runs",
    "final_performance",
    "load_run",
    "moving_average",
    "summarize_runs",
]
