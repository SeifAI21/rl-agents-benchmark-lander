"""
Shared env factory, seeding, and logging.

Every agent (DQN, REINFORCE, A2C) imports from here so runs stay comparable.

What's in here:
- DEFAULT_* constants   : env name, seeds, episode budget, log paths
- set_seed(seed)        : seeds python / numpy / torch
- get_device()          : returns cuda if available, else cpu
- make_env(...)         : builds the env and returns (env, EnvInfo)
- RunLogger             : writes episode stats to CSV and TensorBoard
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required. Install: pip install torch") from e

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError("Gymnasium is required. Install: pip install 'gymnasium[box2d]'") from e


# ---------------------------------------------------------------------------
# Shared constants. Agree with the team before changing.
# ---------------------------------------------------------------------------

DEFAULT_ENV_NAME: str = "LunarLander-v3"
DEFAULT_SEEDS: list[int] = [0, 1, 2, 42, 123]
DEFAULT_MAX_EPISODES: int = 1000
DEFAULT_MAX_STEPS_PER_EPISODE: int = 1000  # LunarLander's built-in limit
DEFAULT_LOG_DIR: str = "results/logs"
DEFAULT_TB_DIR: str = "results/tensorboard"


# ---------------------------------------------------------------------------
# Seeding and device.
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed python, numpy, torch. Call once per run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """cuda if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Environment factory.
# ---------------------------------------------------------------------------

@dataclass
class EnvInfo:
    """Info each agent needs to size its networks."""

    env_name: str
    obs_dim: int
    is_discrete: bool
    n_actions: Optional[int]          # set if discrete
    action_dim: Optional[int]         # set if continuous
    action_low: Optional[np.ndarray]  # set if continuous
    action_high: Optional[np.ndarray] # set if continuous
    max_episode_steps: int
    seed: int


def make_env(
    env_name: str = DEFAULT_ENV_NAME,
    seed: int = 42,
    continuous: bool = False,
    render_mode: Optional[str] = None,
    record_stats: bool = True,
) -> tuple[gym.Env, EnvInfo]:
    """
    Build the env and return (env, EnvInfo).

    continuous=True gives the 2-d thrust variant (A2C only; DQN needs False).
    render_mode: None (fastest), "rgb_array", or "human".

    Step returns (obs, reward, terminated, truncated, info).
    done = terminated or truncated. Only `terminated` zeros bootstrap targets —
    truncation is a time limit, not a real terminal.
    Reset returns (obs, info).
    """
    env = gym.make(env_name, continuous=continuous, render_mode=render_mode)

    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)

    # Seed the env and its spaces so .sample() is reproducible.
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    obs_dim = int(np.prod(env.observation_space.shape))
    max_steps = getattr(env.spec, "max_episode_steps", None) or DEFAULT_MAX_STEPS_PER_EPISODE

    info = EnvInfo(
        env_name=env_name,
        obs_dim=obs_dim,
        is_discrete=is_discrete,
        n_actions=int(env.action_space.n) if is_discrete else None,
        action_dim=int(env.action_space.shape[0]) if not is_discrete else None,
        action_low=None if is_discrete else env.action_space.low.astype(np.float32),
        action_high=None if is_discrete else env.action_space.high.astype(np.float32),
        max_episode_steps=int(max_steps),
        seed=seed,
    )
    return env, info


# ---------------------------------------------------------------------------
# Logging: CSV (for plots) + TensorBoard (for live curves).
# ---------------------------------------------------------------------------

LOG_COLUMNS = ["episode", "reward", "length", "loss"]


def csv_path(algo: str, seed: int, log_dir: str = DEFAULT_LOG_DIR) -> str:
    """Path: results/logs/{algo}_seed{seed}.csv"""
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{algo}_seed{seed}.csv")


def tb_dir(algo: str, seed: int, tb_root: str = DEFAULT_TB_DIR) -> str:
    """Path: results/tensorboard/{algo}/seed{seed}/"""
    path = os.path.join(tb_root, algo, f"seed{seed}")
    os.makedirs(path, exist_ok=True)
    return path


# Old name, kept for compatibility.
log_path = csv_path


class RunLogger:
    """
    Writes to CSV and TensorBoard at the same time.

    Example:
        with RunLogger("dqn", seed=42) as logger:
            for ep in range(1000):
                ...
                logger.log_episode(ep, reward=ret, length=length, loss=loss)
                logger.log_scalar("train/epsilon", eps, step=global_step)
    """

    def __init__(
        self,
        algo: str,
        seed: int,
        log_dir: str = DEFAULT_LOG_DIR,
        tb_root: str = DEFAULT_TB_DIR,
        enable_tb: bool = True,
    ) -> None:
        self.algo = algo
        self.seed = seed
        self.csv_file = csv_path(algo, seed, log_dir)

        # Write the header (overwrites any previous run for this algo+seed).
        with open(self.csv_file, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)

        self._writer = None
        if enable_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise ImportError(
                    "TensorBoard requires the `tensorboard` package. "
                    "Install: pip install tensorboard"
                ) from e
            self._writer = SummaryWriter(log_dir=tb_dir(algo, seed, tb_root))

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        loss: Optional[float] = None,
    ) -> None:
        """One row to CSV, same values mirrored to TensorBoard."""
        with open(self.csv_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [episode, float(reward), int(length), "" if loss is None else float(loss)]
            )
        if self._writer is not None:
            self._writer.add_scalar("episode/reward", float(reward), episode)
            self._writer.add_scalar("episode/length", int(length), episode)
            if loss is not None:
                self._writer.add_scalar("episode/loss", float(loss), episode)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Extra scalar for TensorBoard only (e.g. epsilon, grad norm)."""
        if self._writer is not None:
            self._writer.add_scalar(tag, float(value), step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
