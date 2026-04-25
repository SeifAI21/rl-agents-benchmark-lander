from dataclasses import dataclass

from src.env_utils import (
    DEFAULT_ENV_NAME,
    DEFAULT_MAX_EPISODES,
    DEFAULT_MAX_STEPS_PER_EPISODE,
)


@dataclass
class DQNConfig:
    env_name: str = DEFAULT_ENV_NAME
    seed: int = 42
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    buffer_size: int = 10_000
    target_update_episodes: int = 10
    num_episodes: int = DEFAULT_MAX_EPISODES
    max_steps: int = DEFAULT_MAX_STEPS_PER_EPISODE
    hidden_size: int = 64
    device: str | None = None
    early_stop_solved: bool = False
    print_every_episodes: int = 100
    enable_tensorboard: bool = True
