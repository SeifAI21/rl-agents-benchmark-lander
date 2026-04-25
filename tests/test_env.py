"""
Sanity check for env_utils.

Run from project root: python tests/test_env.py

Checks:
- env loads in discrete and continuous modes
- obs / action shapes match EnvInfo
- 5 random-policy episodes run end-to-end
- RunLogger writes CSV + TensorBoard
"""

from __future__ import annotations

import os
import sys

import numpy as np

# So `python tests/test_env.py` works from the project root without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.env_utils import (
    DEFAULT_ENV_NAME,
    DEFAULT_SEEDS,
    RunLogger,
    get_device,
    make_env,
    set_seed,
)


def run_random_episodes(continuous: bool, n_episodes: int = 5, seed: int = 42) -> list[float]:
    """Run n random-policy episodes, return their rewards."""
    set_seed(seed)
    env, info = make_env(DEFAULT_ENV_NAME, seed=seed, continuous=continuous)

    mode = "continuous" if continuous else "discrete"
    print(f"\n--- {DEFAULT_ENV_NAME} [{mode}] ---")
    print(f"  obs_dim         = {info.obs_dim}")
    print(f"  is_discrete     = {info.is_discrete}")
    print(f"  n_actions       = {info.n_actions}")
    print(f"  action_dim      = {info.action_dim}")
    print(f"  action_low/high = {info.action_low} / {info.action_high}")
    print(f"  max_steps       = {info.max_episode_steps}")

    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        assert obs.shape == (info.obs_dim,), f"expected ({info.obs_dim},), got {obs.shape}"

        done = False
        ep_return = 0.0
        ep_len = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = terminated or truncated

        # The wrapper reports the true totals in info on the final step.
        recorded = step_info.get("episode")
        if recorded is not None:
            assert abs(recorded["r"] - ep_return) < 1e-4, "reward sum disagrees with wrapper"
            assert int(recorded["l"]) == ep_len, "length disagrees with wrapper"

        returns.append(ep_return)
        print(f"  ep {ep}: return={ep_return:8.2f}  length={ep_len}")

    env.close()
    return returns


def test_logging_schema() -> None:
    """Write a few rows with RunLogger; TB file check is optional."""
    import shutil

    log_dir = "results_test/logs"
    tb_root = "results_test/tensorboard"
    with RunLogger("randomtest", seed=0, log_dir=log_dir, tb_root=tb_root) as logger:
        for ep in range(3):
            logger.log_episode(episode=ep, reward=ep * 10.0, length=100 + ep, loss=None)
        logger.log_scalar("train/dummy", 1.23, step=0)

    with open(logger.csv_file) as f:
        rows = f.read().strip().splitlines()
    print(f"\nlog file {logger.csv_file} -> {len(rows)} rows (incl. header)")
    assert len(rows) == 4, rows
    print(f"  header: {rows[0]}")
    print(f"  row 1 : {rows[1]}")

    if logger.tb_enabled:
        tb_files = [
            f
            for f in os.listdir(os.path.join(tb_root, "randomtest", "seed0"))
            if f.startswith("events.out")
        ]
        assert tb_files, "no TensorBoard event file written"
        print(f"  tb event: {tb_files[0]}")
    else:
        print("  tensorboard disabled; CSV logging verified")

    shutil.rmtree("results_test")


def main() -> None:
    print(f"device = {get_device()}")
    print(f"DEFAULT_SEEDS = {DEFAULT_SEEDS}")

    discrete_returns = run_random_episodes(continuous=False, n_episodes=5)
    print(f"  mean random-policy return (discrete)   = {np.mean(discrete_returns):.2f}")

    continuous_returns = run_random_episodes(continuous=True, n_episodes=5)
    print(f"  mean random-policy return (continuous) = {np.mean(continuous_returns):.2f}")

    test_logging_schema()

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
