"""Run a trained agent in LunarLander, optionally with a live window or GIF.

Two render modes are useful:
- "human"      : opens a pygame window so you can watch the lander land
                 (needs a display; works on the user's local Linux box).
- "rgb_array"  : returns RGB frames per step; we collect them and write a
                 GIF via imageio. Headless-friendly.

We import the env factory from the shared `src.env_utils` module that lives
on the env-setup branch (and is therefore on every downstream branch).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running as a script even when the repo isn't pip-installed: append the
# repo root to sys.path so `src.env_utils` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.env_utils import DEFAULT_ENV_NAME, DEFAULT_MAX_STEPS_PER_EPISODE, make_env, set_seed

from .adapters import BaseAdapter


@dataclass
class EpisodeResult:
    total_reward: float
    length: int
    terminated: bool
    truncated: bool
    frames: list[np.ndarray]


def _run_episode(
    adapter: BaseAdapter,
    *,
    env_id: str,
    seed: int,
    render_mode: Optional[str],
    max_steps: int,
    step_delay: float,
) -> EpisodeResult:
    env, info = make_env(
        env_id,
        seed=seed,
        continuous=False,
        render_mode=render_mode,
        record_stats=False,
    )
    if not info.is_discrete or info.n_actions is None:
        env.close()
        raise ValueError(f"play.py needs a discrete-action env; got {env_id}.")

    if info.obs_dim != adapter.cfg.state_dim:
        env.close()
        raise ValueError(
            f"Env obs_dim={info.obs_dim} but adapter expects {adapter.cfg.state_dim}. "
            "Pass --state-dim to match the trained model."
        )
    if info.n_actions != adapter.cfg.action_dim:
        env.close()
        raise ValueError(
            f"Env n_actions={info.n_actions} but adapter expects {adapter.cfg.action_dim}. "
            "Pass --action-dim to match the trained model."
        )

    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    total = 0.0
    steps = 0
    terminated = truncated = False

    for _step in range(max_steps):
        action = adapter.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        steps += 1

        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))
        elif render_mode == "human":
            env.render()
            if step_delay > 0:
                time.sleep(step_delay)

        if terminated or truncated:
            break

    env.close()
    return EpisodeResult(
        total_reward=total,
        length=steps,
        terminated=bool(terminated),
        truncated=bool(truncated),
        frames=frames,
    )


def play_live(
    adapter: BaseAdapter,
    *,
    env_id: str = DEFAULT_ENV_NAME,
    seed: int = 42,
    max_steps: int = DEFAULT_MAX_STEPS_PER_EPISODE,
    step_delay: float = 0.02,
    n_episodes: int = 1,
) -> list[EpisodeResult]:
    """Open a pygame window and watch the agent play `n_episodes`."""
    set_seed(seed)
    results: list[EpisodeResult] = []
    for ep in range(n_episodes):
        ep_seed = seed + ep
        print(f"[play_live] episode {ep + 1}/{n_episodes}  seed={ep_seed}")
        result = _run_episode(
            adapter,
            env_id=env_id,
            seed=ep_seed,
            render_mode="human",
            max_steps=max_steps,
            step_delay=step_delay,
        )
        outcome = "landed" if result.terminated and result.total_reward > 0 else (
            "crashed" if result.terminated else "timed out"
        )
        print(
            f"[play_live] episode {ep + 1} -> return={result.total_reward:.2f} "
            f"length={result.length} ({outcome})"
        )
        results.append(result)
    return results


def play_to_gif(
    adapter: BaseAdapter,
    gif_path: str | Path,
    *,
    env_id: str = DEFAULT_ENV_NAME,
    seed: int = 42,
    max_steps: int = DEFAULT_MAX_STEPS_PER_EPISODE,
    fps: int = 30,
) -> EpisodeResult:
    """Render one episode to RGB frames and write a GIF."""
    set_seed(seed)
    result = _run_episode(
        adapter,
        env_id=env_id,
        seed=seed,
        render_mode="rgb_array",
        max_steps=max_steps,
        step_delay=0.0,
    )
    if not result.frames:
        print("Warning: no frames captured, GIF not written.", file=sys.stderr)
        return result

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is required to write GIFs. Install with: pip install imageio"
        ) from exc

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, result.frames, fps=fps)
    print(
        f"[play_to_gif] wrote {gif_path.resolve()} "
        f"({len(result.frames)} frames, return={result.total_reward:.2f})"
    )
    return result


def best_demo_seed(
    adapter: BaseAdapter,
    *,
    env_id: str = DEFAULT_ENV_NAME,
    n_trials: int = 32,
    max_steps: int = DEFAULT_MAX_STEPS_PER_EPISODE,
) -> tuple[int, float]:
    """Pick the reset seed (in 0..n_trials-1) that yields the best return.

    Helpful when the trained agent is good on average but a particular seed
    starts the lander in a tough spot — we'd rather record a successful demo.
    """
    best_seed = 0
    best_return = float("-inf")
    for s in range(n_trials):
        result = _run_episode(
            adapter,
            env_id=env_id,
            seed=s,
            render_mode=None,
            max_steps=max_steps,
            step_delay=0.0,
        )
        if result.total_reward > best_return:
            best_return = result.total_reward
            best_seed = s
    return best_seed, best_return


def evaluate_greedy(
    adapter: BaseAdapter,
    *,
    env_id: str = DEFAULT_ENV_NAME,
    n_episodes: int = 100,
    base_seed: int = 1000,
    max_steps: int = DEFAULT_MAX_STEPS_PER_EPISODE,
) -> dict[str, float]:
    """Run `n_episodes` greedy episodes (no rendering) and summarize them.

    Uses a base seed offset (default 1000) so eval seeds don't overlap with
    common training seeds (0..999).
    """
    returns: list[float] = []
    lengths: list[int] = []
    for ep in range(n_episodes):
        result = _run_episode(
            adapter,
            env_id=env_id,
            seed=base_seed + ep,
            render_mode=None,
            max_steps=max_steps,
            step_delay=0.0,
        )
        returns.append(result.total_reward)
        lengths.append(result.length)

    arr = np.asarray(returns, dtype=np.float64)
    return {
        "n_episodes": len(returns),
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std(ddof=0)) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "solved_rate": float(np.mean(arr >= 200.0)) if arr.size else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
    }
