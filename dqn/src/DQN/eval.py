"""Greedy evaluation (epsilon=0); optional live window or GIF of one episode."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from src.env_utils import DEFAULT_ENV_NAME, DEFAULT_MAX_STEPS_PER_EPISODE, make_env, set_seed

from .agent import DQNAgent
from .config import DQNConfig
from .utils import resolve_device


def run_greedy_evaluation(
    env: gym.Env,
    agent: DQNAgent,
    *,
    num_episodes: int,
    max_steps: int,
    seed: int | None = None,
) -> list[float]:
    returns: list[float] = []
    base_seed = 42 if seed is None else seed
    for ep in range(num_episodes):
        state, _ = env.reset(seed=base_seed + ep)
        total = 0.0
        for _step in range(max_steps):
            action = agent.act(state, 0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    return returns


def run_one_greedy_return(
    env_id: str,
    agent: DQNAgent,
    *,
    max_steps: int,
    seed: int,
) -> float:
    """One headless greedy episode; return total reward (for demo seed search)."""
    env, _info = make_env(env_id, seed=seed, continuous=False, record_stats=False)
    state, _ = env.reset(seed=seed)
    total = 0.0
    for _step in range(max_steps):
        action = agent.act(state, 0.0)
        state, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    env.close()
    return total


def pick_demo_seed(
    env_id: str,
    agent: DQNAgent,
    *,
    max_steps: int,
    n_trials: int,
) -> tuple[int, float]:
    """Try reset seeds 0 .. n_trials-1; pick seed with highest greedy return."""
    best_seed = 0
    best_return = float("-inf")
    for s in range(n_trials):
        r = run_one_greedy_return(env_id, agent, max_steps=max_steps, seed=s)
        if r > best_return:
            best_return = r
            best_seed = s
    return best_seed, best_return


def play_one_greedy_visual(
    agent: DQNAgent,
    *,
    env_id: str,
    max_steps: int,
    seed: int | None,
    render_mode: str,
    step_delay: float,
    gif_path: str | None,
) -> float:
    """
    Run a single greedy episode with visualization (human window or rgb_array frames).

    render_mode: "human" for live pygame window, "rgb_array" to capture frames for GIF.
    """
    use_seed = 42 if seed is None else seed
    env, _info = make_env(
        env_id,
        seed=use_seed,
        continuous=False,
        render_mode=render_mode,
        record_stats=False,
    )
    state, _ = env.reset(seed=use_seed)

    frames: list[np.ndarray] = []
    total = 0.0

    for _step in range(max_steps):
        action = agent.act(state, 0.0)
        state, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)

        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        elif render_mode == "human":
            env.render()
            if step_delay > 0:
                time.sleep(step_delay)

        if terminated or truncated:
            break

    env.close()

    if gif_path and not frames:
        print("Warning: no rgb frames captured; GIF not written.", file=sys.stderr)
    if gif_path and frames:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            print("Install imageio to save GIF: pip install imageio", file=sys.stderr)
            raise exc
        Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"Saved episode GIF: {Path(gif_path).resolve()}")

    return total


def summarize_returns(returns: list[float]) -> dict[str, float]:
    arr = np.asarray(returns, dtype=np.float64)
    solved_rate = float(np.mean(arr >= 200.0)) if len(arr) else 0.0
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "min": float(np.min(arr)) if len(arr) else 0.0,
        "max": float(np.max(arr)) if len(arr) else 0.0,
        "episodes_ge_200_rate": solved_rate,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained DQN (greedy). Use --watch for live Lunar Lander, "
        "or --gif to save one greedy episode."
    )
    p.add_argument("--checkpoint", type=str, default="results/dqn/model.pth")
    p.add_argument("--env-id", type=str, default=DEFAULT_ENV_NAME)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--watch",
        action="store_true",
        help="After metrics, play one greedy episode in a live window (pygame; needs a display).",
    )
    p.add_argument(
        "--gif",
        type=str,
        default=None,
        metavar="PATH",
        help="After metrics, record one greedy episode to this GIF file (rgb_array; headless OK).",
    )
    p.add_argument(
        "--step-delay",
        type=float,
        default=0.02,
        help="Seconds to sleep after each env step when using --watch (default 0.02).",
    )
    p.add_argument(
        "--demo-trials",
        type=int,
        default=None,
        metavar="N",
        help="With --gif/--watch: try reset seeds 0..N-1 headless and visualize the best "
        "(default N=32; use 0 for fixed --vis-seed / 42).",
    )
    p.add_argument(
        "--vis-seed",
        type=int,
        default=None,
        help="When --demo-trials 0: reset seed for GIF/live (default: same as --seed).",
    )
    return p.parse_args(argv)


def resolve_vis_seed(args: argparse.Namespace, agent: DQNAgent) -> tuple[int, str]:
    """
    Decide which reset seed to use for GIF / live demo.

    Default (with --gif/--watch): search demo_trials seeds and pick best greedy return.
    """
    want_viz = bool(args.gif or args.watch)
    if not want_viz:
        return args.seed, ""

    trials = args.demo_trials
    if trials is None:
        trials = 32
    if trials <= 0:
        seed = args.vis_seed if args.vis_seed is not None else args.seed
        return seed, f"fixed reset seed {seed} (--demo-trials 0)"

    best_s, best_r = pick_demo_seed(
        args.env_id,
        agent,
        max_steps=args.max_steps,
        n_trials=trials,
    )
    return best_s, f"best of {trials} resets (seed {best_s}, greedy return {best_r:.2f})"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"Checkpoint not found: {ckpt.resolve()}", file=sys.stderr)
        return 1

    config = DQNConfig(env_name=args.env_id, seed=args.seed)
    if args.device is not None:
        config.device = args.device if args.device.lower() != "auto" else None
    set_seed(config.seed)

    env, info = make_env(args.env_id, seed=args.seed, continuous=False)
    if not info.is_discrete or info.n_actions is None:
        raise ValueError("DQN evaluation requires a discrete-action environment.")

    device = resolve_device(config.device)
    print(f"Using device: {device}")
    print(f"Using env: {info.env_name} | seed={info.seed}")
    print(f"Loading: {ckpt.resolve()}")

    agent = DQNAgent(info.obs_dim, info.n_actions, config)
    agent.load(str(ckpt))

    returns = run_greedy_evaluation(
        env,
        agent,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    stats = summarize_returns(returns)

    print(f"\nGreedy evaluation over {args.episodes} episodes ({args.env_id}):")
    print(f"  mean return:   {stats['mean']:.2f} +/- {stats['std']:.2f}")
    print(f"  median:        {stats['median']:.2f}")
    print(f"  min / max:     {stats['min']:.2f} / {stats['max']:.2f}")
    print(
        f"  share >= 200:  {stats['episodes_ge_200_rate']:.1%} "
        "(common 'solved' threshold for Lunar Lander-style tasks)"
    )

    env.close()

    if args.gif or args.watch:
        vis_seed, vis_note = resolve_vis_seed(args, agent)
        print(f"\nDemo ({vis_note})")

    if args.gif:
        score = play_one_greedy_visual(
            agent,
            env_id=args.env_id,
            max_steps=args.max_steps,
            seed=vis_seed,
            render_mode="rgb_array",
            step_delay=0.0,
            gif_path=args.gif,
        )
        print(f"\nRecorded greedy episode return (GIF run): {score:.2f}")

    if args.watch:
        print("\nOpening live Lunar Lander (close the window when done)...")
        score = play_one_greedy_visual(
            agent,
            env_id=args.env_id,
            max_steps=args.max_steps,
            seed=vis_seed,
            render_mode="human",
            step_delay=args.step_delay,
            gif_path=None,
        )
        print(f"Live greedy episode return: {score:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
