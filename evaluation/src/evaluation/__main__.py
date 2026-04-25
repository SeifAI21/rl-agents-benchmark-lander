"""CLI entry point: `python -m evaluation <command> [...]`.

Subcommands:

    report   aggregate logs, write all plots + the markdown summary
    plot     just the plots (no markdown)
    play     load a checkpoint and watch / record one or more episodes
    eval     load a checkpoint and run N greedy episodes (no rendering)

Run any subcommand with `-h` for full options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import build_adapter
from .aggregate import discover_runs
from .plot import plot_all, write_index
from .play import best_demo_seed, evaluate_greedy, play_live, play_to_gif
from .report import write_markdown


# ---------------------------------------------------------------------------
# Subcommand: report (full pipeline).
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    grouped = discover_runs(args.log_dir, algos=args.algo or None)
    if not grouped:
        print(
            f"No CSV runs found under {args.log_dir!r}. "
            "Train at least one agent first (results/logs/{algo}_seed{seed}.csv).",
            file=sys.stderr,
        )
        return 1

    print("Discovered runs:")
    for algo, runs in sorted(grouped.items(), key=lambda kv: kv[0].lower()):
        print(f"  {algo:<12} seeds={runs.seeds}")

    plots = plot_all(
        grouped,
        out_dir=args.out_dir,
        window=args.window,
        last_n=args.last_n,
    )
    write_index(plots, args.out_dir)
    md_path = write_markdown(
        grouped,
        out_path=args.report_path,
        last_n=args.last_n,
        window=args.window,
        plots=plots,
    )
    print(f"\nWrote markdown report -> {Path(md_path).resolve()}")
    print(f"Plot index           -> {(Path(args.out_dir) / 'plots.txt').resolve()}")
    for label, path in plots.items():
        print(f"  {label:<22} {path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: plot (skip markdown).
# ---------------------------------------------------------------------------


def cmd_plot(args: argparse.Namespace) -> int:
    grouped = discover_runs(args.log_dir, algos=args.algo or None)
    if not grouped:
        print(f"No CSV runs found under {args.log_dir!r}.", file=sys.stderr)
        return 1
    plots = plot_all(grouped, out_dir=args.out_dir, window=args.window, last_n=args.last_n)
    write_index(plots, args.out_dir)
    for label, path in plots.items():
        print(f"  {label:<22} {path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: play (live / GIF).
# ---------------------------------------------------------------------------


def cmd_play(args: argparse.Namespace) -> int:
    adapter = build_adapter(
        args.algo,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden=args.hidden,
        deterministic=not args.stochastic,
        device=args.device,
    )
    adapter.load(args.checkpoint)
    print(f"Loaded {args.algo} checkpoint: {Path(args.checkpoint).resolve()}")

    seed = args.seed
    if args.demo_trials > 0:
        best_seed, best_return = best_demo_seed(
            adapter,
            env_id=args.env_id,
            n_trials=args.demo_trials,
            max_steps=args.max_steps,
        )
        print(
            f"Best demo seed across {args.demo_trials} trials: "
            f"{best_seed} (greedy return {best_return:.2f})"
        )
        seed = best_seed

    if args.gif:
        play_to_gif(
            adapter,
            args.gif,
            env_id=args.env_id,
            seed=seed,
            max_steps=args.max_steps,
            fps=args.fps,
        )

    if args.watch:
        play_live(
            adapter,
            env_id=args.env_id,
            seed=seed,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
            n_episodes=args.episodes,
        )

    if not args.gif and not args.watch:
        # Default: at least summarize a few greedy episodes so the command does something.
        stats = evaluate_greedy(
            adapter,
            env_id=args.env_id,
            n_episodes=args.episodes,
            base_seed=seed,
            max_steps=args.max_steps,
        )
        print(f"\nGreedy eval over {stats['n_episodes']} episodes:")
        print(f"  mean={stats['mean']:.2f} std={stats['std']:.2f} "
              f"min={stats['min']:.2f} max={stats['max']:.2f}")
        print(f"  solved rate (>=200): {stats['solved_rate']:.1%}  "
              f"mean length={stats['mean_length']:.1f}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: eval (no rendering).
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    adapter = build_adapter(
        args.algo,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden=args.hidden,
        deterministic=not args.stochastic,
        device=args.device,
    )
    adapter.load(args.checkpoint)
    stats = evaluate_greedy(
        adapter,
        env_id=args.env_id,
        n_episodes=args.episodes,
        base_seed=args.seed,
        max_steps=args.max_steps,
    )
    print(f"Greedy eval ({args.algo}, {stats['n_episodes']} eps):")
    print(f"  mean = {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"  min  = {stats['min']:.2f}    max = {stats['max']:.2f}")
    print(f"  solved rate (>=200): {stats['solved_rate']:.1%}")
    print(f"  mean episode length: {stats['mean_length']:.1f}")
    return 0


# ---------------------------------------------------------------------------
# Argparse plumbing.
# ---------------------------------------------------------------------------


def _add_common_play_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--algo", required=True, choices=sorted(("dqn", "reinforce", "a2c")),
                   help="Which agent's checkpoint we're loading.")
    p.add_argument("--checkpoint", required=True, type=str, help="Path to model.pth / *.pt file.")
    p.add_argument("--env-id", default="LunarLander-v3")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--state-dim", type=int, default=8,
                   help="Override only if you trained on a non-LunarLander env.")
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--hidden", type=int, default=None,
                   help="Override the adapter's default hidden size if your model differs.")
    p.add_argument("--device", default=None, help="cpu / cuda / mps; default autodetect.")
    p.add_argument("--stochastic", action="store_true",
                   help="For policy-gradient agents: sample actions instead of argmax.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Evaluation & visualization for the RL agents benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # report
    pr = sub.add_parser("report", help="Aggregate logs + plots + markdown report.")
    pr.add_argument("--log-dir", default="results/logs")
    pr.add_argument("--out-dir", default="results/plots")
    pr.add_argument("--report-path", default="results/report/summary.md")
    pr.add_argument("--window", type=int, default=100)
    pr.add_argument("--last-n", type=int, default=100)
    pr.add_argument("--algo", action="append", default=[],
                    help="Restrict to specific algos (repeatable).")
    pr.set_defaults(func=cmd_report)

    # plot
    pp = sub.add_parser("plot", help="Just write plots, no markdown report.")
    pp.add_argument("--log-dir", default="results/logs")
    pp.add_argument("--out-dir", default="results/plots")
    pp.add_argument("--window", type=int, default=100)
    pp.add_argument("--last-n", type=int, default=100)
    pp.add_argument("--algo", action="append", default=[])
    pp.set_defaults(func=cmd_plot)

    # play
    pl = sub.add_parser("play", help="Load a checkpoint, watch live / record GIF / eval.")
    _add_common_play_args(pl)
    pl.add_argument("--watch", action="store_true",
                    help="Open the LunarLander pygame window (needs a display).")
    pl.add_argument("--gif", default=None, help="Path to write a GIF for one episode.")
    pl.add_argument("--fps", type=int, default=30, help="GIF frame rate.")
    pl.add_argument("--step-delay", type=float, default=0.02,
                    help="Pause between steps in --watch mode (seconds).")
    pl.add_argument("--demo-trials", type=int, default=0,
                    help="Try seeds 0..N-1 headless and pick the best for visualization.")
    pl.set_defaults(func=cmd_play)

    # eval
    pe = sub.add_parser("eval", help="Greedy evaluation only (no render).")
    _add_common_play_args(pe)
    pe.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
