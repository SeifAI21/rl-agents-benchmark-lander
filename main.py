"""A2C (Advantage Actor-Critic) Agent — LunarLander-v3 Training"""

import argparse
from train import train
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="A2C — LunarLander-v3")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model", type=str, default="checkpoints/a2c/actor_best.pth")
    args = parser.parse_args()

    if args.mode == "train":
        train(
            num_episodes=args.episodes,
            n_steps=args.n_steps,
            seed=args.seed,
        )
    else:
        evaluate(
            actor_path=args.model,
            seed=args.seed,
            render=args.render,
        )


if __name__ == "__main__":
    main()
