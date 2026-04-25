"""Evaluation utilities for trained A2C agent."""

import numpy as np
import torch
import gymnasium as gym

from networks import ActorNetwork


def evaluate(
    actor_path: str = "checkpoints/a2c/actor_best.pth",
    num_episodes: int = 10,
    seed: int = 0,
    render: bool = False,
):
    """
    Load and evaluate a trained actor network.
    
    Returns:
        list: Episode rewards for comparison and analysis.
    """
    env_kwargs = {"render_mode": "human"} if render else {}
    env = gym.make("LunarLander-v3", **env_kwargs)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = ActorNetwork(state_dim, action_dim)
    actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    actor.eval()

    rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        total = 0
        done = False
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                dist = actor.get_distribution(state_t)
                action = dist.sample().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
        print(f"  Eval ep {ep + 1}: {total:.1f}")

    env.close()
    print(f"\nMean eval reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    return rewards
