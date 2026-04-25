"""Training loop for A2C agent on LunarLander-v3."""

import numpy as np
import torch
import os

try:
    from environment import make_env
except ImportError:
    import gymnasium as gym

    def make_env(seed: int = 42):
        env = gym.make("LunarLander-v3")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        return env


from agent import A2CAgent
from logger import Logger


def train(
    num_episodes: int = 1000,
    n_steps: int = 5,
    seed: int = 42,
    save_dir: str = "checkpoints/a2c",
    log_dir: str = "logs/a2c",
    print_every: int = 50,
    solve_threshold: float = 200.0,
):
    """Train A2C agent on LunarLander-v3."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim)
    logger = Logger(log_dir=log_dir)
    os.makedirs(save_dir, exist_ok=True)

    best_avg = -np.inf

    print(f"\nA2C Training — LunarLander-v3")
    print(f"Episodes: {num_episodes} | n-steps: {n_steps} | seed: {seed}\n")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_steps = 0
        ep_actor_losses, ep_critic_losses, ep_entropies = [], [], []

        done = False
        while not done:
            states, actions, log_probs, rewards, dones = [], [], [], [], []

            for _ in range(n_steps):
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(terminated or truncated))

                episode_reward += reward
                episode_steps += 1
                state = next_state
                done = terminated or truncated

                if done:
                    break

            if not done:
                last_state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    last_value = agent.critic(last_state_t).item()
            else:
                last_value = 0.0

            returns = agent.compute_returns(rewards, dones, last_value)
            a_loss, c_loss, entropy = agent.update(states, actions, log_probs, returns)
            ep_actor_losses.append(a_loss)
            ep_critic_losses.append(c_loss)
            ep_entropies.append(entropy)

        mean_actor = np.mean(ep_actor_losses)
        mean_critic = np.mean(ep_critic_losses)
        mean_ent = np.mean(ep_entropies)

        moving_avg = logger.log(
            episode, episode_reward, episode_steps,
            mean_actor, mean_critic, mean_ent
        )

        if episode % print_every == 0:
            print(
                f"Ep {episode:>4} | "
                f"Reward: {episode_reward:>8.1f} | "
                f"Avg(100): {moving_avg:>7.1f} | "
                f"A-Loss: {mean_actor:>7.4f} | "
                f"C-Loss: {mean_critic:>7.4f} | "
                f"Entropy: {mean_ent:.3f}"
            )

        if moving_avg > best_avg and episode >= 100:
            best_avg = moving_avg
            torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_best.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(save_dir, "critic_best.pth"))

        if moving_avg >= solve_threshold and episode >= 100:
            print(f"\n✅ Solved at episode {episode}! Avg reward: {moving_avg:.1f}")
            break

    torch.save(agent.actor.state_dict(), os.path.join(save_dir, "actor_final.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(save_dir, "critic_final.pth"))
    logger.close()
    env.close()

    print(f"\nLogs → {log_dir}/training_log.csv")
    print(f"Models → {save_dir}/")
    return agent
