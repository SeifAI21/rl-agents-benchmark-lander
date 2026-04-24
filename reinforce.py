import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from env_utils import make_env, set_seed, get_device, RunLogger


# -- Policy Network --

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        logits = self.network(x)
        return Categorical(logits=logits)  # Distribution


# -- Action Selection --

def select_action(policy, state, device):
    state_tensor = torch.FloatTensor(state).to(device)
    dist = policy(state_tensor)
    action = dist.sample()              # Sample from distribution
    log_prob = dist.log_prob(action)    # Saving for loss
    return action.item(), log_prob


# -- Discounted Returns --

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    # Normalized returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# -- Policy Update --

def update_policy(optimizer, policy, log_probs, returns):
    loss = torch.stack([-log_prob * G for log_prob, G in zip(log_probs, returns)]).sum()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)   # added clipping back in
    optimizer.step()
    return loss.item()


# -- Training Loop --

# Environment fully solved (200) in ~2600-3000 episodes at some seeds 
# usually tappers off ad around 130 - 150 
def train_reinforce(seed=42, num_episodes=3000, gamma=0.99, lr=3e-4):

    set_seed(seed)
    device = get_device()
    env, env_info = make_env(seed=seed, continuous=False)

    print(f"Env: {env_info.env_name}")
    print(f"State dim: {env_info.obs_dim}, Actions: {env_info.n_actions}")
    print(f"Device: {device}")

    policy = PolicyNetwork(
        state_dim=env_info.obs_dim,
        action_dim=env_info.n_actions,
        hidden_dim=128
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    with RunLogger("reinforce", seed=seed, enable_tb=False) as logger:
        rewards_per_episode = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            log_probs = []
            rewards = []
            done = False
            steps = 0

            while not done:
                action, log_prob = select_action(policy, state, device)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
                steps += 1

            returns = compute_returns(rewards, gamma)
            loss = update_policy(optimizer, policy, log_probs, returns)

            total_reward = sum(rewards)
            rewards_per_episode.append(total_reward)

            logger.log_episode(episode=episode, reward=total_reward, length=steps, loss=loss)

            if (episode + 1) % 100 == 0:
                avg = np.mean(rewards_per_episode[-100:])
                print(f"Episode {episode+1} | Avg Reward (last 100): {avg:.2f} | Loss: {loss:.4f}")

    env.close()
    return policy, rewards_per_episode


# -- Main --

if __name__ == "__main__":
    policy, rewards = train_reinforce(seed=42, num_episodes=3000)

    # Save outputs for Person 5
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/rewards", exist_ok=True)

    np.save("results/rewards/reinforce_rewards.npy", np.array(rewards))
    torch.save(policy.state_dict(), "results/models/reinforce_policy.pth")

    print("Saved: results/rewards/reinforce_rewards.npy")
    print("Saved: results/models/reinforce_policy.pth")
    print("CSV logs saved to: results/logs/reinforce_seed42.csv")