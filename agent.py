"""A2C (Advantage Actor-Critic) Agent implementation."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from networks import ActorNetwork, CriticNetwork


class A2CAgent:
    """Advantage Actor-Critic agent for discrete action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        print(f"[A2C] Device: {self.device}")
        print(f"[A2C] Actor params:  {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"[A2C] Critic params: {sum(p.numel() for p in self.critic.parameters()):,}")

    def select_action(self, state: np.ndarray):
        """Sample action from policy and return action + log probability."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist = self.actor.get_distribution(state_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def compute_returns(self, rewards: list, dones: list, last_value: float) -> list:
        """Compute discounted returns with bootstrapping from last state value."""
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def update(
        self,
        states: list,
        actions: list,
        log_probs: list,
        returns: list,
    ):
        """Perform one gradient update step for both actor and critic networks."""
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        log_probs_t = torch.stack(log_probs).to(self.device)

        # Detect NaNs in inputs
        if torch.isnan(returns_t).any():
            print("[ERROR] NaN detected in returns!")
            return float('nan'), float('nan'), float('nan')
        if torch.isnan(log_probs_t).any():
            print("[ERROR] NaN detected in log_probs!")
            return float('nan'), float('nan'), float('nan')

        values = self.critic(states_t)
        advantages = (returns_t - values).detach()
        
        # Safe advantage normalization
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-4:  # Only normalize if std is significant
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:  # Just center if std is too small
            advantages = advantages - adv_mean

        critic_loss = F.mse_loss(values, returns_t)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        dist = self.actor.get_distribution(states_t)
        entropy = dist.entropy().mean()
        actor_loss = -(log_probs_t * advantages).mean() - self.entropy_coef * entropy

        # Detect NaNs in losses
        if torch.isnan(actor_loss) or torch.isnan(critic_loss):
            print("[ERROR] NaN detected in losses!")
            print(f"  Actor loss: {actor_loss}")
            print(f"  Critic loss: {critic_loss}")
            print(f"  Advantages min/max/mean/std: {advantages.min()}/{advantages.max()}/{advantages.mean()}/{advantages.std()}")
            return float('nan'), float('nan'), float('nan')

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()
