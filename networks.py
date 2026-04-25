"""Neural network architectures for A2C agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """Policy network that maps state → action probability distribution."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        # Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("[WARNING] NaN/Inf detected in actor logits!")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                                torch.zeros_like(logits), logits)
        # Stable softmax with temperature scaling
        probs = F.softmax(logits / 1.0, dim=-1)
        # Clamp probabilities to avoid exact zeros and NaNs
        probs = torch.clamp(probs, min=1e-8, max=1.0 - 1e-8)
        # Normalize to ensure valid probability distribution
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return probs

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        probs = self.forward(state)
        # Additional NaN check before creating distribution
        if torch.isnan(probs).any():
            print("[WARNING] NaN detected in actor probabilities!")
            probs = torch.ones_like(probs) / probs.shape[-1]
        return Categorical(probs)


class CriticNetwork(nn.Module):
    """Value network that maps state → scalar V(s) for advantage baseline."""

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
