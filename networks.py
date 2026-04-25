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
        return F.softmax(self.net(x), dim=-1)

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        probs = self.forward(state)
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
