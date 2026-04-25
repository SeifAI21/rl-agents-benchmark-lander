"""Feedforward Q-network (same layout as the Lunar Lander DQN tutorial notebook)."""

import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    """State -> Q(a|s) for each discrete action (Mnih et al., 2015)."""

    def __init__(self, state_size: int = 8, action_size: int = 4, hidden_size: int = 64):
        super().__init__()
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
