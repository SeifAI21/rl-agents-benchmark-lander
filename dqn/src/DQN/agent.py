import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .networks import DeepQNetwork
from .replay_buffer import ReplayBuffer
from .utils import resolve_device


class DQNAgent:
    """DQN: twin Q-nets, replay memory, and notebook-style `act` / `step` / `update_model`."""

    def __init__(self, state_dim: int, action_dim: int, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.device = resolve_device(config.device)
        self.action_size = action_dim

        self.q_network = DeepQNetwork(state_dim, action_dim, config.hidden_size).to(self.device)
        self.target_network = DeepQNetwork(state_dim, action_dim, config.hidden_size).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.buffer_size)

        self.last_loss: float | None = None

    def act(self, state, eps: float) -> int:
        """ε-greedy: random action with probability ~eps (same as reference notebook)."""
        if random.random() > eps:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state_t)
            self.q_network.train()
            return int(action_values.argmax(dim=1).item())
        return random.randint(0, self.action_size - 1)

    def step(self, state, action, reward, next_state, done):
        self.last_loss = None
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) > self.config.batch_size:
            self.update_model()

    def update_model(self) -> None:
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.asarray(dones, dtype=np.float32), device=self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            expected_q_values = rewards + self.config.gamma * next_q_values * (1.0 - dones)

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.last_loss = float(loss.detach().cpu().item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def save(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(state)
        self.update_target_network()
