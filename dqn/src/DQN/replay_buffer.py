import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Deque replay memory with uniform random batches (`np.stack` states, notebook-style)."""

    def __init__(self, buffer_size: int = 10_000):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states, axis=0),
            actions,
            rewards,
            np.stack(next_states, axis=0),
            dones,
        )

    def __len__(self):
        return len(self.buffer)
