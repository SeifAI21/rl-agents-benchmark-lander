"""Model adapters for the three agents.

Each adapter knows:
1. How to construct a torch network whose `state_dict` matches what the
   training-side code saved.
2. How to turn a raw env observation into a discrete action.

The shapes here mirror the architectures in the teammates' branches:

- `dqn`        : 3-layer MLP, hidden=64, output=Q(s, .)            (DQN-Implementation)
- `reinforce`  : 2-hidden-layer MLP, hidden=128, output=logits(.)  (reinforce)
- `a2c`        : actor MLP with hidden=256 under `net.*` keys
                 (`actor_best.pth`), separate critic checkpoint

If a teammate ships a different architecture, register a new adapter via
`AGENT_REGISTRY[name] = MyAdapter` (see `make_adapter`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Network definitions (must stay binary-compatible with each agent's training
# code so we can `load_state_dict(strict=True)`).
# ---------------------------------------------------------------------------


class DQNNet(nn.Module):
    """Matches `dqn.src.DQN.networks.DeepQNetwork`."""

    def __init__(self, state_size: int = 8, action_size: int = 4, hidden_size: int = 64):
        super().__init__()
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ReinforcePolicyNet(nn.Module):
    """Matches `reinforce.PolicyNetwork` (Sequential under `network.*`)."""

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class A2CActorNet(nn.Module):
    """Matches the shipped A2C actor checkpoint (`actor_best.pth`).

    Checkpoint keys are `net.0.weight`, `net.0.bias`, ... so we keep the
    attribute name as `net` for strict `load_state_dict` compatibility.
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ---------------------------------------------------------------------------
# Adapter base class + concrete adapters.
# ---------------------------------------------------------------------------


@dataclass
class AdapterConfig:
    state_dim: int = 8
    action_dim: int = 4
    hidden: int | None = None
    deterministic: bool = True
    device: str | None = None


class BaseAdapter:
    """Common interface that `play.py` uses regardless of the algo."""

    name: str = "base"
    default_hidden: int = 64
    is_stochastic: bool = False  # whether the policy is naturally stochastic

    def __init__(self, cfg: AdapterConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        hidden = cfg.hidden if cfg.hidden is not None else self.default_hidden
        self.net = self._build_network(cfg.state_dim, cfg.action_dim, hidden)
        self.net.to(self.device)
        self.net.eval()

    def _build_network(self, state_dim: int, action_dim: int, hidden: int) -> nn.Module:
        raise NotImplementedError

    def load(self, checkpoint_path: str | Path) -> "BaseAdapter":
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")
        # `weights_only=True` was added in torch 1.13; fall back for older torch.
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=self.device)
        # Accept full save dicts that wrap the model under common keys.
        if isinstance(state, dict):
            for key in (
                "model",
                "policy",
                "q_network",
                "actor",
                "actor_state_dict",
                "state_dict",
            ):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        self.net.load_state_dict(state)
        return self

    @torch.no_grad()
    def forward_logits(self, obs: np.ndarray) -> torch.Tensor:
        """Override per algorithm — returns whatever drives action selection."""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.net(x)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        """Default: argmax over the network output. Stochastic adapters override."""
        out = self.forward_logits(obs)
        return int(out.argmax(dim=-1).item())


class DQNAdapter(BaseAdapter):
    name = "dqn"
    default_hidden = 64
    is_stochastic = False

    def _build_network(self, state_dim: int, action_dim: int, hidden: int) -> nn.Module:
        return DQNNet(state_dim, action_dim, hidden)


class ReinforceAdapter(BaseAdapter):
    name = "reinforce"
    default_hidden = 128
    is_stochastic = True

    def _build_network(self, state_dim: int, action_dim: int, hidden: int) -> nn.Module:
        return ReinforcePolicyNet(state_dim, action_dim, hidden)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        logits = self.forward_logits(obs)
        if self.cfg.deterministic:
            return int(logits.argmax(dim=-1).item())
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


class A2CAdapter(BaseAdapter):
    name = "a2c"
    default_hidden = 256
    is_stochastic = True

    def _build_network(self, state_dim: int, action_dim: int, hidden: int) -> nn.Module:
        return A2CActorNet(state_dim, action_dim, hidden)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        logits = self.forward_logits(obs)
        if self.cfg.deterministic:
            return int(logits.argmax(dim=-1).item())
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Factory + registry.
# ---------------------------------------------------------------------------


AdapterFactory = Callable[[AdapterConfig], BaseAdapter]

AGENT_REGISTRY: dict[str, AdapterFactory] = {
    "dqn": DQNAdapter,
    "reinforce": ReinforceAdapter,
    "a2c": A2CAdapter,
}


def build_adapter(
    algo: str,
    state_dim: int = 8,
    action_dim: int = 4,
    *,
    hidden: int | None = None,
    deterministic: bool = True,
    device: str | None = None,
) -> BaseAdapter:
    """Construct an adapter for `algo` (case-insensitive)."""
    key = algo.lower()
    if key not in AGENT_REGISTRY:
        raise KeyError(
            f"Unknown algo {algo!r}. Available: {sorted(AGENT_REGISTRY)}. "
            "Register a custom adapter via AGENT_REGISTRY[name] = factory."
        )
    cfg = AdapterConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=hidden,
        deterministic=deterministic,
        device=device,
    )
    return AGENT_REGISTRY[key](cfg)
