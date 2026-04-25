"""Tests for the model adapters.

We don't need real trained checkpoints — the adapters' job is to (a) build
a network of the right shape, (b) load a state_dict, and (c) emit a valid
discrete action. We exercise all three with random weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.adapters import (
    AGENT_REGISTRY,
    A2CAdapter,
    DQNAdapter,
    DQNNet,
    ReinforceAdapter,
    ReinforcePolicyNet,
    build_adapter,
)


@pytest.mark.parametrize("algo", ["dqn", "reinforce", "a2c"])
def test_build_adapter_outputs_valid_action(algo: str) -> None:
    adapter = build_adapter(algo, state_dim=8, action_dim=4)
    obs = np.random.default_rng(0).normal(size=(8,)).astype(np.float32)
    action = adapter.act(obs)
    assert isinstance(action, int)
    assert 0 <= action < 4


def test_build_adapter_unknown_algo() -> None:
    with pytest.raises(KeyError):
        build_adapter("ppo")


def test_dqn_adapter_loads_dqn_state_dict(tmp_path: Path) -> None:
    net = DQNNet(8, 4, 64)
    ckpt = tmp_path / "dqn.pth"
    torch.save(net.state_dict(), ckpt)

    adapter = DQNAdapter.__new__(DQNAdapter)
    adapter.__init__(  # type: ignore[misc]
        cfg=type("C", (), {"state_dim": 8, "action_dim": 4, "hidden": 64,
                            "deterministic": True, "device": "cpu"})(),
    )
    adapter.load(ckpt)
    obs = np.zeros((8,), dtype=np.float32)
    assert 0 <= adapter.act(obs) < 4


def test_reinforce_adapter_loads_policy_state_dict(tmp_path: Path) -> None:
    net = ReinforcePolicyNet(8, 4, 128)
    ckpt = tmp_path / "policy.pth"
    torch.save(net.state_dict(), ckpt)

    adapter = build_adapter("reinforce", state_dim=8, action_dim=4)
    adapter.load(ckpt)
    obs = np.zeros((8,), dtype=np.float32)
    assert 0 <= adapter.act(obs) < 4


def test_load_checkpoint_with_wrapping_dict(tmp_path: Path) -> None:
    net = DQNNet(8, 4, 64)
    payload = {"q_network": net.state_dict(), "epoch": 7}
    ckpt = tmp_path / "wrapped.pth"
    torch.save(payload, ckpt)

    adapter = build_adapter("dqn", state_dim=8, action_dim=4)
    adapter.load(ckpt)
    obs = np.zeros((8,), dtype=np.float32)
    assert 0 <= adapter.act(obs) < 4


def test_stochastic_adapter_returns_varied_actions() -> None:
    torch.manual_seed(0)
    adapter = build_adapter("reinforce", state_dim=8, action_dim=4, deterministic=False)
    obs = np.zeros((8,), dtype=np.float32)
    actions = {adapter.act(obs) for _ in range(200)}
    # With softmax + multinomial we should hit more than one action across 200 tries.
    assert len(actions) > 1


def test_registry_exposes_all_three() -> None:
    assert set(AGENT_REGISTRY) == {"dqn", "reinforce", "a2c"}
    for cls in (DQNAdapter, ReinforceAdapter, A2CAdapter):
        assert cls.__name__.endswith("Adapter")
