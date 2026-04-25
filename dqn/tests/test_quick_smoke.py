"""Short integration smoke test."""

import pytest

pytest.importorskip("gymnasium")


def test_minimal_train_loop():
    from src.env_utils import make_env

    from DQN.agent import DQNAgent
    from DQN.config import DQNConfig
    from DQN.train import train

    config = DQNConfig(
        num_episodes=2,
        max_steps=25,
        batch_size=8,
        buffer_size=500,
        target_update_episodes=1,
        enable_tensorboard=False,
    )
    env, info = make_env(config.env_name, seed=config.seed, continuous=False, record_stats=False)
    assert info.n_actions is not None

    agent = DQNAgent(info.obs_dim, info.n_actions, config)
    rewards, losses = train(env, agent, config)
    assert len(rewards) == 2
    assert len(losses) == 2
    env.close()
