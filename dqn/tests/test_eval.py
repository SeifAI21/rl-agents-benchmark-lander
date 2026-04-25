import pytest

from DQN.eval import summarize_returns

pytest.importorskip("gymnasium")


def test_summarize_returns():
    s = summarize_returns([100.0, 200.0, 300.0])
    assert s["mean"] == 200.0
    assert s["episodes_ge_200_rate"] == pytest.approx(2 / 3)


def test_run_greedy_evaluation_smoke():
    from src.env_utils import make_env

    from DQN.agent import DQNAgent
    from DQN.config import DQNConfig
    from DQN.eval import run_greedy_evaluation

    config = DQNConfig(
        num_episodes=1,
        max_steps=15,
        batch_size=4,
        buffer_size=100,
        device="cpu",
    )
    env, info = make_env(config.env_name, seed=config.seed, continuous=False, record_stats=False)
    assert info.n_actions is not None

    agent = DQNAgent(info.obs_dim, info.n_actions, config)
    returns = run_greedy_evaluation(env, agent, num_episodes=2, max_steps=20, seed=0)
    assert len(returns) == 2
    env.close()
