"""CLI: train DQN using the shared env-setup helpers."""

from src.env_utils import RunLogger, make_env, set_seed

from .agent import DQNAgent
from .config import DQNConfig
from .train import train
from .utils import plot_training_curves, resolve_device, save_results


def main() -> None:
    config = DQNConfig()
    set_seed(config.seed)

    env, info = make_env(config.env_name, seed=config.seed, continuous=False)
    if not info.is_discrete or info.n_actions is None:
        raise ValueError("DQN requires a discrete-action Lunar Lander environment.")

    print(f"Using device: {resolve_device(config.device)}")
    print(
        f"Using env: {info.env_name} | seed={info.seed} | "
        f"obs_dim={info.obs_dim} | actions={info.n_actions}"
    )

    agent = DQNAgent(info.obs_dim, info.n_actions, config)

    save_dir = "results/dqn"
    with RunLogger("dqn", seed=config.seed, enable_tb=config.enable_tensorboard) as logger:
        rewards, losses = train(env, agent, config, logger=logger)

    save_results(rewards, losses, save_dir=save_dir)
    plot_path = plot_training_curves(rewards, losses, save_dir=save_dir)
    print(f"Saved plot: {plot_path}")

    agent.save(f"{save_dir}/model.pth")
    env.close()


if __name__ == "__main__":
    main()
