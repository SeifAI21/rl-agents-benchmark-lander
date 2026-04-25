from collections import deque

import numpy as np


def train(env, agent, config, logger=None):
    """
    Notebook-style training with env-setup logging support.

    Env creation/seeding happens in `DQN.main` via `src.env_utils.make_env`.
    """
    scores: list[float] = []
    losses: list[float] = []
    scores_window: deque[float] = deque(maxlen=100)
    eps = config.epsilon_start
    global_step = 0

    for i_episode in range(1, config.num_episodes + 1):
        # Use deterministic per-episode resets to align with env-setup README guidance.
        state, _ = env.reset(seed=config.seed + i_episode - 1)
        score = 0.0
        ep_losses: list[float] = []
        episode_length = 0

        for _step in range(config.max_steps):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            if agent.last_loss is not None:
                ep_losses.append(agent.last_loss)

            state = next_state
            score += float(reward)
            episode_length += 1
            global_step += 1

            if done:
                break

        scores_window.append(score)
        scores.append(score)
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        losses.append(avg_loss)

        eps = max(config.epsilon_min, config.epsilon_decay * eps)

        if i_episode % config.target_update_episodes == 0:
            agent.update_target_network()

        if logger is not None:
            logger.log_episode(
                episode=i_episode,
                reward=score,
                length=episode_length,
                loss=avg_loss,
            )
            logger.log_scalar("train/epsilon", eps, step=global_step)

        mean_100 = float(np.mean(scores_window)) if scores_window else score
        print(
            f"Episode {i_episode} | Score: {score:.2f} | "
            f"Avg100: {mean_100:.2f} | Loss: {avg_loss:.4f} | Eps: {eps:.3f}"
        )

        if (
            config.early_stop_solved
            and i_episode % config.print_every_episodes == 0
            and len(scores_window) >= 100
            and mean_100 >= 200.0
        ):
            break

    return scores, losses


if __name__ == "__main__":
    from .main import main

    main()
