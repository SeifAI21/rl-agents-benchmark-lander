import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def resolve_device(explicit: str | None = None) -> torch.device:
    if explicit and explicit.strip().lower() not in ("", "auto"):
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_results(rewards, losses, save_dir="results/dqn"):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "rewards.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i, r])

    with open(os.path.join(save_dir, "losses.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "loss"])
        for i, loss_val in enumerate(losses):
            writer.writerow([i, loss_val])


def plot_training_curves(
    rewards,
    losses,
    save_dir="results/dqn",
    filename="training_curves.png",
):
    os.makedirs(save_dir, exist_ok=True)
    episodes_r = range(len(rewards))
    episodes_l = range(len(losses))

    fig, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_r.plot(episodes_r, rewards, color="tab:blue", linewidth=1.0, label="Reward")
    ax_r.set_ylabel("Episode return")
    ax_r.set_title("DQN training — rewards")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="upper right")

    ax_l.plot(episodes_l, losses, color="tab:orange", linewidth=1.0, label="Avg loss")
    ax_l.set_xlabel("Episode")
    ax_l.set_ylabel("Mean loss (episode)")
    ax_l.set_title("DQN training — loss")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
