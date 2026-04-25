"""Training logger for A2C agent."""

import os
import csv
import time
import numpy as np
from collections import deque


class Logger:
    """Logs training metrics to CSV file."""

    def __init__(self, log_dir: str = "logs/a2c"):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "episode", "reward", "steps",
            "actor_loss", "critic_loss", "entropy",
            "moving_avg_100", "elapsed_s"
        ])
        self._reward_window = deque(maxlen=100)
        self._t0 = time.time()

    def log(self, episode, reward, steps, actor_loss, critic_loss, entropy):
        self._reward_window.append(reward)
        moving_avg = np.mean(self._reward_window)
        elapsed = round(time.time() - self._t0, 1)

        self._writer.writerow([
            episode, round(reward, 2), steps,
            round(actor_loss, 5), round(critic_loss, 5),
            round(entropy, 5), round(moving_avg, 2), elapsed
        ])
        self._file.flush()
        return moving_avg

    def close(self):
        self._file.close()
        print(f"[Logger] Log saved to {self.csv_path}")
