# Final Project — Environment & Setup (Person 1)

Shared setup used by DQN, REINFORCE, A2C, and the evaluation code.

## Environment: `LunarLander-v3` (discrete, 4 actions)

| Property          | Value                                           |
|-------------------|-------------------------------------------------|
| State             | 8-dim vector (pos, vel, angle, ω, 2 legs)       |
| Action (primary)  | `Discrete(4)` — noop, left, main, right engine  |
| Action (optional) | `Box(-1, 1, (2,))` — continuous thrust (A2C)    |
| Max steps/ep      | 1000                                            |
| Solved            | avg return ≥ 200 over 100 eps                   |

Why discrete: small networks, DQN works as-is, a single seed finishes in tens of minutes so we can run multiple seeds.

## Layout

```
final-project/
├── README.md
├── src/
│   ├── __init__.py
│   └── env_utils.py      # env factory, seeding, logging
└── tests/
    └── test_env.py       # run this before training
```

Other folders (`src/agents/`, `configs/`, `results/`, …) will be added by Persons 2–5.

## Environment

Activate the CUDA conda env before running anything:

```bash
conda activate torch310
```

Deps already installed there: `torch`, `gymnasium[box2d]`, `numpy`, `tensorboard`, `matplotlib`.

## Quick start

```python
from src.env_utils import make_env, set_seed, get_device, RunLogger, DEFAULT_SEEDS

set_seed(42)
device = get_device()

env, info = make_env("LunarLander-v3", seed=42, continuous=False)
# Size networks from info.obs_dim and info.n_actions. Don't hardcode 8 or 4.

with RunLogger("dqn", seed=42) as logger:
    global_step = 0
    for ep in range(1000):
        obs, _ = env.reset(seed=42 + ep)
        done, ret, length = False, 0.0, 0
        while not done:
            action = ...  # your policy
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            ret += reward
            length += 1
            global_step += 1
        logger.log_episode(episode=ep, reward=ret, length=length, loss=None)
        # Optional TB-only scalars:
        # logger.log_scalar("train/epsilon", epsilon, step=global_step)
```

## API (`src/env_utils.py`)

| Symbol                  | What it does                                                      |
|-------------------------|-------------------------------------------------------------------|
| `DEFAULT_ENV_NAME`      | `"LunarLander-v3"`                                                |
| `DEFAULT_SEEDS`         | `[0, 1, 2, 42, 123]` — run all of these                           |
| `DEFAULT_MAX_EPISODES`  | `1000` (agree with team)                                          |
| `set_seed(seed)`        | seeds python / numpy / torch                                      |
| `get_device()`          | cuda if available, else cpu                                       |
| `make_env(...)`         | returns `(env, EnvInfo)`                                          |
| `EnvInfo`               | `obs_dim`, `is_discrete`, `n_actions`, `action_dim`, `action_low/high`, `max_episode_steps`, `seed` |
| `RunLogger(algo, seed)` | writes CSV + TensorBoard                                          |
| `csv_path(algo, seed)`  | `results/logs/{algo}_seed{seed}.csv`                              |
| `tb_dir(algo, seed)`    | `results/tensorboard/{algo}/seed{seed}/`                          |

### Step / reset

- `env.step(a)` → `(obs, reward, terminated, truncated, info)`.
- Use `done = terminated or truncated` to end the loop.
- Only `terminated` should zero bootstrap targets (truncation is a time limit, not a real terminal).
- `env.reset(seed=...)` → `(obs, info)`. Pass `seed=base + ep` each episode for reproducibility.

### Episode stats

`RecordEpisodeStatistics` adds `step_info["episode"]["r"]` and `["l"]` on the final step. The sanity test checks these match your own sums.

## Logging (CSV + TensorBoard)

`RunLogger` writes to both:

- **CSV** — `results/logs/{algo}_seed{seed}.csv` with columns:
  ```
  episode, reward, length, loss
  ```
- **TensorBoard** — `results/tensorboard/{algo}/seed{seed}/`.

`loss` can be blank if the algo doesn't track a per-episode loss. Each agent picks what it logs (DQN: mean TD loss; REINFORCE: PG loss; A2C: actor+critic loss) and notes it in its own README.

Extra scalars (epsilon, grad norm, entropy, …) go through `logger.log_scalar(tag, value, step)` — TB only.

### View TensorBoard

```bash
tensorboard --logdir results/tensorboard
```

Then open http://localhost:6006. All runs show up side-by-side.

## Sanity check

```bash
python tests/test_env.py
```

Runs 5 random episodes in each mode, checks shapes, and exercises the logger. Use the loop in `tests/test_env.py::run_random_episodes` as a template.

## Fairness rules

- All algos run on every seed in `DEFAULT_SEEDS`.
- All algos run for `DEFAULT_MAX_EPISODES` episodes.
- All algos use `make_env("LunarLander-v3", seed=<seed>, continuous=False)`.
- All agents log through `RunLogger(algo, seed)`.

If you need to change any of these, raise it in the group chat **before** long runs.
