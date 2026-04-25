# Evaluation & Visualization (Person 5)

Reads the per-episode CSV logs every agent writes via `src.env_utils.RunLogger`
and the saved checkpoints from DQN / REINFORCE / A2C, then produces:

- combined reward-vs-episode plots (with moving averages and ± std band)
- per-agent training curves and loss plots
- a markdown summary comparing **convergence speed**, **stability**, and **final performance**
- a "game" viewer: load any agent's checkpoint and watch it play live in the
  LunarLander pygame window, optionally recording a GIF

## Current project status

- DQN: real `seed42` checkpoint/log integrated
- REINFORCE: real `seed42` checkpoint/log integrated
- A2C: real actor/critic checkpoints and real `seed42` log integrated

Current report is generated on a fair aligned set:

- shared seed set: `[42]` for all algorithms
- shared episode budget: `1000` episodes for all algorithms

## Data contract (do not change without team agreement)

Every agent's training loop writes:

```
results/logs/{algo}_seed{seed}.csv           # columns: episode, reward, length, loss
results/<wherever>/<checkpoint>.pth          # per-agent (paths below)
```

Reference checkpoint paths used by the play CLI by default:

| Algo        | Default checkpoint location           | Source branch          |
|-------------|---------------------------------------|------------------------|
| `dqn`       | `results/dqn/model.pth`               | `DQN-Implementation`   |
| `reinforce` | `results/models/reinforce_policy.pth` | `reinforce`            |
| `a2c`       | `results/models/a2c_actor.pth` (TBD)  | (A2C branch, pending)  |

If a teammate uses a different network architecture, register an adapter:

```python
from evaluation.adapters import AGENT_REGISTRY, BaseAdapter

class MyA2CAdapter(BaseAdapter):
    name = "a2c"
    default_hidden = 256
    def _build_network(self, sd, ad, h):
        return MyActor(sd, ad, h)

AGENT_REGISTRY["a2c"] = MyA2CAdapter
```

## Layout

```
evaluation/
├── README.md
├── src/
│   └── evaluation/
│       ├── __init__.py
│       ├── __main__.py     # CLI: report / plot / play / eval
│       ├── aggregate.py    # CSV loading, smoothing, summary stats
│       ├── adapters.py     # minimal model classes for each algo
│       ├── plot.py         # matplotlib plots (Agg backend, headless OK)
│       ├── play.py         # live pygame + GIF rendering
│       └── report.py       # markdown summary
└── tests/
    ├── test_aggregate.py
    ├── test_adapters.py
    └── test_plot_and_report.py
```

## Quick start

From the repo root:

```bash
# 1. Aggregate everything: per-agent + comparison plots + markdown report.
python -m evaluation report

# 2. Just the plots, no markdown.
python -m evaluation plot --window 100

# 3. Watch a trained DQN play live (needs a display).
python -m evaluation play \
    --algo dqn \
    --checkpoint results/dqn/model.pth \
    --watch \
    --episodes 3

# 4. Record one greedy episode as a GIF (headless-friendly).
python -m evaluation play \
    --algo reinforce \
    --checkpoint results/models/reinforce_policy.pth \
    --gif results/gifs/reinforce_demo.gif \
    --demo-trials 16

# 5. Headless greedy evaluation (mean ± std over N episodes).
python -m evaluation eval \
    --algo dqn \
    --checkpoint results/dqn/model.pth \
    --episodes 100
```

Outputs land under:

```
results/plots/
    curves_dqn.png
    curves_reinforce.png
    curves_a2c.png
    comparison_rewards.png
    final_performance.png
    plots.txt           # index for the markdown report
results/report/
    summary.md          # the report scaffold to extend with conclusions
results/gifs/
    *.gif               # whatever you recorded with `play --gif`
```

## Integrating teammate artifacts

When a teammate sends real files, copy them into the standard paths, then
rerun the report:

```bash
# Example: replace REINFORCE seed42 artifacts
cp ~/Downloads/reinforce_seed42.csv results/logs/reinforce_seed42.csv
cp ~/Downloads/reinforce_policy.pth results/models/reinforce_policy.pth

# Rebuild plots + summary
python -m evaluation report
```

Expected destination paths:

- DQN model: `results/dqn/model.pth`
- DQN logs: `results/logs/dqn_seed{seed}.csv`
- REINFORCE model: `results/models/reinforce_policy.pth`
- REINFORCE logs: `results/logs/reinforce_seed{seed}.csv`
- A2C model: `results/models/a2c_actor.pth`
- A2C critic: `results/models/a2c_critic.pth`
- A2C logs: `results/logs/a2c_seed{seed}.csv`

## What the report measures

| Metric                | How it's computed                                                              |
|-----------------------|--------------------------------------------------------------------------------|
| Final return          | per-seed mean over the **last `--last-n` episodes** (default 100), then mean ± std across seeds |
| Convergence episode   | first episode where the moving-average return ≥ 200 (window=`--window`), averaged over seeds that solve |
| Stability             | per-seed std of the last `--last-n` returns, averaged across seeds (lower is better) |
| Solved threshold line | the green dashed line at 200 on every reward plot                              |

## Fairness checks (Person 5's responsibility)

When you run `python -m evaluation report`, eyeball the printout:

- All algos should have the same set of seeds (`DEFAULT_SEEDS = [0, 1, 2, 42, 123]`).
- All algos should have roughly the same number of episodes (we truncate to the
  shortest seed inside the algo when averaging, but big mismatches across algos
  bias the comparison).
- The shared env factory `src.env_utils.make_env("LunarLander-v3", seed=...)` is the
  only way the env should be built — anyone who created a custom env breaks the
  comparison.

If something looks off, raise it in the group chat **before** rerunning long jobs.

## Tests

```bash
python -m pytest evaluation/tests
```

Tests use synthetic CSVs and random model weights, so they pass without any
trained checkpoints. They cover:

- CSV parsing (incl. blank `loss` cells), filename inference, run grouping
- moving averages, summary stats, convergence detection, stability
- adapter construction + state-dict loading for all three algorithms
- end-to-end PNG and markdown writing on a fake log directory
