# RL Agents Benchmark — LunarLander

This repo contains:

- shared environment setup in `src/env_utils.py`
- DQN training code in `dqn/src/DQN/`
- evaluation and visualization tooling in `evaluation/` (Person 5 task)

## Current branch status (`evaluation-visualization`)

- DQN artifacts: real `seed42` model/log integrated
- REINFORCE artifacts: real `seed42` model/log integrated
- A2C artifacts: real actor/critic checkpoints + real `seed42` log integrated

Current comparison is fully real for the shared seed set used in this branch:

- Seed set: `[42]` for all algorithms
- Episode count: `1000` for all algorithms

## Main docs

- Environment and shared logging contract: `src/env_utils.py` and `tests/test_env.py`
- Evaluation/visualization workflow and CLI: `evaluation/README.md`

## Quick commands

```bash
# Shared env sanity check
python -m pytest tests/test_env.py

# Evaluation test suite
python -m pytest evaluation/tests

# Regenerate report + plots
python -m evaluation report

# Watch a trained policy play
python -m evaluation play --algo dqn --checkpoint results/dqn/model.pth --watch

# Optional gameplay GIFs for all agents
python -m evaluation play --algo dqn --checkpoint results/dqn/model.pth --gif results/gifs/dqn_demo.gif --demo-trials 16
python -m evaluation play --algo reinforce --checkpoint results/models/reinforce_policy.pth --gif results/gifs/reinforce_demo.gif --demo-trials 16
python -m evaluation play --algo a2c --checkpoint results/models/a2c_actor.pth --hidden 256 --gif results/gifs/a2c_demo.gif --demo-trials 16
```

## Fairness rules (for final report)

- Use shared env factory `make_env("LunarLander-v3", continuous=False)`
- Use the same seeds for all agents: `[0, 1, 2, 42, 123]`
- Use the same episode budget for all agents
- Log through `RunLogger` so every CSV has:
  `episode,reward,length,loss`
