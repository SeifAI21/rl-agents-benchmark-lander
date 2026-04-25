# RL Agents Benchmark — LunarLander

This repo contains:

- shared environment setup in `src/env_utils.py`
- DQN training code in `dqn/src/DQN/`
- evaluation and visualization tooling in `evaluation/` (Person 5 task)

## Current branch status (`evaluation-visualization`)

- DQN artifacts: real `seed42` model/log generated
- REINFORCE artifacts: real `seed42` model/log integrated
- A2C artifacts: pending teammate handoff (model + logs)

Until A2C arrives, report outputs are partially real:

- DQN: mixed (real seed42 + synthetic other seeds)
- REINFORCE: mixed (real seed42 + synthetic other seeds)
- A2C: currently synthetic placeholders

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
```

## Fairness rules (for final report)

- Use shared env factory `make_env("LunarLander-v3", continuous=False)`
- Use the same seeds for all agents: `[0, 1, 2, 42, 123]`
- Use the same episode budget for all agents
- Log through `RunLogger` so every CSV has:
  `episode,reward,length,loss`
