# Math Evolution Agent — Phase 1

> **Paper:** [Two Failure Modes of Curriculum Reinforcement Learning: Stochastic Evaluation Gates and Reward Scale Mismatch](ARXIV_PLACEHOLDER) — Leon Katz, 2026

A PPO agent that teaches itself mathematics from scratch — no human demonstrations, no pre-training. Starting from single-digit addition, it progresses through a 13-level curriculum spanning arithmetic, algebra, and physics, learning entirely from reward signals.

The original configuration appeared permanently stuck at Multiplication. This repository documents the two failure modes responsible and the fixes that raise the median final level from Multiplication (semantic 3.4) to Kinematics (semantic 8.0) across 20 independent seeds.

---

## Results

| Configuration | Seeds | Median final level | Semantic level |
|---|---|---|---|
| Baseline (abs reward + stochastic gate) | 9 | Multiplication L5 | 3.4 |
| Fixed (rel reward + det gate) | 20 | Kinematics L11 | **7.65 ± 0.57** |
| Extended 30M steps | 3 | Energy L12 | 9.0 |

70% of seeds reach Kinematics (F = ma, d = vt) at 15M steps. All three 30M-step runs complete the full curriculum through Energy conservation.

---

## Quick Start

**Prerequisites:** Docker and Docker Compose.

```bash
# 1. Clone and configure
git clone https://github.com/leonkatz/math-evolution-agent-phase-1-published.git
cd math-evolution-agent-phase-1-published
cp .env.example .env
# Edit .env if desired (defaults match the paper's primary configuration)

# 2. Start training
docker compose up

# 3. Monitor at http://localhost:8081
```

Training runs for 15M steps by default (~12–24 hours depending on hardware). The dashboard at `http://localhost:8081` shows live progress, level transitions, and experiment history.

To run a specific seed or step count:
```bash
SEED=7 STEPS=30000000 docker compose up
```

To run the supervised baseline (§6.5):
```bash
docker compose run --rm trainer python baseline/train_supervised.py --seed 42
```

To evaluate a trained checkpoint:
```bash
docker compose run --rm trainer python evaluate.py --checkpoint /app/checkpoints/agent_final.pt
```

---

## Repository Structure

```
code/
  train.py                   # PPO training loop
  evaluate.py                # Evaluate a trained checkpoint
  agent/model.py             # Actor-Critic network + PPO update
  environments/math_env.py   # 13-level math curriculum environment
  baseline/train_supervised.py  # Supervised baseline (§6.5)
  analysis/                  # Figure generation scripts
  db/                        # Experiment logging (PostgreSQL)
  dashboard.py               # Training monitor (http://localhost:8081)
paper/
  paper.md                   # Full paper
  figures/                   # Paper figures
checkpoints/                 # Saved model weights (populated during training)
data/                        # Training logs and CSV exports
```

---

## Configuration

Copy `.env.example` to `.env` before running. Key settings:

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_PASSWORD` | `changeme` | Database password (any string) |
| `SEED` | `42` | Random seed — controls weight init and env sampling |
| `STEPS` | `15000000` | Training steps (15M = paper primary sweep) |
| `N_ENVS` | `1` | Parallel environments (1 = cleanest data, matches n_envs=1 sweep) |
| `DASHBOARD_PORT` | `8081` | Port for the training monitor |

---

## Reproducing Paper Results

**Finding 1 — Stochastic evaluation ceiling** (§6.1):
The baseline uses stochastic advancement evaluation. To reproduce the ceiling, edit `train.py` and set `USE_DETERMINISTIC_EVAL = False`.

**Finding 2 — Reward normalization** (§6.2):
The default `.env` uses relative reward. To reproduce the absolute reward baseline, set `--reward_mode absolute` in `docker-compose.yml`'s trainer command.

**20-seed primary sweep** (§6.2):
```bash
for seed in $(seq 1 20); do SEED=$seed docker compose up -d trainer; done
```

**30M extended runs** (§6.6):
```bash
SEED=42 STEPS=30000000 docker compose up
```

---

## Citation

```bibtex
@article{katz2026mathevolution,
  title   = {Two Failure Modes of Curriculum Reinforcement Learning:
             Stochastic Evaluation Gates and Reward Scale Mismatch},
  author  = {Katz, Leon},
  year    = {2026},
  note    = {arXiv:ARXIV_PLACEHOLDER}
}
```

---

## License

MIT
