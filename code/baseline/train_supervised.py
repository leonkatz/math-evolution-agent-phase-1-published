"""
Supervised Baseline — Math Curriculum Agent

Identical architecture to the PPO agent (MathReasoningNetwork) but trained
with labeled (state, correct_answer) pairs and MSE loss instead of PPO.

This is the comparison experiment: does removing human-labeled examples
actually matter, or would supervised learning do just as well?

The hypothesis is that supervised learning will plateau earlier and generalize
less well — because it learns to memorize mappings from state to answer rather
than internalizing the underlying mathematical structure through reward shaping.

Usage:
  python baseline/train_supervised.py
  python baseline/train_supervised.py --steps 15000000 --seed 42
  python baseline/train_supervised.py --checkpoint baseline/logs/checkpoints_seed42/final.pt
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments import MathEvolutionEnv, LEVEL_NAMES
from agent.model import MathReasoningNetwork  # same architecture — apples-to-apples comparison


# ── CLI args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100_000,
                    help="Max gradient updates (each update sees BATCH_SIZE problems). "
                         "Training stops early if the final curriculum level is mastered.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to resume from")
args = parser.parse_args()

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE        = 2048        # labeled problems per update (generous — unlimited supply)
LEARNING_RATE     = 3e-4        # same as PPO agent
HIDDEN_DIM        = 256         # same as PPO agent
MASTERY_THRESHOLD = 0.85        # same mastery bar as PPO agent
MASTERY_EPISODES  = 200         # same deterministic eval as PPO agent
MASTERY_CHECK_INTERVAL = 2_000  # check every N updates (not steps — different unit here)
LOG_INTERVAL      = 500         # log every N updates
SAVE_INTERVAL     = 10_000      # save checkpoint every N updates

_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR   = os.path.join(_BASELINE_DIR, "logs")
LOG_FILE  = os.path.join(_BASELINE_DIR, "logs", f"supervised_seed{args.seed}.csv")
CKPT_DIR  = os.path.join(_BASELINE_DIR, "logs", f"checkpoints_seed{args.seed}")


# ── Data generation ──────────────────────────────────────────────────────────
def generate_batch(env, n_samples: int):
    """
    Sample n_samples problems from the environment.
    Returns (states, targets) where target = correct_answer / answer_scale,
    normalized to [-1, 1] — the same space as the actor_mean output.

    Key advantage over PPO: unlimited labeled data. We never run out of examples
    and there's no exploration noise — the correct answer is always known.
    """
    states  = np.empty((n_samples, 5), dtype=np.float32)
    targets = np.empty(n_samples,      dtype=np.float32)

    for i in range(n_samples):
        state, _ = env.reset()
        # Normalize correct answer to [-1, 1] — same space as tanh actor output
        target = np.clip(env.correct_answer / env.answer_scale, -1.0, 1.0)
        states[i]  = state
        targets[i] = target

    return torch.FloatTensor(states), torch.FloatTensor(targets)


# ── Mastery evaluation ───────────────────────────────────────────────────────
def eval_mastery(env, network, device, n_episodes: int = MASTERY_EPISODES) -> float:
    """
    Measure mastery using deterministic (noise-free) predictions.
    Identical in spirit to eval_deterministic_mastery() in train.py.

    Uses the network's actor_mean output directly — no stochastic sampling.
    This is the fair comparison: both supervised and PPO agents are evaluated
    the same way, using their deterministic best-guess answer.
    """
    saved_level = env.level
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, _, _ = network(state_t)      # actor_mean is tanh-bounded [-1, 1]
            action = np.array([mean.cpu().item()])
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    env.level = saved_level
    return float(np.mean(rewards))


# ── Training loop ────────────────────────────────────────────────────────────
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same architecture as PPO agent — only the training method differs
    network   = MathReasoningNetwork(state_dim=5, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=1e-5)

    start_level = 0

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        network.load_state_dict(ckpt["network"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_level = ckpt.get("level", 0)
        print(f"Resumed from {args.checkpoint} at level {start_level}")

    env = MathEvolutionEnv(
        start_level=start_level,
        mastery_threshold=MASTERY_THRESHOLD,
        mastery_window=200,
    )

    print("=" * 60)
    print("  SUPERVISED BASELINE — TRAINING START")
    print("=" * 60)
    print(f"  Architecture: same as PPO (MathReasoningNetwork, dim={HIDDEN_DIM})")
    print(f"  Training method: MSE loss on labeled (state, answer) pairs")
    print(f"  Batch size: {BATCH_SIZE:,} labeled problems per update")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"  Starting level: {start_level} ({LEVEL_NAMES[start_level]})")
    print("=" * 60)

    with open(LOG_FILE, "w") as f:
        f.write("update,level,level_name,avg_loss,det_mastery,elapsed_min,seed\n")

    loss_window = deque(maxlen=100)
    start_time  = time.time()
    det_mastery = 0.0
    last_mastery_check = -MASTERY_CHECK_INTERVAL

    for update in range(1, args.steps + 1):

        # Generate fresh batch — unlimited labeled data, no replay buffer needed
        states, targets = generate_batch(env, BATCH_SIZE)
        states  = states.to(device)
        targets = targets.to(device)

        # Forward: get actor_mean (deterministic prediction, tanh-bounded)
        network.train()
        mean, _, _ = network(states)
        mean = mean.squeeze(-1)   # (batch,)

        loss = F.mse_loss(mean, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 0.5)
        optimizer.step()

        loss_window.append(loss.item())

        # ── Mastery check ──────────────────────────────────────────────────
        if update - last_mastery_check >= MASTERY_CHECK_INTERVAL:
            last_mastery_check = update
            network.eval()
            det_mastery = eval_mastery(env, network, device)
            if det_mastery >= MASTERY_THRESHOLD:
                if env.level < len(LEVEL_NAMES) - 1:
                    old_level = env.level
                    env.advance_level()
                    print(f"\n  >>> LEVEL UP: {LEVEL_NAMES[old_level]} → {LEVEL_NAMES[env.level]} "
                          f"(mastery={det_mastery:.3f})\n")
                else:
                    # Final level mastered — curriculum complete, no need to keep training
                    elapsed = (time.time() - start_time) / 60
                    print(f"\n  ✓ CURRICULUM COMPLETE at update {update:,} | "
                          f"Level {env.level} ({LEVEL_NAMES[env.level]}) | "
                          f"Mastery {det_mastery:.3f} | {elapsed:.1f}min")
                    with open(LOG_FILE, "a") as f:
                        f.write(f"{update},{env.level},{LEVEL_NAMES[env.level]},"
                                f"{float(np.mean(loss_window)):.6f},{det_mastery:.4f},"
                                f"{elapsed:.2f},{args.seed}\n")
                    break

        # ── Logging ────────────────────────────────────────────────────────
        if update % LOG_INTERVAL == 0:
            elapsed    = (time.time() - start_time) / 60
            avg_loss   = float(np.mean(loss_window))
            level      = env.level
            print(
                f"Update {update:>8,} | "
                f"Level {level} ({LEVEL_NAMES[level][:18]:<18}) | "
                f"Loss {avg_loss:.5f} | "
                f"Mastery(det) {det_mastery:.3f} | "
                f"{elapsed:.1f}min"
            )
            with open(LOG_FILE, "a") as f:
                f.write(f"{update},{level},{LEVEL_NAMES[level]},"
                        f"{avg_loss:.6f},{det_mastery:.4f},{elapsed:.2f},{args.seed}\n")

        # ── Checkpoint ────────────────────────────────────────────────────
        if update % SAVE_INTERVAL == 0:
            path = os.path.join(CKPT_DIR, f"supervised_update{update}.pt")
            torch.save({
                "network":   network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "level":     env.level,
                "update":    update,
            }, path)

    # Final save
    final_path = os.path.join(CKPT_DIR, "final.pt")
    torch.save({
        "network":   network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "level":     env.level,
        "update":    args.steps,
    }, final_path)

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining complete in {elapsed:.1f} minutes.")
    print(f"Final level: {env.level} — {LEVEL_NAMES[env.level]}")
    print(f"Final deterministic mastery: {det_mastery:.3f}")


if __name__ == "__main__":
    main()
