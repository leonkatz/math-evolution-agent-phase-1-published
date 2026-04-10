"""
Evaluate the trained agent.

Shows the agent solving problems at each level it has learned,
displaying its predictions vs correct answers.

Run: python evaluate.py
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Ensure code/ is on sys.path so imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from environments import (MathEvolutionEnv, EnvConfig,
                          LEVEL_NAMES, LEVEL_NAMES_REFINED, LEVEL_NAMES_REFINED_V2,
                          LEVEL_NAMES_BEYOND_ENERGY, LEVEL_NAMES_REFINED_BEYOND,
                          LEVEL_NAMES_REFINED_V2_BEYOND, LEVEL_NAMES_PER_TABLE,
                          LEVEL_NAMES_PER_TABLE_BEYOND, LEVEL_NAMES_NO_CONCEPT)
from agent import PPOAgent

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="logs/checkpoints/agent_final.pt",
                    help="Path to checkpoint file")
parser.add_argument("--refined_mul", action="store_true", default=False)
parser.add_argument("--refined_div", action="store_true", default=False)
parser.add_argument("--beyond_energy", action="store_true", default=False)
parser.add_argument("--per_table_mul", action="store_true", default=False)
parser.add_argument("--no_concept_discovery", action="store_true", default=False)
args = parser.parse_args()

PROBLEMS_PER_LEVEL = 10

# Mirror train.py level name selection — must match how the checkpoint was trained
if args.no_concept_discovery:
    level_names = LEVEL_NAMES_NO_CONCEPT
elif args.per_table_mul and args.beyond_energy:
    level_names = LEVEL_NAMES_PER_TABLE_BEYOND
elif args.per_table_mul:
    level_names = LEVEL_NAMES_PER_TABLE
elif args.refined_mul and args.refined_div and args.beyond_energy:
    level_names = LEVEL_NAMES_REFINED_V2_BEYOND
elif args.refined_mul and args.refined_div:
    level_names = LEVEL_NAMES_REFINED_V2
elif args.refined_mul and args.beyond_energy:
    level_names = LEVEL_NAMES_REFINED_BEYOND
elif args.refined_mul:
    level_names = LEVEL_NAMES_REFINED
elif args.beyond_energy:
    level_names = LEVEL_NAMES_BEYOND_ENERGY
else:
    level_names = LEVEL_NAMES


def evaluate():
    agent = PPOAgent(state_dim=5, hidden_dim=256)

    if not os.path.exists(args.checkpoint):
        print(f"No checkpoint found at {args.checkpoint}. Train first with: python train.py")
        return

    agent.load(args.checkpoint)
    agent.network.eval()

    print("=" * 60)
    print("  MATH EVOLUTION AGENT — EVALUATION")
    print("=" * 60)

    for level in range(len(level_names)):
        env = MathEvolutionEnv(start_level=level, mastery_threshold=1.1,
                               refined_mul=args.refined_mul, refined_div=args.refined_div,
                               beyond_energy=args.beyond_energy, per_table_mul=args.per_table_mul,
                               no_concept_discovery=args.no_concept_discovery)  # won't level up
        env.level = level

        correct_count = 0
        total_error = 0.0

        print(f"\nLevel {level}: {level_names[level]}")
        print("-" * 40)

        for i in range(PROBLEMS_PER_LEVEL):
            state, _ = env.reset()
            action = agent.select_action(state, deterministic=True)
            action_env = np.array([float(action)])
            _, reward, _, _, info = env.step(action_env)

            correct = info["correct_answer"]
            predicted = info["predicted"]
            error = info["error"]
            total_error += error

            close = error < max(abs(correct) * 0.05, 0.5)  # within 5% or 0.5 absolute
            if close:
                correct_count += 1

            print(f"  Problem {i+1:2d}: "
                  f"Answer={correct:10.3f} | "
                  f"Predicted={predicted:10.3f} | "
                  f"Error={error:8.3f} | "
                  f"{'OK' if close else 'MISS'}")

        accuracy = correct_count / PROBLEMS_PER_LEVEL * 100
        avg_err = total_error / PROBLEMS_PER_LEVEL
        print(f"  Accuracy: {accuracy:.0f}% | Avg Error: {avg_err:.3f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    evaluate()
