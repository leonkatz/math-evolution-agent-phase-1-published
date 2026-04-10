"""
Item 1 — Stochastic vs Deterministic Mastery Gap Analysis

Loads training CSV logs and visualises the gap between:
  - det_mastery:   noise-free actor mean (what the agent truly learned)
  - stoch_mastery: average reward during stochastic PPO training (noisy signal)

The gap is the "false ceiling" — the amount by which standard stochastic
evaluation underestimates the agent's actual competence.

Requires the new CSV format (from updated train.py):
  step, level, level_name, avg_reward, det_mastery, stoch_mastery, entropy_coef, loss, ...

Usage:
  python analysis/plot_mastery_gap.py                      # reads logs/*.csv
  python analysis/plot_mastery_gap.py --logdir logs        # explicit path
  python analysis/plot_mastery_gap.py --seed 42            # single seed
  python analysis/plot_mastery_gap.py --save gap_plot.png  # save instead of show

Dependencies:
  pip install matplotlib pandas
"""

import os
import sys
import argparse
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
except ImportError:
    print("Missing dependencies. Run:  pip install matplotlib pandas")
    sys.exit(1)

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, default="logs",
                    help="Directory containing training_log_seed*.csv files")
parser.add_argument("--seed", type=int, default=None,
                    help="If set, plot only this seed")
parser.add_argument("--save", type=str, default=None,
                    help="Save figure to this path instead of showing interactively")
args = parser.parse_args()


def load_logs(logdir: str, seed_filter=None):
    pattern = os.path.join(logdir, "training_log_seed*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No CSV files found matching: {pattern}")
        sys.exit(1)

    dfs = {}
    for f in files:
        df = pd.read_csv(f)
        # Detect format: old CSVs have 'mastery' column (=det_mastery only)
        # New CSVs have 'det_mastery' AND 'stoch_mastery'
        if "stoch_mastery" not in df.columns:
            print(f"  Skipping {os.path.basename(f)} — old format (no stoch_mastery column)")
            print(f"  Re-run training with the updated train.py to get stoch_mastery data.")
            continue
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else os.path.basename(f)
        if seed_filter is not None and seed != seed_filter:
            continue
        dfs[seed] = df
        print(f"  Loaded seed={seed}: {len(df)} rows, max step={df['step'].max():,}")

    if not dfs:
        print("\nNo compatible logs found. The stoch_mastery column was added in the")
        print("updated train.py — it will be present in the exploration run and later runs.")
        sys.exit(1)
    return dfs


def plot_gap(dfs: dict, save_path=None):
    n_seeds = len(dfs)
    fig = plt.figure(figsize=(14, 4 * (n_seeds + 1)))
    gs  = gridspec.GridSpec(n_seeds + 1, 1, hspace=0.45)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    all_gaps = []

    for idx, (seed, df) in enumerate(sorted(dfs.items())):
        ax = fig.add_subplot(gs[idx])

        steps      = df["step"].values / 1e6   # millions
        det        = df["det_mastery"].values
        stoch      = df["stoch_mastery"].values
        gap        = det - stoch
        all_gaps.append((steps, gap))

        c = colors[idx % len(colors)]
        ax.plot(steps, det,   color=c, lw=2,    label="Deterministic mastery (true)")
        ax.plot(steps, stoch, color=c, lw=2, ls="--", alpha=0.6, label="Stochastic mastery (measured)")
        ax.fill_between(steps, stoch, det, alpha=0.15, color=c, label=f"Gap (mean={gap.mean():.3f})")

        # Level-up markers
        level_changes = df[df["level"].diff() > 0]
        for _, row in level_changes.iterrows():
            ax.axvline(row["step"] / 1e6, color="gray", ls=":", lw=1, alpha=0.7)
            ax.text(row["step"] / 1e6, 0.02, f"L{int(row['level'])}", fontsize=7,
                    ha="center", color="gray")

        ax.axhline(0.85, color="red", ls=":", lw=1, alpha=0.5, label="Mastery threshold (0.85)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mastery score")
        ax.set_title(f"Seed {seed} — stochastic vs deterministic mastery")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Aggregate gap plot ────────────────────────────────────────────────────
    ax_agg = fig.add_subplot(gs[n_seeds])

    for idx, (steps, gap) in enumerate(all_gaps):
        seed = sorted(dfs.keys())[idx]
        ax_agg.plot(steps, gap, alpha=0.5, lw=1.5, label=f"Seed {seed}")

    # Compute mean gap across seeds on a common step grid
    if len(all_gaps) > 1:
        max_steps = max(s[-1] for s, _ in all_gaps)
        grid      = np.linspace(0, max_steps, 500)
        interp    = [np.interp(grid, s, g) for s, g in all_gaps]
        mean_gap  = np.mean(interp, axis=0)
        std_gap   = np.std(interp, axis=0)
        ax_agg.plot(grid, mean_gap, "k-", lw=2.5, label=f"Mean gap (avg={mean_gap.mean():.3f})")
        ax_agg.fill_between(grid, mean_gap - std_gap, mean_gap + std_gap,
                             alpha=0.15, color="black")

    ax_agg.set_xlabel("Training steps (millions)")
    ax_agg.set_ylabel("Gap (det − stoch mastery)")
    ax_agg.set_title("Aggregate: stochastic false ceiling magnitude across all seeds")
    ax_agg.axhline(0, color="gray", ls="-", lw=0.5)
    ax_agg.legend(fontsize=8)
    ax_agg.grid(True, alpha=0.3)

    plt.suptitle("Finding 1: Stochastic Mastery False Ceiling\n"
                 "PPO exploration noise systematically underestimates true agent competence",
                 fontsize=12, fontweight="bold", y=1.01)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print(f"Loading logs from: {args.logdir}")
    dfs = load_logs(args.logdir, seed_filter=args.seed)
    print(f"Plotting {len(dfs)} seed(s)...")
    plot_gap(dfs, save_path=args.save)
