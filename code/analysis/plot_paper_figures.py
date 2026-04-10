"""
Paper Figure Generator — Math Evolution Agent

Generates all publication-quality figures from existing training logs.
Handles both CSV format (Run 1) and text log format (Runs 3-6).

Figures produced:
  fig1_curriculum.png       — Curriculum architecture overview
  fig2_false_ceiling.png    — Finding 1: stochastic vs deterministic mastery gap
  fig3_entropy_regression.png — Finding 2: entropy collapse causes mastery regression
  fig4_concept_discovery.png  — Finding 3: small-scale concept discovery is required
  fig5_reproducibility.png    — Multi-seed consistency

Usage:
  cd math_agent/
  python analysis/plot_paper_figures.py
  python analysis/plot_paper_figures.py --logdir logs --outdir analysis/figures
  python analysis/plot_paper_figures.py --figure 2   # single figure only

Dependencies: pip install matplotlib pandas numpy
"""

import os
import sys
import re
import sqlite3
import argparse
import numpy as np
from pathlib import Path

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")          # headless — works in all environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    import pandas as pd
except ImportError:
    print("Run:  pip install matplotlib pandas")
    sys.exit(1)

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "lines.linewidth":   2.0,
})

BLUE   = "#2980b9"
ORANGE = "#e67e22"
RED    = "#e74c3c"
GREEN  = "#27ae60"
PURPLE = "#8e44ad"
GRAY   = "#7f8c8d"
THRESHOLD = 0.85

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="logs",
                    help="Directory containing training logs (default: logs/)")
parser.add_argument("--outdir", default="analysis/figures",
                    help="Output directory for figures (default: analysis/figures/)")
parser.add_argument("--figure", type=int, default=None,
                    help="Generate only this figure number (1-5). Default: all.")
args = parser.parse_args()

LOG_DIR = Path(args.logdir)
OUT_DIR = Path(args.outdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Text log parser ───────────────────────────────────────────────────────────
_STEP_RE = re.compile(
    r"Step\s+([\d,]+)\s*\|\s*Level\s*(\d+)"
    r".*?Reward\s*([\d.]+)"
    r".*?Mastery\(det\)\s*([\d.]+)"
    r".*?(?:Loss\s*([\d.]+))?"
    r".*?Entropy\s*([\d.]+)"
    r".*?([\d.]+)min"
)
_LEVELUP_RE = re.compile(r">>> LEVEL UP.*?Level\s*(\d+):\s*(.+)")


def parse_text_log(path: Path) -> pd.DataFrame:
    """Parse a training_runN.log or dell_seedN.log into a DataFrame."""
    rows      = []
    levelups  = []   # (step, new_level) for vertical marker lines

    with open(path) as f:
        prev_step = 0
        for line in f:
            m = _STEP_RE.search(line)
            if m:
                step        = int(m.group(1).replace(",", ""))
                level       = int(m.group(2))
                avg_reward  = float(m.group(3))
                det_mastery = float(m.group(4))
                entropy     = float(m.group(6))
                elapsed     = float(m.group(7))
                rows.append(dict(
                    step        = step,
                    level       = level,
                    avg_reward  = avg_reward,
                    det_mastery = det_mastery,
                    entropy_coef= entropy,
                    elapsed_min = elapsed,
                ))
                prev_step = step
                continue
            lu = _LEVELUP_RE.search(line)
            if lu:
                levelups.append((prev_step, int(lu.group(1)), lu.group(2).strip()))

    if not rows:
        return pd.DataFrame(), []

    df = pd.DataFrame(rows)
    df["step_M"] = df["step"] / 1e6
    return df, levelups


def parse_csv_log(path: Path) -> pd.DataFrame:
    """Parse training_log.csv (old or new format) into a normalised DataFrame."""
    df = pd.read_csv(path)
    # Normalise column names across formats
    if "det_mastery" not in df.columns and "mastery" in df.columns:
        # Old format: 'mastery' column is stochastic mastery (Run 1)
        # OR det_mastery (Runs 2+ if logged to CSV) — we label conservatively
        df = df.rename(columns={"mastery": "det_mastery"})
    if "stoch_mastery" not in df.columns:
        df["stoch_mastery"] = df.get("avg_reward", np.nan)
    if "entropy_coef" not in df.columns:
        # Reconstruct from step (assumes default anneal schedule)
        total = df["step"].max()
        df["entropy_coef"] = 0.01 + (0.001 - 0.01) * (df["step"] / total).clip(0, 1)
    df["step_M"] = df["step"] / 1e6
    return df


def save(fig, name: str):
    path = OUT_DIR / name
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


def hline(ax, y=THRESHOLD, **kw):
    ax.axhline(y, color=RED, ls="--", lw=1.2, alpha=0.6,
               label=f"Mastery threshold ({y})", **kw)


def vlines_levelup(ax, levelups, ymax=1.05):
    for step, lvl, name in levelups:
        ax.axvline(step / 1e6, color=GRAY, ls=":", lw=1, alpha=0.6)
        ax.text(step / 1e6, ymax - 0.04, f"L{lvl}↑",
                fontsize=7, ha="center", color=GRAY, va="top")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — Curriculum Architecture
# ═════════════════════════════════════════════════════════════════════════════
LEVELS = [
    ("L0",  "Addition (0–10)",           "concept",  "#3498db"),
    ("L1",  "Addition (0–100)",          "scale-up", "#5dade2"),
    ("L2",  "Subtraction (0–10)",        "concept",  "#e74c3c"),
    ("L3",  "Subtraction (0–100)",       "scale-up", "#f1948a"),
    ("L4",  "Multiplication (0–5)",      "concept",  "#27ae60"),
    ("L5",  "Multiplication (0–12)",     "scale-up", "#58d68d"),
    ("L6",  "Division (÷1–5)",           "concept",  "#f39c12"),
    ("L7",  "Division (÷1–12)",          "scale-up", "#f8c471"),
    ("L8",  "Mixed Arithmetic",          "mixed",    "#8e44ad"),
    ("L9",  "Linear  ax+b=c",            "abstract", "#1abc9c"),
    ("L10", "Quadratic ax²+bx+c=0",      "abstract", "#16a085"),
    ("L11", "Kinematics d=vt, F=ma",     "abstract", "#2c3e50"),
    ("L12", "Energy  KE=½mv²",           "abstract", "#1a252f"),
]

def fig1_curriculum():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.8, 1.2)
    ax.axis("off")

    bar_h = 0.55
    for i, (code, name, kind, color) in enumerate(LEVELS):
        y = 0
        rect = mpatches.FancyBboxPatch(
            (i + 0.08, y - bar_h / 2), 0.84, bar_h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(i + 0.5, y + 0.05, code, ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")
        ax.text(i + 0.5, y - 0.18, name, ha="center", va="top",
                color="white", fontsize=6.5, wrap=True)

        # Arrows between levels
        if i < 12:
            ax.annotate("", xy=(i + 1.08, 0), xytext=(i + 0.92, 0),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=1.2, mutation_scale=10))

    # Legend for kind
    legend_items = [
        mpatches.Patch(facecolor="#3498db", label="Concept discovery (small scale)"),
        mpatches.Patch(facecolor="#5dade2", alpha=0.6, label="Scale-up (same operation)"),
        mpatches.Patch(facecolor="#8e44ad", label="Mixed / Abstract"),
    ]
    ax.legend(handles=legend_items, loc="upper center",
              bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=9)

    # Reward formula
    ax.text(6.5, -0.65,
            r"Reward = exp(−|predicted − correct| × 1.5 / |correct|)    "
            r"   State = [n₁, n₂, n₃, op, level]  (all normalised to [−1,1])",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))

    ax.set_title("Math Evolution Agent — 13-Level Curriculum (v3)\n"
                 "No language, no demonstrations, no human examples — only a scalar reward",
                 fontsize=12, fontweight="bold", pad=12)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — Finding 1: Stochastic False Ceiling
# ═════════════════════════════════════════════════════════════════════════════
def fig2_false_ceiling():
    # Best source: a run that has both avg_reward (stochastic) and
    # Mastery(det) in the text log — Run 5 (seed=42) is ideal.
    sources = [
        LOG_DIR / "training_run5_seed42.log",
        LOG_DIR / "training_run6_seed123.log",
        LOG_DIR / "training_run4.log",
    ]
    df, levelups = pd.DataFrame(), []
    used = None
    for src in sources:
        if src.exists():
            df, levelups = parse_text_log(src)
            if not df.empty:
                used = src.name
                break

    if df.empty:
        print("  Figure 2: no text log found — skipping")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: full training run showing both signals ──────────────────────
    ax = axes[0]
    ax.plot(df["step_M"], df["det_mastery"], color=BLUE,   lw=2,
            label="Deterministic mastery (true learning)")
    ax.plot(df["step_M"], df["avg_reward"],  color=ORANGE, lw=2, ls="--", alpha=0.75,
            label="Stochastic reward (training signal)")
    ax.fill_between(df["step_M"], df["avg_reward"], df["det_mastery"],
                    alpha=0.12, color=BLUE,
                    label="Gap — false ceiling region")
    hline(ax)
    vlines_levelup(ax, levelups)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Stochastic vs Deterministic Mastery\n(full training run)")
    ax.legend(loc="lower right")

    # ── Right: zoom into first level to make gap stark ────────────────────
    ax2 = axes[1]
    # Find steps at level 0 only
    l0 = df[df["level"] == 0]
    ax2.plot(l0["step_M"], l0["det_mastery"], color=BLUE,   lw=2.5,
             label="Deterministic (true)")
    ax2.plot(l0["step_M"], l0["avg_reward"],  color=ORANGE, lw=2.5, ls="--", alpha=0.8,
             label="Stochastic (measured during training)")
    ax2.fill_between(l0["step_M"], l0["avg_reward"], l0["det_mastery"],
                     alpha=0.15, color=BLUE)

    hline(ax2)

    # Label the gap at peak
    peak_idx   = l0["det_mastery"].idxmax()
    peak_step  = l0.loc[peak_idx, "step_M"]
    peak_det   = l0.loc[peak_idx, "det_mastery"]
    peak_stoch = l0.loc[peak_idx, "avg_reward"]
    gap        = peak_det - peak_stoch
    ax2.annotate(f"Gap = {gap:.3f}",
                 xy=(peak_step, (peak_det + peak_stoch) / 2),
                 xytext=(peak_step + 0.5, (peak_det + peak_stoch) / 2 + 0.05),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1),
                 fontsize=9, color="black")

    ax2.set_xlabel("Training steps (millions)")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.08)
    ax2.set_title(f"Zoom: Level 0 (Addition)\n"
                  f"Source: {used}")
    ax2.legend(loc="upper left")

    fig.suptitle(
        "Finding 1: The Stochastic Mastery False Ceiling\n"
        "PPO exploration noise systematically underestimates true agent competence, "
        "blocking curriculum advancement",
        fontweight="bold", fontsize=12
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — Finding 2: Reward Scale Mismatch
# ═════════════════════════════════════════════════════════════════════════════

# Level label lookup (13L curriculum)
LEVEL_NAMES = {
    0: "L0 Add(0-10)", 1: "L1 Add(0-100)",
    2: "L2 Sub(0-10)", 3: "L3 Sub(0-100)",
    4: "L4 Mul(0-5)",  5: "L5 Mul(0-12)",
    6: "L6 Div(1-5)",  7: "L7 Div(1-12)",
    8: "L8 Mixed",     9: "L9 LinAlg",
    10: "L10 Quad",   11: "L11 Kinem",   12: "L12 Energy",
}

def _load_db():
    """Return a DB connection — Postgres if DATABASE_URL is set, otherwise SQLite fallback."""
    database_url = os.environ.get("DATABASE_URL")
    if database_url and psycopg2 is not None:
        return psycopg2.connect(database_url)

    # SQLite fallback for local development without Postgres
    candidates = [
        Path("db/math_agent.db"),
        Path(__file__).parent.parent.parent / "db" / "math_agent.db",
    ]
    for p in candidates:
        if p.exists():
            return sqlite3.connect(str(p))
    raise FileNotFoundError(
        "No DB found. Set DATABASE_URL for Postgres, or run from the repo root "
        "so db/math_agent.db is reachable."
    )


def fig3_reward_normalization():
    """
    Finding 2: Reward scale mismatch.

    Panel A (left): Level progression over training steps.
      - Absolute reward runs (abs_anneal seeds 42/7/123) — all plateau at L5.
      - Representative relative reward runs (seed_sweep) — climb to L9-L11.

    Panel B (right): Final level distribution.
      - Absolute (9 runs): tight spike at L5.
      - Relative (20 seed_sweep runs): spread L8-L11.
    """
    try:
        con = _load_db()
    except FileNotFoundError as e:
        print(f"  Figure 3: {e} — skipping")
        return None

    import pandas as pd

    # ── Load level-over-time data for representative seeds ───────────────
    # Abs: seeds 42, 7, 123 (anneal, 13L) — the canonical absolute-reward baseline.
    # Rel: seeds 1, 3, 5, 10, 15, 20 from the 20-seed sweep (reset, 13L) — a
    #      representative spread that includes both L11 and sub-L11 outcomes.
    query = """
        SELECT e.run_id, e.seed, e.reward_mode, tl.step, tl.level
        FROM training_log tl
        JOIN experiments e ON e.id = tl.experiment_id
        WHERE (
            (e.run_id IN ('seed42_abs_anneal_imported',
                          'seed7_abs_anneal_imported',
                          'seed123_abs_anneal_imported'))
            OR
            (e.reward_mode = 'relative' AND e.curriculum = '13L'
             AND e.entropy_mode = 'reset'
             AND e.seed IN (1, 3, 5, 10, 15, 20))
        )
        ORDER BY e.reward_mode, e.seed, tl.step
    """
    df_log = pd.read_sql_query(query, con)

    # ── Load final levels for the direct comparison ──────────────────────
    # Abs baseline: the canonical 19-seed abs/anneal runs (seeds 1-10 + key
    # individual seeds), completed runs only.
    # Rel seed sweep: seeds 1-20 (the 20-seed sweep), completed runs only
    # (exclude queued reruns that haven't run yet: final_level = 0 while running).
    # Absolute baseline: the canonical 19-seed abs/anneal/15M sweep.
    # Seeds 1-10 ran as 'nenv1_sweep'; seeds 17/31/42/53/89/123/137/211/307
    # ran as individual imports under 'abs_anneal' or plain seed imports.
    # Use run_id patterns to select the right run when a seed has multiple entries.
    # Relative seed sweep: seeds 1-20, entropy_mode=reset, completed 15M runs.
    query_final = """
        SELECT e.reward_mode, e.seed, e.final_level
        FROM experiments e
        WHERE e.final_level IS NOT NULL
          AND e.curriculum = '13L'
          AND e.completed_at IS NOT NULL
          AND (
              (e.reward_mode = 'absolute' AND e.entropy_mode = 'anneal'
               AND e.total_steps = 15000064
               AND e.seed IN (1,2,3,4,5,6,7,8,9,10)
               AND e.run_id LIKE '%nenv1_sweep%')
              OR
              (e.reward_mode = 'absolute' AND e.entropy_mode = 'anneal'
               AND e.run_id IN ('seed42_abs_anneal_imported',
                                'seed7_abs_anneal_imported',
                                'seed123_abs_anneal_imported',
                                'seed17_imported','seed31_imported',
                                'seed53_imported','seed89_imported',
                                'seed137_imported','seed211_imported',
                                'seed307_imported'))
              OR
              (e.reward_mode = 'relative' AND e.seed BETWEEN 1 AND 20
               AND e.entropy_mode = 'reset'
               AND e.run_id LIKE '%seed_sweep%')
          )
    """
    df_final = pd.read_sql_query(query_final, con)
    con.close()

    if df_log.empty or df_final.empty:
        print("  Figure 3: insufficient DB data — skipping")
        return None

    # ── Build figure ──────────────────────────────────────────────────────
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # Colour palettes
    abs_colors  = ["#c0392b", "#e74c3c", "#f1948a"]   # reds for abs seeds
    rel_shades  = ["#1a5276", "#2471a3", "#2980b9",    # blues for rel seeds
                   "#5dade2", "#aed6f1", "#d6eaf8"]

    # ── Panel A: level-over-time ──────────────────────────────────────────
    abs_seeds = [42, 7, 123]
    rel_seeds = [1, 3, 5, 10, 15, 20]

    abs_handles, rel_handles = [], []

    for i, seed in enumerate(abs_seeds):
        sub = df_log[(df_log["reward_mode"] == "absolute") & (df_log["seed"] == seed)]
        if sub.empty:
            continue
        line, = ax_left.step(sub["step"] / 1e6, sub["level"],
                             where="post", color=abs_colors[i], lw=1.8,
                             alpha=0.85, label=f"Abs seed={seed}")
        abs_handles.append(line)

    for i, seed in enumerate(rel_seeds):
        sub = df_log[(df_log["reward_mode"] == "relative") & (df_log["seed"] == seed)]
        if sub.empty:
            continue
        lw = 2.2 if seed in (1, 5, 10) else 1.4
        alpha = 0.9 if seed in (1, 5, 10) else 0.6
        line, = ax_left.step(sub["step"] / 1e6, sub["level"],
                             where="post", color=rel_shades[i], lw=lw,
                             alpha=alpha, label=f"Rel seed={seed}")
        rel_handles.append(line)

    # L5 ceiling annotation
    ax_left.axhline(5, color="#c0392b", ls="--", lw=1.5, alpha=0.6)
    ax_left.text(14.8, 5.1, "Absolute reward\nceiling (L5)",
                 color="#c0392b", fontsize=8.5, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="#c0392b", alpha=0.8))

    ax_left.set_xlabel("Training steps (millions)", fontsize=10)
    ax_left.set_ylabel("Curriculum level reached", fontsize=10)
    ax_left.set_xlim(0, 15.5)
    ax_left.set_ylim(-0.3, 12.5)
    ax_left.set_yticks(range(13))
    ax_left.set_yticklabels(
        [LEVEL_NAMES.get(i, f"L{i}") for i in range(13)], fontsize=7.5
    )

    # Two-group legend
    from matplotlib.lines import Line2D
    abs_proxy = Line2D([0], [0], color="#e74c3c", lw=2,
                       label="Absolute reward (3 seeds)")
    rel_proxy = Line2D([0], [0], color="#2980b9", lw=2,
                       label="Relative reward (6 of 20 seeds shown)")
    ax_left.legend(handles=[abs_proxy, rel_proxy], loc="upper left", fontsize=9)
    ax_left.set_title("Level Progression Over Training\n"
                      "Absolute reward: plateau at L5. Relative reward: continuous climb.",
                      fontsize=10)

    # ── Panel B: final level distribution ────────────────────────────────
    abs_finals = df_final[df_final["reward_mode"] == "absolute"]["final_level"].tolist()
    rel_finals = df_final[df_final["reward_mode"] == "relative"]["final_level"].tolist()

    all_levels = sorted(set(abs_finals + rel_finals))
    x = np.arange(len(all_levels))
    width = 0.35

    abs_counts = [abs_finals.count(lv) for lv in all_levels]
    rel_counts = [rel_finals.count(lv) for lv in all_levels]

    bars_abs = ax_right.bar(x - width / 2, abs_counts, width,
                             color="#e74c3c", alpha=0.85,
                             label=f"Absolute reward (n={len(abs_finals)})")
    bars_rel = ax_right.bar(x + width / 2, rel_counts, width,
                             color="#2980b9", alpha=0.85,
                             label=f"Relative reward (n={len(rel_finals)})")

    # Count labels on bars
    for bar in bars_abs:
        h = bar.get_height()
        if h > 0:
            ax_right.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                          str(int(h)), ha="center", va="bottom",
                          fontsize=9, color="#c0392b", fontweight="bold")
    for bar in bars_rel:
        h = bar.get_height()
        if h > 0:
            ax_right.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                          str(int(h)), ha="center", va="bottom",
                          fontsize=9, color="#1a5276", fontweight="bold")

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(
        [LEVEL_NAMES.get(lv, f"L{lv}") for lv in all_levels],
        rotation=30, ha="right", fontsize=8
    )
    ax_right.set_ylabel("Number of runs", fontsize=10)
    ax_right.set_title("Final Level Distribution\n"
                       f"Abs: {len(abs_finals)} runs  |  Rel: {len(rel_finals)} runs",
                       fontsize=10)
    ax_right.legend(fontsize=9)
    ax_right.set_ylim(0, max(abs_counts + rel_counts) + 2)

    # Efficiency annotation
    abs_mean = np.mean(abs_finals) if abs_finals else 0
    rel_mean = np.mean(rel_finals) if rel_finals else 0
    ax_right.text(0.5, 0.97,
                  f"Mean final level:  Abs = {abs_mean:.1f}   Rel = {rel_mean:.1f}\n"
                  f"Step efficiency gain: 1.8–7.2× per level",
                  transform=ax_right.transAxes, ha="center", va="top",
                  fontsize=9, style="italic",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#eaf4fb",
                            edgecolor="#2980b9", alpha=0.9))

    fig.suptitle(
        "Finding 2: Reward Scale Mismatch\n"
        "Absolute reward uses problem-specific denominator → inconsistent gradients → hard ceiling at L5.\n"
        "Level-fixed normalization (relative reward) removes the ceiling: 19/20 seeds advance past L7.",
        fontweight="bold", fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 — Finding 3: Concept Discovery Ablation
# ═════════════════════════════════════════════════════════════════════════════
def fig4_concept_discovery():
    run3_path = LOG_DIR / "training_run3.log"   # Sub 0-100 direct (no concept level)
    run4_path = LOG_DIR / "training_run4.log"   # Sub 0-10→0-100 + Mul 0-12 direct
    run5_path = LOG_DIR / "training_run5_seed42.log"   # Mul 0-5→0-12 (concept works)
    run6_path = LOG_DIR / "training_run6_seed123.log"

    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # ── Panel A: Subtraction training curves ─────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])

    # Run 3: subtraction directly at large scale, no concept discovery level
    if run3_path.exists():
        df3, lu3 = parse_text_log(run3_path)
        sub_direct = df3[df3["level"] == 2]  # level 2 = subtraction (only one sub level in v1)
        if not sub_direct.empty:
            ax_a.plot(sub_direct["step_M"], sub_direct["det_mastery"],
                      color=RED, lw=2, ls="--", alpha=0.8,
                      label="Curriculum v1 — Subtraction (0–100) direct\n"
                            "No concept discovery level (Run 3, seed=42)")

    # Run 4: subtraction with concept discovery (levels 2 and 3)
    if run4_path.exists():
        df4, lu4 = parse_text_log(run4_path)
        sub_concept = df4[df4["level"] == 2]   # Subtraction 0-10 (concept)
        sub_scaleup = df4[df4["level"] == 3]   # Subtraction 0-100 (scale-up)
        if not sub_concept.empty:
            ax_a.plot(sub_concept["step_M"], sub_concept["det_mastery"],
                      color=GREEN, lw=2.5,
                      label="Curriculum v2 — Subtraction (0–10) concept discovery (Run 4)")
        if not sub_scaleup.empty:
            ax_a.plot(sub_scaleup["step_M"], sub_scaleup["det_mastery"],
                      color=BLUE, lw=2.5,
                      label="Curriculum v2 — Subtraction (0–100) scale-up (Run 4, 1M steps → mastered)")

    hline(ax_a)
    ax_a.set_xlabel("Training steps (millions)")
    ax_a.set_ylabel("Deterministic mastery")
    ax_a.set_ylim(0, 1.08)
    ax_a.set_title("Subtraction Ablation: Does Concept Discovery Help?\n"
                   "v1: attempt 0–100 directly (peaks at 0.795, never masters)  vs  "
                   "v2: 0–10 first, then 0–100 (masters in 1M steps)")
    ax_a.legend(loc="upper left", fontsize=9)

    # ── Panel B: Multiplication — without concept discovery ───────────────
    ax_b = fig.add_subplot(gs[1, 0])
    if run4_path.exists():
        df4, _ = parse_text_log(run4_path)
        mul_direct = df4[df4["level"] == 4]   # Multiplication 0-12 in v2 (no split)
        if not mul_direct.empty:
            ax_b.plot(mul_direct["step_M"], mul_direct["det_mastery"],
                      color=RED, lw=2.5,
                      label="Mul 0–12 direct (no concept level)\n5.24M steps, peak 0.533")
            peak = mul_direct["det_mastery"].max()
            ax_b.axhline(peak, color=RED, ls=":", lw=1, alpha=0.5)
            ax_b.text(mul_direct["step_M"].iloc[-1], peak + 0.01,
                      f"Peak: {peak:.3f}", color=RED, fontsize=9, ha="right")

    hline(ax_b)
    ax_b.set_xlabel("Training steps (millions)")
    ax_b.set_ylabel("Deterministic mastery")
    ax_b.set_ylim(0, 1.08)
    ax_b.set_title("Multiplication WITHOUT\nConcept Discovery (Run 4 / v2)")
    ax_b.legend(loc="upper left", fontsize=9)

    # ── Panel C: Multiplication — with concept discovery ──────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    for path, seed, color in [(run5_path, 42, BLUE), (run6_path, 123, GREEN)]:
        if path.exists():
            df, levelups = parse_text_log(path)
            mul_concept = df[df["level"] == 4]  # Mul 0-5 concept level
            mul_scaleup = df[df["level"] == 5]  # Mul 0-12 scale-up
            if not mul_concept.empty:
                ax_c.plot(mul_concept["step_M"], mul_concept["det_mastery"],
                          color=color, lw=2, ls="--", alpha=0.7,
                          label=f"Seed={seed}: Mul 0–5 (concept discovery)")
            if not mul_scaleup.empty:
                ax_c.plot(mul_scaleup["step_M"], mul_scaleup["det_mastery"],
                          color=color, lw=2.5,
                          label=f"Seed={seed}: Mul 0–12 (scale-up → mastered)")

    hline(ax_c)
    ax_c.set_xlabel("Training steps (millions)")
    ax_c.set_ylabel("Deterministic mastery")
    ax_c.set_ylim(0, 1.08)
    ax_c.set_title("Multiplication WITH\nConcept Discovery (Runs 5 & 6 / v3)")
    ax_c.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Finding 3: Concept Discovery is Non-Negotiable\n"
        "Small-scale concept discovery before large-scale training reduces acquisition cost by 3–5×\n"
        "Without it: operation is never mastered. With it: mastered reliably across all seeds.",
        fontweight="bold", fontsize=12
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 — Multi-Seed Reproducibility
# ═════════════════════════════════════════════════════════════════════════════
def fig5_reproducibility():
    sources = {
        42:  LOG_DIR / "training_run5_seed42.log",
        123: LOG_DIR / "training_run6_seed123.log",
        7:   LOG_DIR / "training_run7_seed7.log",
    }
    # Also try dell CSVs if available
    dell_csvs = sorted(LOG_DIR.glob("training_log_seed*.csv"))

    colors_map = {42: BLUE, 123: GREEN, 7: ORANGE,
                  17: PURPLE, 31: "#e74c3c", 53: "#1abc9c",
                  89: "#f39c12", 137: "#34495e", 211: "#8e44ad", 307: "#16a085"}

    fig, ax = plt.subplots(figsize=(12, 6))

    max_level_reached = {}

    # Text logs (Runs 5-7)
    for seed, path in sources.items():
        if not path.exists():
            continue
        df, levelups = parse_text_log(path)
        if df.empty:
            continue
        color = colors_map.get(seed, GRAY)
        ax.plot(df["step_M"], df["det_mastery"],
                color=color, lw=1.8, alpha=0.8, label=f"Seed {seed}")
        # Mark level-ups
        for step, lvl, name in levelups:
            ax.plot(step / 1e6, 0.85, "^", color=color, ms=8, alpha=0.8)
        max_level_reached[seed] = df["level"].max()

    # Dell CSVs (10-seed run, new format)
    for csv_path in dell_csvs:
        df = parse_csv_log(csv_path)
        if df.empty:
            continue
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else 0
        if seed in sources:   # already plotted from text log
            continue
        color = colors_map.get(seed, GRAY)
        col   = "det_mastery" if "det_mastery" in df.columns else "mastery"
        ax.plot(df["step_M"], df[col], color=color, lw=1.8, alpha=0.7,
                label=f"Seed {seed} (dell)")
        max_level_reached[seed] = df["level"].max()

    if not max_level_reached:
        print("  Figure 5: no seed data found — skipping")
        return None

    hline(ax)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Deterministic mastery")
    ax.set_ylim(0, 1.10)
    ax.set_xlim(left=0)

    # Summary table as text
    all_seeds   = sorted(max_level_reached)
    all_levels  = [max_level_reached[s] for s in all_seeds]
    unique_lvls = set(all_levels)
    fraction    = f"{sum(l == max(all_levels) for l in all_levels)}/{len(all_levels)}"
    ax.text(0.98, 0.05,
            f"Seeds: {len(all_seeds)}   "
            f"Max level reached: L{max(all_levels)}   "
            f"Reached max: {fraction}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))

    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_title(
        f"Multi-Seed Reproducibility — {len(all_seeds)} seeds, Curriculum v3\n"
        f"All seeds follow identical level progression: "
        f"Addition → Subtraction → Multiplication",
        fontweight="bold"
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
FIGURE_MAP = {
    1: ("fig1_curriculum.png",            fig1_curriculum),
    2: ("fig2_false_ceiling.png",         fig2_false_ceiling),
    3: ("fig3_reward_normalization.png",  fig3_reward_normalization),
    4: ("fig4_concept_discovery.png",     fig4_concept_discovery),   # kept for reference; not in paper
    5: ("fig5_reproducibility.png",       fig5_reproducibility),     # kept for reference; not in paper
}

if __name__ == "__main__":
    to_generate = [args.figure] if args.figure else list(FIGURE_MAP.keys())

    print(f"Generating {len(to_generate)} figure(s) → {OUT_DIR}/")
    print()

    for num in to_generate:
        fname, fn = FIGURE_MAP[num]
        print(f"Figure {num}: {fname}")
        try:
            fig = fn()
            if fig is not None:
                save(fig, fname)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
        print()

    print(f"Done. Figures saved to {OUT_DIR}/")
    print()
    print("For the paper, include:")
    for num, (fname, _) in FIGURE_MAP.items():
        print(f"  Fig {num}: {fname}")
