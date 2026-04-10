"""
Training Script — Math Evolution Agent

The agent interacts with the math environment, collects experience,
and updates its neural network weights through PPO.

Cumulative learning: pass a checkpoint to resume from a previous run.
  python train.py                                      # start fresh
  python train.py --checkpoint logs/checkpoints/agent_final.pt        # resume
  python train.py --checkpoint agent_final.pt --level 3  # resume at level

Parallel environments (4x faster, more stable gradients):
  python train.py --n_envs 4                           # 4 parallel envs (default)
  python train.py --n_envs 1                           # single env (original behavior)

Multi-seed runs for reproducibility:
  python train.py --seed 42
  python train.py --seed 123
  python train.py --seed 7

No human ever teaches it. It evolves understanding from pure math feedback.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import psutil
from collections import deque
from pathlib import Path
from gymnasium.vector import SyncVectorEnv

# Ensure code/ is on sys.path so imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from environments import (MathEvolutionEnv, EnvConfig,
                          LEVEL_NAMES, LEVEL_NAMES_REFINED, LEVEL_NAMES_REFINED_V2,
                          LEVEL_NAMES_BEYOND_ENERGY, LEVEL_NAMES_REFINED_BEYOND, LEVEL_NAMES_REFINED_V2_BEYOND,
                          LEVEL_NAMES_PER_TABLE, LEVEL_NAMES_PER_TABLE_BEYOND,
                          LEVEL_NAMES_NO_CONCEPT)
from agent import PPOAgent
from db.experiment_db import ExperimentDB
from db.schema import get_semantic_level


# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a .pt checkpoint file to resume from")
parser.add_argument("--level", type=int, default=0,
                    help="Curriculum level to start at (0-12). Ignored if checkpoint sets it.")
parser.add_argument("--steps", type=int, default=2_000_000,
                    help="Total training steps for this run")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (use different values for multi-seed runs)")
parser.add_argument("--n_envs", type=int, default=1,
                    help="Number of parallel environments. Default 1 = fully reproducible "
                         "(same seed → identical run on any machine). Use 4+ for speed "
                         "at the cost of exact cross-machine reproducibility.")
parser.add_argument("--entropy_mode", type=str, default="anneal",
                    choices=["anneal", "fixed", "reset"],
                    help="Entropy schedule: 'anneal' (default, decays 0.01→0.001 over full run), "
                         "'fixed' (constant 0.0055), "
                         "'reset' (resets to 0.01 on each level-up, then re-anneals)")
parser.add_argument("--run_tag", type=str, default="",
                    help="Optional tag appended to log/checkpoint filenames, e.g. 'entropy_fixed'")
parser.add_argument("--refined_mul", action="store_true", default=False,
                    help="Use refined 4-step multiplication curriculum (Mult 0-3/6/9/12) "
                         "instead of the original 2-step version. Adds 2 extra levels (15 total).")
parser.add_argument("--refined_div", action="store_true", default=False,
                    help="Use refined 4-step division curriculum (divisor 1-3/6/9/12). "
                         "Requires --refined_mul. Adds 2 more levels (17 total). "
                         "Eliminates the 5.76x scale discontinuity at the division transition.")
parser.add_argument("--relative_reward", action="store_true", default=False,
                    help="Use relative reward: normalize error by the level's answer_scale "
                         "(level-consistent) instead of per-problem max(|correct|, 1). "
                         "Also uses sharper gradient (3.0 vs 1.5). Hypothesis: tighter "
                         "signal near correct answer helps break mastery ceiling.")
parser.add_argument("--per_table_mul", action="store_true", default=False,
                    help="Use per-table multiplication curriculum (23L): teaches each times "
                         "table (×2–×12) as its own level before a generalisation level. "
                         "Mutually exclusive with --refined_mul / --refined_div.")
parser.add_argument("--beyond_energy", action="store_true", default=False,
                    help="Extend the curriculum with 3 additional physics levels after KE: "
                         "Potential Energy (PE=mgh), Work (W=Fd), Momentum (p=mv). "
                         "Adds 3 levels to any base curriculum (13L→16L, 15L→18L, etc.).")
parser.add_argument("--no_concept_discovery", action="store_true", default=False,
                    help="Use the 11L no-concept-discovery curriculum: skips the Mul(0-5) "
                         "and Div(1-5) warmup levels, going straight to full-scale Mul(0-12) "
                         "and Div(1-12). Tests Finding 3: is concept discovery necessary "
                         "when using relative reward? Incompatible with --refined_mul/div/per_table.")
parser.add_argument("--no_mastery_rescue", action="store_true", default=False,
                    help="Disable the within-level mastery regression rescue mechanism. "
                         "When enabled (default), reset-mode entropy resets if mastery "
                         "peaks then drops by >= 8pp while entropy is near the floor. "
                         "Use this flag to reproduce the 20-seed sweep conditions, which "
                         "predated the rescue mechanism.")
parser.add_argument("--mastery_mode", type=str, default="det",
                    choices=["det", "stoch"],
                    help="Mastery gating for level advancement. "
                         "'det' (default): gates on eval_deterministic_mastery() — noise-free "
                         "policy mean, proven to correctly reflect learning. "
                         "'stoch': gates on the rolling stochastic reward window (avg_reward) — "
                         "the naive approach that the stochastic false ceiling makes unreliable. "
                         "Use this to demonstrate Finding 1: with stochastic gating the agent "
                         "never advances past L0 even with relative reward + entropy reset.")
parser.add_argument("--device", type=str, default=None,
                    help="Device to train on: 'cpu', 'mps' (Apple Silicon), or 'cuda'. "
                         "Defaults to best available: cuda > mps > cpu.")
args = parser.parse_args()

# Active level-name list — mirrors the logic in MathEvolutionEnv.__init__
# (used for checkpoint resume messages before eval_env exists)
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

# ── Configuration ────────────────────────────────────────────────────────────

TOTAL_STEPS       = args.steps         # total environment interactions
ROLLOUT_SIZE      = 512         # steps collected per env before each update
LOG_INTERVAL      = 1_000       # log every N steps
SAVE_INTERVAL     = 50_000      # save checkpoint every N steps
MILESTONE_INTERVAL = 1_000_000  # always save at every 1M step boundary
# ── Storage: resolved via config.py (machine-aware) ──────────────────────────
# config.py reads config/machines.yaml and returns paths for this machine.
# Falls back to Dropbox glob if config.py is unavailable.
_tag = f"_{args.run_tag}" if args.run_tag else ""
try:
    from config import get_config as _get_config
    _cfg = _get_config()
    CHECKPOINT_DIR     = os.path.join(_cfg.checkpoint_path, f"checkpoints_seed{args.seed}{_tag}")
    LOG_FILE           = os.path.join(_cfg.data_path, f"training_log_seed{args.seed}{_tag}.csv")
    CHECKPOINT_MANIFEST = os.path.join(_cfg.data_path, f"checkpoints_manifest_seed{args.seed}{_tag}.csv")
    print(f"Config [{_cfg.name}] — saving to: {_cfg.data_path}")
except Exception as cfg_err:
    print(f"Config load failed ({cfg_err}), falling back to path detection")
    import glob as _glob
    _dropbox_roots = _glob.glob(
        os.path.expanduser("~/Library/CloudStorage/Dropbox*/ai_model/math-evolution-agent-phase-1")
    )
    if _dropbox_roots:
        CHECKPOINT_DIR     = os.path.join(_dropbox_roots[0], "checkpoints", f"checkpoints_seed{args.seed}{_tag}")
        LOG_FILE           = os.path.join(_dropbox_roots[0], "logs", f"training_log_seed{args.seed}{_tag}.csv")
        CHECKPOINT_MANIFEST = os.path.join(_dropbox_roots[0], "logs", f"checkpoints_manifest_seed{args.seed}{_tag}.csv")
        print(f"Config fallback (Dropbox) — saving to: {_dropbox_roots[0]}")
    else:
        CHECKPOINT_DIR     = f"logs/checkpoints_seed{args.seed}{_tag}"
        LOG_FILE           = f"logs/training_log_seed{args.seed}{_tag}.csv"
        CHECKPOINT_MANIFEST = f"logs/checkpoints_manifest_seed{args.seed}{_tag}.csv"
        print("Config fallback (local) — saving to logs/")

HIDDEN_DIM        = 256
LEARNING_RATE     = 3e-4
MASTERY_THRESHOLD = 0.85        # deterministic avg reward needed to advance level
MASTERY_WINDOW    = 200         # rolling window size for display mastery

# Deterministic mastery evaluation (the true learning signal)
MASTERY_CHECK_INTERVAL = 2_000  # check every N training steps
MASTERY_EPISODES       = 200    # episodes per deterministic check

# Entropy annealing: start with more exploration, taper to near-zero over the run
ENTROPY_START = 0.01
ENTROPY_END   = 0.001
# For 'reset' mode: how many steps to anneal over after each level-up.
# Set to 5M so harder levels (e.g. Division, Kinematics) retain meaningful
# exploration for longer before collapsing to the 0.001 floor.
ENTROPY_RESET_WINDOW = 5_000_000

# Mastery regression rescue (reset mode only):
# If det_mastery drops this far below the level's peak while entropy is near
# the floor, reset entropy to ENTROPY_START so the policy can re-explore.
MASTERY_RESCUE_DROP = 0.08   # trigger threshold (8 pp drop from peak)
MASTERY_RESCUE_MAX  = 3      # max rescues per level

# ─────────────────────────────────────────────────────────────────────────────


def make_env(start_level: int, worker_seed: int, cfg: EnvConfig):
    """Factory that returns a callable creating one MathEvolutionEnv instance.

    Each worker gets its own deterministic RNG stream (seed = global_seed * 1000 + worker_idx)
    so problem sequences are reproducible and independent across parallel envs.

    Accepts an EnvConfig so all curriculum flags travel as a single object —
    no six-argument pyramid, no risk of a flag being passed in the wrong position.
    """
    def _init():
        return MathEvolutionEnv.from_config(cfg, start_level=start_level, seed=worker_seed)
    return _init


def eval_deterministic_mastery(env: MathEvolutionEnv, agent: PPOAgent, n_episodes: int = MASTERY_EPISODES) -> float:
    """
    Measure the agent's TRUE mastery using deterministic (noise-free) actions.

    Why this matters:
      PPO requires stochastic actions during training — it samples from a
      Normal distribution centered on the agent's learned answer. That noise
      is essential for the policy gradient math, but it pollutes the mastery
      signal: a well-trained agent can score only ~0.65 stochastically while
      scoring ~0.85 deterministically. Using noisy actions for level-up checks
      caused the agent to get stuck even after genuinely mastering a level.

      This function bypasses noise entirely — the agent outputs its mean
      prediction (the actual answer it has learned), giving an honest measure
      of what was internalized, not what exploration noise produced.

    Uses a single (non-vectorized) eval_env to keep this clean and separate
    from the training environments.
    """
    saved_level = env.level
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        action = agent.select_action(state, deterministic=True)
        action_env = np.array([float(action)])
        _, reward, _, _, _ = env.step(action_env)
        rewards.append(reward)
    # Restore level in case eval_reset changed anything (it shouldn't)
    env.level = saved_level
    return float(np.mean(rewards))


def collect_rollout(envs: SyncVectorEnv, agent: PPOAgent, rollout_size: int, n_envs: int) -> tuple[dict, dict]:
    """
    Interact with n_envs parallel environments for rollout_size steps each.

    Each step, all n_envs environments receive actions simultaneously.
    Total experience per call: rollout_size × n_envs transitions.

    SyncVectorEnv auto-resets any terminated environments — ideal for our
    single-step math episodes where every problem terminates immediately.

    Returns flattened experience tensors of shape (rollout_size * n_envs, ...).
    """
    all_states, all_actions, all_rewards = [], [], []
    all_log_probs, all_values, all_dones = [], [], []

    obs, _ = envs.reset()  # obs shape: (n_envs, 5)

    for _ in range(rollout_size):
        obs_t = torch.FloatTensor(obs).to(agent.device)  # (n_envs, 5)

        with torch.no_grad():
            action, log_prob, _, value = agent.network.get_action_and_value(obs_t)
            # action:   (n_envs, 1)   — tanh-bounded answer per env
            # log_prob: (n_envs,)
            # value:    (n_envs,)

        action_np = action.cpu().numpy()  # (n_envs, 1) — what envs.step() expects

        next_obs, reward, terminated, truncated, info = envs.step(action_np)
        done = (terminated | truncated).astype(float)  # (n_envs,)

        all_states.append(obs)                          # (n_envs, 5)
        all_actions.append(action_np.squeeze(-1))       # (n_envs,)
        all_rewards.append(reward)                      # (n_envs,)
        all_log_probs.append(log_prob.cpu().numpy())    # (n_envs,)
        all_values.append(value.cpu().numpy())          # (n_envs,)
        all_dones.append(done)                          # (n_envs,)

        obs = next_obs  # auto-reset handled by SyncVectorEnv

    # Flatten (rollout_size, n_envs, ...) → (rollout_size * n_envs, ...)
    # Note: episodes are always 1-step, so cross-env return propagation is negligible.
    return {
        "states":    np.array(all_states,    dtype=np.float32).reshape(-1, 5),
        "actions":   np.array(all_actions,   dtype=np.float32).reshape(-1),
        "rewards":   np.array(all_rewards,   dtype=np.float32).reshape(-1),
        "log_probs": np.array(all_log_probs, dtype=np.float32).reshape(-1),
        "values":    np.array(all_values,    dtype=np.float32).reshape(-1),
        "dones":     np.array(all_dones,     dtype=np.float32).reshape(-1),
    }, info


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize environment and agent
    start_level = args.level

    # If resuming from checkpoint, detect which level the agent was on
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            meta_path = args.checkpoint.replace(".pt", "_meta.pt")
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location="cpu")
                start_level = meta.get("level", args.level)
                print(f"Resuming from checkpoint: {args.checkpoint}")
                print(f"Restoring to level {start_level}: {level_names[start_level]}")
            else:
                print(f"Resuming from checkpoint: {args.checkpoint}")
        else:
            print(f"WARNING: Checkpoint not found at {args.checkpoint}. Starting fresh.")
            args.checkpoint = None

    # Set random seeds for reproducibility — log the seed so results can be attributed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Random seed: {args.seed}")

    # ── Build EnvConfig from CLI args — one object for all env creation ──────
    env_cfg = EnvConfig(
        refined_mul=args.refined_mul,
        refined_div=args.refined_div,
        relative_reward=args.relative_reward,
        per_table_mul=args.per_table_mul,
        beyond_energy=args.beyond_energy,
        no_concept_discovery=args.no_concept_discovery,
        mastery_threshold=MASTERY_THRESHOLD,
        mastery_window=MASTERY_WINDOW,
    )

    # ── Vectorized training environments ─────────────────────────────────────
    # n_envs parallel environments run simultaneously, giving a batch of
    # n_envs × ROLLOUT_SIZE transitions per PPO update. More parallel envs
    # means larger, more stable gradient estimates without extra wall-clock time.
    envs = SyncVectorEnv([
        make_env(start_level, worker_seed=args.seed * 1000 + i, cfg=env_cfg)
        for i in range(args.n_envs)
    ])

    # ── Separate evaluation environment ──────────────────────────────────────
    # Kept single (non-vectorized) so eval_deterministic_mastery() is simple.
    # This is the source of truth for level tracking and level-up decisions.
    eval_env = MathEvolutionEnv.from_config(env_cfg, start_level=start_level, seed=args.seed)

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    agent = PPOAgent(
        state_dim=5,
        hidden_dim=HIDDEN_DIM,
        lr=LEARNING_RATE,
        device=device,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)

    # Logging
    reward_window = deque(maxlen=100)
    total_steps = 0
    update_count = 0
    start_time = time.time()
    det_mastery = 0.0          # last deterministic mastery score
    last_mastery_check = -MASTERY_CHECK_INTERVAL  # trigger check immediately
    last_entropy_reset_step = 0  # tracks when entropy was last reset (for 'reset' mode)
    level_peak_mastery = 0.0     # best det_mastery seen at current level (for rescue)
    entropy_rescue_count = 0     # rescues triggered at current level
    proc = psutil.Process()    # handle to this process for CPU% sampling
    proc.cpu_percent()         # first call primes the counter (returns 0.0, discard)

    # ── Database setup ────────────────────────────────────────────────────────
    db = ExperimentDB()
    _curriculum = "13L"
    if args.no_concept_discovery:   _curriculum = "11L-ncd"
    elif args.per_table_mul and args.beyond_energy: _curriculum = "26L"
    elif args.per_table_mul:        _curriculum = "23L"
    elif args.refined_mul and args.refined_div and args.beyond_energy: _curriculum = "20L"
    elif args.refined_mul and args.refined_div: _curriculum = "17L"
    elif args.refined_mul and args.beyond_energy: _curriculum = "18L"
    elif args.refined_mul:          _curriculum = "15L"
    elif args.beyond_energy:        _curriculum = "16L"
    _run_id = (f"seed{args.seed}"
               f"_{'rel' if args.relative_reward else 'abs'}"
               f"_{args.entropy_mode}"
               f"_{_curriculum}"
               f"_{time.strftime('%Y-%m-%d')}"
               + (f"_{args.run_tag}" if args.run_tag else ""))
    # Capture the full set of resolved training parameters so this experiment
    # is self-describing in the DB — no inference from filename needed later.
    _params_json = {
        "seed":                args.seed,
        "steps":               TOTAL_STEPS,
        "n_envs":              args.n_envs,
        "starting_level":      start_level,
        "curriculum":          _curriculum,
        "reward_mode":         "relative" if args.relative_reward else "absolute",
        "entropy_mode":        args.entropy_mode,
        "mastery_mode":        args.mastery_mode,
        "hidden_dim":          HIDDEN_DIM,
        "learning_rate":       LEARNING_RATE,
        "rollout_size":        ROLLOUT_SIZE,
        "mastery_threshold":   MASTERY_THRESHOLD,
        "relative_reward":     args.relative_reward,
        "refined_mul":         args.refined_mul,
        "refined_div":         args.refined_div,
        "beyond_energy":       args.beyond_energy,
        "no_concept_discovery": args.no_concept_discovery,
        "per_table_mul":       args.per_table_mul,
        "run_tag":             args.run_tag or None,
        "device":              device,
    }
    _exp_id = db.create_experiment(
        run_id=_run_id,
        seed=args.seed,
        n_envs=args.n_envs,
        total_steps=TOTAL_STEPS,
        curriculum=_curriculum,
        reward_mode="relative" if args.relative_reward else "absolute",
        entropy_mode=args.entropy_mode,
        device=device,
        params_json=_params_json,
    )
    print(f"  DB experiment #{_exp_id}: {_run_id}")
    # Level-up tracking for db: capture state BEFORE advance
    _level_start_step = 0       # step when current level began
    _level_start_elapsed = 0.0

    # Batch size: transitions collected per PPO update
    batch_size = ROLLOUT_SIZE * args.n_envs

    os.makedirs(os.path.dirname(os.path.abspath(LOG_FILE)), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write("step,level,level_name,avg_reward,det_mastery,stoch_mastery,entropy_coef,loss,elapsed_min,cpu_pct,wall_time,seed\n")

    # Checkpoint manifest: flat CSV that records every .pt file saved.
    # Synced to Mac via sync_workers.sh → imported into checkpoints table.
    # Append mode so resume runs accumulate all checkpoint records.
    _manifest_is_new = not os.path.exists(CHECKPOINT_MANIFEST)
    if _manifest_is_new:
        with open(CHECKPOINT_MANIFEST, "w") as _mf:
            _mf.write("step,level,level_name,checkpoint_path,meta_path,checkpoint_type,det_mastery,wall_time\n")

    def _log_checkpoint_manifest(step: int, level: int, level_name: str, checkpoint_path: str, meta_path: str | None, checkpoint_type: str, det_mastery: float | None) -> None:
        wall_time = time.strftime("%Y-%m-%d %H:%M:%S")
        mastery_str = f"{det_mastery:.4f}" if det_mastery is not None else ""
        with open(CHECKPOINT_MANIFEST, "a") as _mf:
            _mf.write(f"{step},{level},{level_name},{checkpoint_path},{meta_path or ''},{checkpoint_type},{mastery_str},{wall_time}\n")

    print("=" * 60)
    print("  MATH EVOLUTION AGENT — TRAINING START")
    print("=" * 60)
    print(f"  Starting at Level {start_level}: {eval_env.level_names[start_level]}")
    print(f"  Target steps: {TOTAL_STEPS:,}")
    print(f"  Parallel envs: {args.n_envs}  (batch size: {batch_size:,} transitions/update)")
    print(f"  Device: {agent.device}")
    print(f"  Seed: {args.seed}")
    print(f"  Entropy mode: {args.entropy_mode}")
    if args.mastery_mode == "stoch":
        print(f"  Mastery gate: STOCHASTIC (Finding 1 control — expect agent to stall at L0)")
    if args.run_tag:
        print(f"  Run tag: {args.run_tag}")
    if args.relative_reward:
        print(f"  Reward: RELATIVE (level-scale normalized, sharpness=3.0)")
    n_levels = len(eval_env.level_names)
    if args.no_concept_discovery:
        print(f"  Curriculum: NO-CONCEPT-DISCOVERY ({n_levels}L — skips Mul 0-5 & Div 1-5 warmups)")
        print(f"  Finding 3 test: concept discovery necessity with relative reward")
    elif args.per_table_mul and args.beyond_energy:
        print(f"  Curriculum: PER-TABLE + BEYOND-ENERGY ({n_levels}L — ×2–×12 tables + PE/Work/Momentum)")
    elif args.per_table_mul:
        print(f"  Curriculum: PER-TABLE ({n_levels}L — ×2–×12 individual times-table levels)")
    elif args.refined_mul and args.refined_div and args.beyond_energy:
        print(f"  Curriculum: FULLY REFINED + BEYOND-ENERGY ({n_levels}L — 4 mul + 4 div + PE/Work/Momentum)")
    elif args.refined_mul and args.refined_div:
        print(f"  Curriculum: FULLY REFINED ({n_levels}L — 4 mult + 4 div sub-levels)")
    elif args.refined_mul and args.beyond_energy:
        print(f"  Curriculum: REFINED + BEYOND-ENERGY ({n_levels}L — 4 mul sub-levels + PE/Work/Momentum)")
    elif args.refined_mul:
        print(f"  Curriculum: REFINED ({n_levels}L — 4 mult sub-levels)")
    elif args.beyond_energy:
        print(f"  Curriculum: STANDARD + BEYOND-ENERGY ({n_levels}L — adds PE, Work, Momentum after KE)")
    else:
        print(f"  Curriculum: STANDARD ({n_levels}L)")
    print("=" * 60)

    while total_steps < TOTAL_STEPS:
        # Collect experience from all parallel environments
        rollout, last_info = collect_rollout(envs, agent, ROLLOUT_SIZE, args.n_envs)
        total_steps += batch_size  # count actual environment interactions

        # Entropy schedule — controlled by --entropy_mode:
        #  'anneal': classic linear decay over the full run
        #  'fixed':  constant midpoint — no collapse, no exploration boost
        #  'reset':  resets to ENTROPY_START on each level-up, then re-anneals
        #            over ENTROPY_RESET_WINDOW steps — keeps exploration alive
        #            at the start of each new curriculum level
        if args.entropy_mode == "anneal":
            frac = min(total_steps / TOTAL_STEPS, 1.0)
            agent.entropy_coef = ENTROPY_START + (ENTROPY_END - ENTROPY_START) * frac
        elif args.entropy_mode == "fixed":
            agent.entropy_coef = (ENTROPY_START + ENTROPY_END) / 2  # ~0.0055 constant
        elif args.entropy_mode == "reset":
            steps_in_level = total_steps - last_entropy_reset_step
            frac = min(steps_in_level / ENTROPY_RESET_WINDOW, 1.0)
            agent.entropy_coef = ENTROPY_START + (ENTROPY_END - ENTROPY_START) * frac

        # Update agent
        loss = agent.update(rollout)
        update_count += 1

        # ── Deterministic mastery check ───────────────────────────────────────
        # Run noise-free evaluation on the dedicated eval_env to get the agent's
        # true learned score. Uses eval_env (single env) so there's no confusion
        # about which environment's state we're reading.
        if total_steps - last_mastery_check >= MASTERY_CHECK_INTERVAL:
            last_mastery_check = total_steps
            det_mastery = eval_deterministic_mastery(eval_env, agent)

            # ── Mastery regression rescue (reset mode only) ───────────────────
            # If mastery peaks then falls significantly while entropy is near the
            # floor, reset entropy so the policy can re-explore rather than
            # collapsing into a bad local optimum.
            if args.entropy_mode == "reset" and not args.no_mastery_rescue:
                if det_mastery > level_peak_mastery:
                    level_peak_mastery = det_mastery
                elif (level_peak_mastery > 0.5 and
                      det_mastery < level_peak_mastery - MASTERY_RESCUE_DROP and
                      agent.entropy_coef < 0.003 and
                      entropy_rescue_count < MASTERY_RESCUE_MAX):
                    entropy_rescue_count += 1
                    prev_peak = level_peak_mastery
                    level_peak_mastery = det_mastery  # reset peak so rescue doesn't re-fire immediately
                    agent.entropy_coef = ENTROPY_START
                    last_entropy_reset_step = total_steps
                    print(f"  [ENTROPY RESCUE #{entropy_rescue_count}] "
                          f"mastery regressed {prev_peak:.3f}→{det_mastery:.3f}, "
                          f"entropy reset to {ENTROPY_START}")

            # Mastery gating: 'det' uses noise-free policy mean (correct approach);
            # 'stoch' uses the rolling stochastic window (demonstrates Finding 1 —
            # stochastic gating never crosses 0.85, so the agent never advances).
            if args.mastery_mode == "stoch":
                stoch_scores = [e.mastery_score() for e in envs.envs if e.recent_rewards]
                mastery_gate = float(np.mean(stoch_scores)) if stoch_scores else 0.0
            else:
                mastery_gate = det_mastery

            if mastery_gate >= MASTERY_THRESHOLD and eval_env.level < len(eval_env.level_names) - 1:
                # ── FIX: capture pre-advance state BEFORE calling advance_level() ──
                # Log entry at this step must show the OLD level with the triggering
                # det_mastery (≥ 0.85). This corrects the original logging bug where
                # the first post-advance entry showed new level + high det_mastery.
                _pre_advance_level = eval_env.level
                _pre_advance_lname = eval_env.level_names[_pre_advance_level]
                _elapsed_at_advance = (time.time() - start_time) / 60

                eval_env.advance_level()

                # Propagate new level to all training environments
                new_level = eval_env.level
                for sub_env in envs.envs:
                    sub_env.level = new_level
                    sub_env.recent_rewards.clear()  # reset stochastic window per env

                # 'reset' mode: restart entropy so the agent explores the new level freely
                if args.entropy_mode == "reset":
                    agent.entropy_coef = ENTROPY_START
                    last_entropy_reset_step = total_steps
                    level_peak_mastery = 0.0
                    entropy_rescue_count = 0

                # ── Level-up checkpoint ────────────────────────────────────────
                # Save at the EXACT step of every level-up (in addition to periodic saves).
                levelup_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"agent_levelup_{_pre_advance_level}to{new_level}_step{total_steps}.pt"
                )
                agent.save(levelup_path)
                _levelup_meta = {
                    "level": new_level,          # level AFTER advance
                    "prev_level": _pre_advance_level,
                    "steps": total_steps,
                    "det_mastery": det_mastery,  # the score that triggered it
                }
                torch.save(_levelup_meta, levelup_path.replace(".pt", "_meta.pt"))
                print(f"  >> Level-up checkpoint: L{_pre_advance_level}→L{new_level} at step {total_steps:,} (det={det_mastery:.3f})")

                # ── DB: log the level transition (with corrected context) ──────
                _stoch_at_advance = float(np.mean([e.mastery_score() for e in envs.envs]))
                db.log_level_transition(
                    experiment_id=_exp_id,
                    step=total_steps,
                    from_level=_pre_advance_level,
                    to_level=new_level,
                    from_level_name=_pre_advance_lname,
                    to_level_name=eval_env.level_names[new_level],
                    det_mastery=det_mastery,
                    stoch_mastery=_stoch_at_advance,
                    steps_on_level=total_steps - _level_start_step,
                    elapsed_min=_elapsed_at_advance,
                    checkpoint_path=levelup_path,
                )
                db.log_checkpoint(
                    experiment_id=_exp_id,
                    step=total_steps,
                    level=new_level,
                    checkpoint_path=levelup_path,
                    meta_path=levelup_path.replace(".pt", "_meta.pt"),
                    checkpoint_type="levelup",
                    det_mastery=det_mastery,
                )
                _log_checkpoint_manifest(total_steps, new_level, eval_env.level_names[new_level],
                                         levelup_path, levelup_path.replace(".pt", "_meta.pt"),
                                         "levelup", det_mastery)
                _level_start_step = total_steps
                _level_start_elapsed = _elapsed_at_advance

        # Track rewards — average across all parallel environments
        avg_reward = float(np.mean(rollout["rewards"]))
        reward_window.append(avg_reward)
        smoothed_reward = float(np.mean(reward_window))

        # Use eval_env as level source of truth; average mastery across all sub-envs
        level = eval_env.level
        mastery = float(np.mean([e.mastery_score() for e in envs.envs]))

        # Logging
        if total_steps % LOG_INTERVAL < batch_size:
            elapsed = (time.time() - start_time) / 60
            cpu_pct = proc.cpu_percent()               # % since last call
            wall_time = time.strftime("%Y-%m-%d %H:%M:%S")
            lname = eval_env.level_names[level]
            print(
                f"Step {total_steps:>8,} | "
                f"Level {level} ({lname[:18]:<18}) | "
                f"Reward {smoothed_reward:.3f} | "
                f"Mastery(det) {det_mastery:.3f} | "
                f"Loss {loss:.4f} | "
                f"Entropy {agent.entropy_coef:.4f} | "
                f"CPU {cpu_pct:.0f}% | "
                f"{elapsed:.1f}min | "
                f"{wall_time}"
            )
            with open(LOG_FILE, "a") as f:
                f.write(f"{total_steps},{level},{lname},"
                        f"{smoothed_reward:.4f},{det_mastery:.4f},{mastery:.4f},"
                        f"{agent.entropy_coef:.5f},{loss:.4f},{elapsed:.2f},"
                        f"{cpu_pct:.1f},{wall_time},{args.seed}\n")
            # DB log
            db.log_step(
                experiment_id=_exp_id,
                step=total_steps,
                level=level,
                level_name=lname,
                avg_reward=smoothed_reward,
                det_mastery=det_mastery,
                stoch_mastery=mastery,
                entropy_coef=agent.entropy_coef,
                loss=loss,
                elapsed_min=elapsed,
                cpu_pct=cpu_pct,
                wall_time=wall_time,
            )

        # Save checkpoint + metadata (every SAVE_INTERVAL steps)
        if total_steps % SAVE_INTERVAL < batch_size:
            path = os.path.join(CHECKPOINT_DIR, f"agent_step_{total_steps}.pt")
            agent.save(path)
            _meta = {"level": eval_env.level, "steps": total_steps,
                     "seed": args.seed, "curriculum": _curriculum,
                     "reward_mode": "relative" if args.relative_reward else "absolute",
                     "entropy_mode": args.entropy_mode, "n_envs": args.n_envs}
            torch.save(_meta, path.replace(".pt", "_meta.pt"))
            db.log_checkpoint(
                experiment_id=_exp_id, step=total_steps, level=eval_env.level,
                checkpoint_path=path, meta_path=path.replace(".pt", "_meta.pt"),
                checkpoint_type="periodic", det_mastery=det_mastery,
            )
            _log_checkpoint_manifest(total_steps, eval_env.level, eval_env.level_names[eval_env.level],
                                     path, path.replace(".pt", "_meta.pt"),
                                     "periodic", det_mastery)

        # Milestone checkpoint at every 1M step boundary (for easy resume points)
        prev_steps = total_steps - batch_size
        if total_steps // MILESTONE_INTERVAL > prev_steps // MILESTONE_INTERVAL:
            milestone = (total_steps // MILESTONE_INTERVAL) * MILESTONE_INTERVAL
            mpath = os.path.join(CHECKPOINT_DIR, f"agent_{milestone // 1_000_000}M.pt")
            agent.save(mpath)
            _meta = {"level": eval_env.level, "steps": milestone,
                     "seed": args.seed, "curriculum": _curriculum,
                     "reward_mode": "relative" if args.relative_reward else "absolute",
                     "entropy_mode": args.entropy_mode, "n_envs": args.n_envs}
            torch.save(_meta, mpath.replace(".pt", "_meta.pt"))
            db.log_checkpoint(
                experiment_id=_exp_id, step=milestone, level=eval_env.level,
                checkpoint_path=mpath, meta_path=mpath.replace(".pt", "_meta.pt"),
                checkpoint_type="milestone", det_mastery=det_mastery,
            )
            _log_checkpoint_manifest(milestone, eval_env.level, eval_env.level_names[eval_env.level],
                                     mpath, mpath.replace(".pt", "_meta.pt"),
                                     "milestone", det_mastery)
            print(f"  >> Milestone: saved {milestone // 1_000_000}M checkpoint (Level {eval_env.level}: {eval_env.level_names[eval_env.level]})")

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "agent_final.pt")
    agent.save(final_path)
    _meta = {"level": eval_env.level, "steps": TOTAL_STEPS,
             "seed": args.seed, "curriculum": _curriculum,
             "reward_mode": "relative" if args.relative_reward else "absolute",
             "entropy_mode": args.entropy_mode, "n_envs": args.n_envs}
    torch.save(_meta, final_path.replace(".pt", "_meta.pt"))
    db.log_checkpoint(
        experiment_id=_exp_id, step=TOTAL_STEPS, level=eval_env.level,
        checkpoint_path=final_path, meta_path=final_path.replace(".pt", "_meta.pt"),
        checkpoint_type="final", det_mastery=det_mastery,
    )
    _log_checkpoint_manifest(TOTAL_STEPS, eval_env.level, eval_env.level_names[eval_env.level],
                             final_path, final_path.replace(".pt", "_meta.pt"),
                             "final", det_mastery)
    db.complete_experiment(_exp_id,
                           final_level=eval_env.level,
                           final_level_name=eval_env.level_names[eval_env.level])
    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining complete in {elapsed:.1f} minutes.")
    print(f"Final level: {eval_env.level} — {eval_env.level_names[eval_env.level]}")
    print(f"DB experiment #{_exp_id} complete: {_run_id}")


if __name__ == "__main__":
    main()
