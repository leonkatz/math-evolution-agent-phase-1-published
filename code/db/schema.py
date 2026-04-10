"""
Database schema for the Math Evolution Agent experiment database.

All experiment data — training logs, level transitions, checkpoints —
is stored in a single SQLite database for easy querying and analysis.
"""

SCHEMA_SQL = """
-- ─────────────────────────────────────────────────────────────────────────────
-- SEMANTIC LEVEL ENCODING
-- sem_level / final_sem_level columns hold a float that encodes the
-- mathematical domain and sub-difficulty of a curriculum level.
-- This allows cross-curriculum comparison: "Multiplication (0-12)" is always
-- 3.4, whether it is level 5 in the 13L curriculum or level 7 in the 17L one.
--
-- Domain ranges:
--   1.x  Addition      2.x  Subtraction     3.x  Multiplication
--   4.x  Division      5.0  Mixed            6.0  Linear Algebra
--   7.0  Quadratic      8.0  Kinematics       9.x  Energy / PE / Work
--  10.0  Momentum      -1.0  Unknown (level name not in SEMANTIC_LEVELS dict)
--
-- Full mapping: see SEMANTIC_LEVELS dict in schema.py.
-- ─────────────────────────────────────────────────────────────────────────────

-- ─────────────────────────────────────────────────────────────────────────────
-- One row per training run (fully describes the experiment config)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT UNIQUE NOT NULL,   -- human-readable: "seed42_rel_reset_13L_2026-03-15"
    seed            INTEGER NOT NULL,
    n_envs          INTEGER NOT NULL DEFAULT 1,
    total_steps     INTEGER NOT NULL,
    curriculum      TEXT NOT NULL,          -- "13L" | "15L" | "17L" | "11L-ncd" | "23L" | "26L"
    reward_mode     TEXT NOT NULL,          -- "absolute" | "relative"
    entropy_mode    TEXT NOT NULL,          -- "anneal" (linear decay) | "fixed" (constant) | "reset" (resets on level-up)
    device          TEXT,                   -- "cpu" | "mps" (Apple Silicon) | "cuda" (NVIDIA)
    hostname        TEXT,                   -- machine that ran the experiment
    hidden_dim      INTEGER DEFAULT 256,    -- actor-critic network width
    learning_rate   REAL DEFAULT 0.0003,
    rollout_size    INTEGER DEFAULT 512,    -- PPO steps per update
    mastery_threshold REAL DEFAULT 0.85,   -- deterministic mastery score required to advance a level
    started_at      TEXT,                   -- ISO 8601 UTC timestamp
    completed_at    TEXT,                   -- NULL if run was interrupted
    final_level     INTEGER,               -- curriculum level index at end of run (0-based)
    final_level_name TEXT,                 -- human-readable name of final level
    final_sem_level REAL,                  -- semantic level float at end of run (see SEMANTIC LEVEL ENCODING above)
    notes           TEXT,                  -- free-form notes
    imported_from   TEXT                   -- NULL if live-logged; CSV path if imported from historical data
);

-- ─────────────────────────────────────────────────────────────────────────────
-- One row per log interval (every LOG_INTERVAL steps during training)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    step            INTEGER NOT NULL,
    level           INTEGER NOT NULL,
    level_name      TEXT NOT NULL,
    sem_level       REAL,                   -- semantic level float (see SEMANTIC LEVEL ENCODING above)
    avg_reward      REAL,                   -- smoothed stochastic reward (rolling 100-episode average)
    det_mastery     REAL,                   -- deterministic mastery: 200-episode avg with noise-free actions
    stoch_mastery   REAL,                   -- stochastic mastery: rolling avg of noisy training rewards
    entropy_coef    REAL,                   -- PPO entropy regularization coefficient at this step
    loss            REAL,                   -- PPO total loss (policy + value + entropy terms)
    elapsed_min     REAL,                   -- wall-clock minutes elapsed since training start
    cpu_pct         REAL,                   -- CPU utilization percentage at log time
    wall_time       TEXT                    -- ISO 8601 UTC timestamp when logged; for CSV imports
                                            -- this may hold elapsed_min as a string if the original
                                            -- CSV pre-dates the wall_time column (check elapsed_min instead)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- One row per level-up event (the most important events in a run)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS level_transitions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id       INTEGER NOT NULL REFERENCES experiments(id),
    step                INTEGER NOT NULL,
    from_level          INTEGER NOT NULL,
    to_level            INTEGER NOT NULL,
    from_level_name     TEXT NOT NULL,
    to_level_name       TEXT NOT NULL,
    from_sem_level      REAL,               -- semantic level of the level being left
    to_sem_level        REAL,               -- semantic level of the level being entered
    det_mastery         REAL NOT NULL,      -- deterministic mastery score that triggered the advance (≥ mastery_threshold)
    stoch_mastery       REAL,               -- stochastic rolling avg at the same moment (usually < threshold)
    steps_on_level      INTEGER,            -- environment steps spent on from_level before mastering it
    elapsed_min         REAL,               -- wall-clock minutes at time of transition
    checkpoint_path     TEXT                -- path to the level-up checkpoint file (if saved)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- One row per saved checkpoint file
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS checkpoints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id),
    step            INTEGER NOT NULL,
    level           INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL,
    meta_path       TEXT,
    checkpoint_type TEXT NOT NULL,          -- "periodic" | "milestone" | "levelup" | "final"
    det_mastery     REAL                    -- det mastery at time of save (NULL if not checked)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Indexes for common queries
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_training_log_experiment ON training_log(experiment_id);
CREATE INDEX IF NOT EXISTS idx_training_log_step ON training_log(experiment_id, step);
CREATE INDEX IF NOT EXISTS idx_transitions_experiment ON level_transitions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment ON checkpoints(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiments_seed ON experiments(seed);
CREATE INDEX IF NOT EXISTS idx_experiments_reward_mode ON experiments(reward_mode);
"""

# Semantic level mapping — stable across all curriculum variants
SEMANTIC_LEVELS = {
    "Addition (0-10)":                  1.1,
    "Addition (0-100)":                 1.2,
    "Subtraction (0-10)":               2.1,
    "Subtraction (0-100)":              2.2,
    "Multiplication (0-3)":             3.0,
    "Multiplication (0-5)":             3.1,
    "Multiplication (0-6)":             3.2,
    "Multiplication (0-9)":             3.3,
    "Multiplication (0-12)":            3.4,
    "Division (divisor 1-3)":           4.0,
    "Division (divisor 1-5)":           4.1,
    "Division (divisor 1-6)":           4.2,
    "Division (divisor 1-9)":           4.3,
    "Division (divisor 1-12)":          4.4,
    "Mixed Arithmetic":                 5.0,
    "Linear Algebra (ax + b = c)":      6.0,
    "Quadratic (ax^2 + bx + c = 0)":   7.0,
    "Kinematics (d=vt, F=ma)":          8.0,
    "Energy (KE, Work)":                9.0,
    "Potential Energy (PE=mgh)":        9.1,
    "Work (W=Fd)":                      9.2,
    "Momentum (p=mv)":                  10.0,
    # Per-table multiplication levels
    "Multiplication ×2":                3.02,
    "Multiplication ×3":                3.03,
    "Multiplication ×4":                3.04,
    "Multiplication ×5":                3.05,
    "Multiplication ×6":                3.06,
    "Multiplication ×7":                3.07,
    "Multiplication ×8":                3.08,
    "Multiplication ×9":                3.09,
    "Multiplication ×10":               3.10,
    "Multiplication ×11":               3.11,
    "Multiplication ×12":               3.12,
    "Multiplication (all tables)":      3.13,
}


def get_semantic_level(level_name: str) -> float:
    """Return the semantic level for a given level name. Returns -1.0 if unknown."""
    return SEMANTIC_LEVELS.get(level_name, -1.0)
