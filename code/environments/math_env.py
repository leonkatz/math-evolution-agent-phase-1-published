"""
Math Evolution Environment

The agent receives pure numerical input. It has no language, no human intuition.
It discovers mathematical relationships purely through reward signals.

Curriculum design principle:
  Every new operation is introduced first at small scale (concept discovery),
  then at large scale (generalization). This mirrors the structure used for addition
  and has been extended consistently to subtraction, multiplication, and division.

Curriculum levels:
  0  - Addition (0-10)           concept discovery
  1  - Addition (0-100)          scale up
  2  - Subtraction (0-10)        concept discovery
  3  - Subtraction (0-100)       scale up
  4  - Multiplication (0-5)      concept discovery (times tables up to 5×5)
  5  - Multiplication (0-12)     scale up (full times tables)
  6  - Division (divisor 1-5)    concept discovery
  7  - Division (divisor 1-12)   scale up
  8  - Mixed arithmetic          all four operations at full scale
  9  - Algebra: solve for x (linear)
  10 - Quadratic equations
  11 - Physics: kinematics (d = v*t, F = m*a)
  12 - Physics: energy (KE = 0.5*m*v^2)
"""

import numpy as np
import gymnasium as gym
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from gymnasium import spaces


@dataclass
class EnvConfig:
    """
    All curriculum flags in one place.

    Pass one EnvConfig instead of six boolean kwargs whenever you create a
    MathEvolutionEnv — eliminates the flag-duplication between train.py and
    the env factory, and makes configs easy to log, save, and compare.

    Usage:
        cfg = EnvConfig(refined_mul=True, refined_div=True, relative_reward=True)
        env = MathEvolutionEnv.from_config(cfg, start_level=0, seed=42)
    """
    refined_mul:          bool  = False
    refined_div:          bool  = False
    relative_reward:      bool  = False
    per_table_mul:        bool  = False
    beyond_energy:        bool  = False
    no_concept_discovery: bool  = False
    mastery_threshold:    float = 0.70
    mastery_window:       int   = 100


LEVEL_NAMES = [
    "Addition (0-10)",
    "Addition (0-100)",
    "Subtraction (0-10)",
    "Subtraction (0-100)",
    "Multiplication (0-5)",
    "Multiplication (0-12)",
    "Division (divisor 1-5)",
    "Division (divisor 1-12)",
    "Mixed Arithmetic",
    "Linear Algebra (ax + b = c)",
    "Quadratic (ax^2 + bx + c = 0)",
    "Kinematics (d=vt, F=ma)",
    "Energy (KE, Work)",
]

# Refined curriculum: 4 multiplication sub-levels instead of 2.
# Each step is a ~4x scale increase rather than one 5.76x jump.
# Input numbers (n1, n2) are always normalized by 12 (global max multiplier)
# so the agent sees consistent representations across all mult levels.
LEVEL_NAMES_REFINED = [
    "Addition (0-10)",              # L0  — unchanged
    "Addition (0-100)",             # L1  — unchanged
    "Subtraction (0-10)",           # L2  — unchanged
    "Subtraction (0-100)",          # L3  — unchanged
    "Multiplication (0-3)",         # L4  — NEW: 3-table max, product ≤ 9
    "Multiplication (0-6)",         # L5  — NEW: 6-table max, product ≤ 36
    "Multiplication (0-9)",         # L6  — NEW: 9-table max, product ≤ 81
    "Multiplication (0-12)",        # L7  — full times tables (was L5)
    "Division (divisor 1-5)",       # L8  — was L6
    "Division (divisor 1-12)",      # L9  — was L7
    "Mixed Arithmetic",             # L10 — was L8
    "Linear Algebra (ax + b = c)",  # L11 — was L9
    "Quadratic (ax^2 + bx + c = 0)",# L12 — was L10
    "Kinematics (d=vt, F=ma)",      # L13 — was L11
    "Energy (KE, Work)",            # L14 — was L12
]

# Full refined curriculum: 4 multiplication sub-levels + 4 division sub-levels.
# Division inputs are always normalized by 144 (global max dividend = 12×12)
# so the agent sees consistent representations across all division levels.
# This eliminates the 5.76x scale discontinuity that caused the division wall.
LEVEL_NAMES_REFINED_V2 = [
    "Addition (0-10)",              # L0
    "Addition (0-100)",             # L1
    "Subtraction (0-10)",           # L2
    "Subtraction (0-100)",          # L3
    "Multiplication (0-3)",         # L4  — refined mul sub-level 1
    "Multiplication (0-6)",         # L5  — refined mul sub-level 2
    "Multiplication (0-9)",         # L6  — refined mul sub-level 3
    "Multiplication (0-12)",        # L7  — refined mul sub-level 4 (full)
    "Division (divisor 1-3)",       # L8  — NEW: divisor ≤ 3, dividend ≤ 9
    "Division (divisor 1-6)",       # L9  — NEW: divisor ≤ 6, dividend ≤ 36
    "Division (divisor 1-9)",       # L10 — NEW: divisor ≤ 9, dividend ≤ 81
    "Division (divisor 1-12)",      # L11 — NEW: full division (divisor ≤ 12)
    "Mixed Arithmetic",             # L12
    "Linear Algebra (ax + b = c)",  # L13
    "Quadratic (ax^2 + bx + c = 0)",# L14
    "Kinematics (d=vt, F=ma)",      # L15
    "Energy (KE, Work)",            # L16
]

# ── Beyond-energy extension (appended to any base curriculum) ─────────────────
_BEYOND_ENERGY_SUFFIX = [
    "Potential Energy (PE=mgh)",   # 9.1
    "Work (W=Fd)",                 # 9.2
    "Momentum (p=mv)",             # 10.0
]
LEVEL_NAMES_BEYOND_ENERGY   = LEVEL_NAMES          + _BEYOND_ENERGY_SUFFIX   # 16L
LEVEL_NAMES_REFINED_BEYOND  = LEVEL_NAMES_REFINED  + _BEYOND_ENERGY_SUFFIX   # 18L
LEVEL_NAMES_REFINED_V2_BEYOND = LEVEL_NAMES_REFINED_V2 + _BEYOND_ENERGY_SUFFIX  # 20L

# ── Per-table multiplication curriculum (23L) ──────────────────────────────────
# Teaches each times table as its own level before generalising.
# Levels 4–14: multiplier fixed at (level - 2), other factor 1–12.
# Level 15: both factors free (generalization check).
LEVEL_NAMES_PER_TABLE = [
    "Addition (0-10)",              # L0   → 1.1
    "Addition (0-100)",             # L1   → 1.2
    "Subtraction (0-10)",           # L2   → 2.1
    "Subtraction (0-100)",          # L3   → 2.2
    "Multiplication ×2",            # L4   → 3.02
    "Multiplication ×3",            # L5   → 3.03
    "Multiplication ×4",            # L6   → 3.04
    "Multiplication ×5",            # L7   → 3.05
    "Multiplication ×6",            # L8   → 3.06
    "Multiplication ×7",            # L9   → 3.07
    "Multiplication ×8",            # L10  → 3.08
    "Multiplication ×9",            # L11  → 3.09
    "Multiplication ×10",           # L12  → 3.91
    "Multiplication ×11",           # L13  → 3.92
    "Multiplication ×12",           # L14  → 3.93
    "Multiplication (all tables)",  # L15  → 3.94
    "Division (divisor 1-5)",       # L16  → 4.2
    "Division (divisor 1-12)",      # L17  → 4.4
    "Mixed Arithmetic",             # L18  → 5.0
    "Linear Algebra (ax + b = c)",  # L19  → 6.0
    "Quadratic (ax^2 + bx + c = 0)",# L20 → 7.0
    "Kinematics (d=vt, F=ma)",      # L21  → 8.0
    "Energy (KE, Work)",            # L22  → 9.0
]
LEVEL_NAMES_PER_TABLE_BEYOND = LEVEL_NAMES_PER_TABLE + _BEYOND_ENERGY_SUFFIX  # 26L

# ── No-concept-discovery curriculum (11L-ncd) ─────────────────────────────────
# Same as standard 13L but with the two "concept discovery" warmup levels removed:
#   • Mul (0-5)  [original L4] — skipped; goes straight to Mul (0-12)
#   • Div (1-5)  [original L6] — skipped; goes straight to Div (1-12)
# Used to test Finding 3: is concept discovery still necessary with relative reward?
LEVEL_NAMES_NO_CONCEPT = [
    "Addition (0-10)",              # L0  → sem 1.1
    "Addition (0-100)",             # L1  → sem 1.2
    "Subtraction (0-10)",           # L2  → sem 2.1
    "Subtraction (0-100)",          # L3  → sem 2.2
    "Multiplication (0-12)",        # L4  → sem 3.4  (skips Mul 0-5 warmup)
    "Division (divisor 1-12)",      # L5  → sem 4.4  (skips Div 1-5 warmup)
    "Mixed Arithmetic",             # L6  → sem 5.0
    "Linear Algebra (ax + b = c)",  # L7  → sem 6.0
    "Quadratic (ax^2 + bx + c = 0)",# L8  → sem 7.0
    "Kinematics (d=vt, F=ma)",      # L9  → sem 8.0
    "Energy (KE, Work)",            # L10 → sem 9.0
]

# Operation codes fed into state vector (normalized by 8.0 in _make_state)
OP_ADD  = 0.0
OP_SUB  = 1.0
OP_MUL  = 2.0
OP_DIV  = 3.0
OP_LIN  = 4.0   # linear algebra
OP_QUAD = 5.0   # quadratic
OP_KIN  = 6.0   # kinematics  (OP_KIN + 0.1 used for F=ma variant)
OP_ENRG = 7.0   # kinetic energy KE = ½mv²
# Beyond-energy extensions (sub-codes stay within [0, 8.0] for clean normalization)
OP_PE   = 7.1   # potential energy  PE = mgh
OP_WORK = 7.2   # work              W  = Fd
OP_MOM  = 7.3   # momentum          p  = mv


class MathEvolutionEnv(gym.Env):
    """
    State:  [num1, num2, num3, operation_code, difficulty_level]
            All values normalized to [-1, 1] range for clean gradient flow.

    Action: Single continuous value — the agent's answer.

    Reward: exp(-|error| / scale) — smooth gradient, max 1.0 for perfect answer.
            The agent is never told the correct answer. It learns from reward shape alone.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, start_level=0, mastery_threshold=0.70, mastery_window=100,
                 refined_mul=False, refined_div=False, relative_reward=False,
                 per_table_mul=False, beyond_energy=False, no_concept_discovery=False,
                 seed=None):
        super().__init__()

        # Each env gets its own persistent RNG so problem sequences are
        # deterministic given the seed and independent across workers.
        self.rng = np.random.default_rng(seed)

        self.refined_mul = refined_mul
        self.refined_div = refined_div
        self.relative_reward = relative_reward
        self.per_table_mul = per_table_mul
        self.beyond_energy = beyond_energy
        self.no_concept_discovery = no_concept_discovery

        # Select level name list from all flag combinations (priority: per_table > refined_div > refined_mul)
        if no_concept_discovery:
            self.level_names = LEVEL_NAMES_NO_CONCEPT                    # 11L-ncd
        elif per_table_mul and beyond_energy:
            self.level_names = LEVEL_NAMES_PER_TABLE_BEYOND          # 26L
        elif per_table_mul:
            self.level_names = LEVEL_NAMES_PER_TABLE                 # 23L
        elif refined_mul and refined_div and beyond_energy:
            self.level_names = LEVEL_NAMES_REFINED_V2_BEYOND         # 20L
        elif refined_mul and refined_div:
            self.level_names = LEVEL_NAMES_REFINED_V2                # 17L
        elif refined_mul and beyond_energy:
            self.level_names = LEVEL_NAMES_REFINED_BEYOND            # 18L
        elif refined_mul:
            self.level_names = LEVEL_NAMES_REFINED                   # 15L
        elif beyond_energy:
            self.level_names = LEVEL_NAMES_BEYOND_ENERGY             # 16L
        else:
            self.level_names = LEVEL_NAMES                           # 13L

        self.level = start_level
        self.mastery_threshold = mastery_threshold
        self.mastery_window = mastery_window

        # ── C3 FIX: use deque(maxlen=mastery_window) instead of list + pop(0).
        # deque.append() drops the oldest element automatically (O(1) vs O(n)).
        self.recent_rewards = deque(maxlen=mastery_window)
        self.episode_count = 0

        # State: 5 values, all normalized
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Action: normalized value in [-1, 1] — environment scales to answer range
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.current_problem = None
        self.correct_answer = None
        self.answer_scale = 1.0  # used for normalization of state

        # ── C1: Build dispatch table at init time.
        # Maps level_name → generator(rng) based on curriculum flags.
        # No if/elif chains in _generate_problem() — just a dict lookup.
        self._generators = self._build_generators()

    @classmethod
    def from_config(cls, cfg: "EnvConfig", start_level: int = 0, seed=None):
        """
        Construct a MathEvolutionEnv from an EnvConfig object.

        Preferred over passing individual boolean kwargs — one config object
        travels cleanly through make_env() factories, logging, and checkpoints.

        Example:
            cfg = EnvConfig(refined_mul=True, relative_reward=True)
            envs = SyncVectorEnv([MathEvolutionEnv.from_config(cfg, seed=42+i)
                                  for i in range(n_envs)])
        """
        return cls(
            start_level=start_level,
            mastery_threshold=cfg.mastery_threshold,
            mastery_window=cfg.mastery_window,
            refined_mul=cfg.refined_mul,
            refined_div=cfg.refined_div,
            relative_reward=cfg.relative_reward,
            per_table_mul=cfg.per_table_mul,
            beyond_energy=cfg.beyond_energy,
            no_concept_discovery=cfg.no_concept_discovery,
            seed=seed,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        problem, answer, state, scale = self._generate_problem()
        self.current_problem = problem
        self.correct_answer = answer
        self.answer_scale = scale
        return state.astype(np.float32), {}

    def step(self, action):
        # Agent outputs [-1, 1] via tanh — scale to actual answer range
        predicted = float(action[0]) * self.answer_scale
        correct = self.correct_answer

        if self.relative_reward:
            # Relative reward: normalize error by the level's answer_scale.
            # This gives a consistent signal across all problems at a level —
            # an error of 3 on 5+3=8 and an error of 3 on 9+9=18 are treated
            # equally (both are 3/20 = 15% of the level's range).
            # Sharpness raised to 3.0 so the gradient is tighter near the
            # correct answer — pushes the agent to be more precise.
            scale = max(self.answer_scale, 1.0)
            error = abs(predicted - correct) / scale
            reward = float(np.exp(-error * 3.0))
        else:
            # Original reward: normalize by the individual correct answer.
            scale = max(abs(correct), 1.0)
            error = abs(predicted - correct) / scale
            reward = float(np.exp(-error * 1.5))  # softer curve → richer gradient signal

        # Track stochastic rewards for display only.
        # deque auto-drops oldest when full — no manual pop(0) needed.
        # NOTE: level-up is handled externally via advance_level(),
        # using deterministic (noise-free) actions for a true mastery signal.
        self.recent_rewards.append(reward)

        leveled_up = False  # level-up handled externally via advance_level()

        # Each episode is one problem (single step)
        terminated = True
        truncated = False

        info = {
            "correct_answer": correct,
            "predicted": predicted,
            "error": abs(predicted - correct),
            "level": self.level,
            "level_name": self.level_names[self.level],
            "mastery": np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            "leveled_up": leveled_up,
        }

        # Next state is meaningless after termination, return zeros
        next_state = np.zeros(5, dtype=np.float32)
        return next_state, reward, terminated, truncated, info

    # ── C1: Dispatch table ────────────────────────────────────────────────────

    def _build_generators(self):
        """
        Build a dict mapping level_name → generator(rng) for the active curriculum.

        Called once at __init__ time after all flags are resolved, so the table
        is exact for the specific curriculum variant in use.

        "Multiplication (0-12)" has two encodings (input_scale differs between
        standard and refined curricula) — this is resolved at build time.
        """
        p = partial

        # Multiplication (0-12): refined curricula normalize inputs by the global
        # max factor (12) for consistent encoding; original uses answer_scale (144).
        mul12_in = 12.0 if self.refined_mul else 144.0

        return {
            # ── Arithmetic ────────────────────────────────────────────────────
            "Addition (0-10)":                p(self._g_add, max_val=10,  in_scale=20.0),
            "Addition (0-100)":               p(self._g_add, max_val=100, in_scale=200.0),
            "Subtraction (0-10)":             p(self._g_sub, max_val=10,  in_scale=10.0),
            "Subtraction (0-100)":            p(self._g_sub, max_val=100, in_scale=100.0),

            # ── Multiplication sub-levels ─────────────────────────────────────
            "Multiplication (0-3)":           p(self._g_mul, max_f=3,  in_s=12.0,  ans_s=9.0),
            "Multiplication (0-5)":           p(self._g_mul, max_f=5,  in_s=25.0,  ans_s=25.0),
            "Multiplication (0-6)":           p(self._g_mul, max_f=6,  in_s=12.0,  ans_s=36.0),
            "Multiplication (0-9)":           p(self._g_mul, max_f=9,  in_s=12.0,  ans_s=81.0),
            "Multiplication (0-12)":          p(self._g_mul, max_f=12, in_s=mul12_in, ans_s=144.0),

            # ── Division sub-levels ───────────────────────────────────────────
            # Standard/refined_mul division uses in_s=25 (small scale).
            # Refined_div uses in_s=144 (global max dividend) for all div levels.
            "Division (divisor 1-3)":         p(self._g_div, max_d=3,  in_s=144.0, ans_s=3.0),
            "Division (divisor 1-5)":         p(self._g_div, max_d=5,  in_s=25.0,  ans_s=5.0),
            "Division (divisor 1-6)":         p(self._g_div, max_d=6,  in_s=144.0, ans_s=6.0),
            "Division (divisor 1-9)":         p(self._g_div, max_d=9,  in_s=144.0, ans_s=9.0),
            "Division (divisor 1-12)":        p(self._g_div, max_d=12, in_s=144.0, ans_s=12.0),

            # ── Higher operations ─────────────────────────────────────────────
            "Mixed Arithmetic":               self._g_mixed,
            "Linear Algebra (ax + b = c)":    self._g_linear,
            "Quadratic (ax^2 + bx + c = 0)":  self._g_quadratic,
            "Kinematics (d=vt, F=ma)":        self._g_kinematics,
            "Energy (KE, Work)":              self._g_energy_ke,

            # ── Beyond-energy extension ───────────────────────────────────────
            "Potential Energy (PE=mgh)":      self._g_potential_energy,
            "Work (W=Fd)":                    self._g_work,
            "Momentum (p=mv)":               self._g_momentum,

            # ── Per-table generalisation level ────────────────────────────────
            # Individual ×N tables are handled in _generate_per_table_problem().
            "Multiplication (all tables)":    self._g_mul_all_tables,
        }

    def _generate_problem(self):
        """
        Dispatch to the generator for the current curriculum level.

        C1 refactor: replaces the original 300-line if/elif chain with a
        single dict lookup. The dispatch table is built once at __init__,
        parameterised to the exact curriculum variant in use.

        C4 fix: the original code had `return self._generate_problem()` as
        a fallback (infinite recursion if an unregistered level was reached).
        Now we raise a clear ValueError instead.
        """
        lvl = self.level

        # Per-table curriculum has its own code path for ×2–×12 table levels
        # (levels 4–15); standard dispatch table handles the rest.
        if self.per_table_mul:
            return self._generate_per_table_problem(lvl, self.rng)

        # With the dispatch-by-name table, NCD level remapping is unnecessary:
        # LEVEL_NAMES_NO_CONCEPT uses the same level names as LEVEL_NAMES so
        # "Multiplication (0-12)" at NCD-L4 correctly maps to the same generator
        # as standard-L5. No index juggling needed.

        level_name = self.level_names[lvl]
        gen = self._generators.get(level_name)
        if gen is None:
            raise ValueError(
                f"No generator registered for level {lvl}: {level_name!r}. "
                f"This is a bug — all level names should be in the dispatch table."
            )
        return gen(self.rng)

    # ── Atomic generators ─────────────────────────────────────────────────────
    # Each generator takes `rng` and returns (problem_tuple, answer, state, scale).
    # Parameters are baked in via functools.partial at _build_generators() time.

    def _g_add(self, rng, max_val, in_scale):
        a = int(rng.integers(0, max_val + 1))
        b = int(rng.integers(0, max_val + 1))
        return (a, b, '+'), float(a + b), self._make_state(a, b, 0, OP_ADD, in_scale), in_scale

    def _g_sub(self, rng, max_val, in_scale):
        a = int(rng.integers(0, max_val + 1))
        b = int(rng.integers(0, a + 1))
        return (a, b, '-'), float(a - b), self._make_state(a, b, 0, OP_SUB, in_scale), in_scale

    def _g_mul(self, rng, max_f, in_s, ans_s):
        a = int(rng.integers(0, max_f + 1))
        b = int(rng.integers(0, max_f + 1))
        return (a, b, '*'), float(a * b), self._make_state(a, b, 0, OP_MUL, in_s, ans_s), ans_s

    def _g_div(self, rng, max_d, in_s, ans_s):
        b = int(rng.integers(1, max_d + 1))
        a = b * int(rng.integers(1, max_d + 1))
        return (a, b, '/'), float(a / b), self._make_state(a, b, 0, OP_DIV, in_s, ans_s), ans_s

    def _g_mixed(self, rng):
        op = rng.choice([OP_ADD, OP_SUB, OP_MUL, OP_DIV])
        return self._generate_problem_for_op(op, rng)

    def _g_linear(self, rng):
        a = float(rng.integers(1, 10))
        b = float(rng.integers(-10, 11))
        x = float(rng.integers(-10, 11))
        c = a * x + b
        return (a, b, c, 'linear'), x, self._make_state(a, b, c, OP_LIN, 100.0, 10.0), 10.0

    def _g_quadratic(self, rng):
        r1 = float(rng.integers(-5, 6))
        r2 = float(rng.integers(-5, 6))
        b_c = -(r1 + r2)
        c_c = r1 * r2
        return (1, b_c, c_c, 'quad'), min(r1, r2), self._make_state(1.0, b_c, c_c, OP_QUAD, 25.0, 5.0), 5.0

    def _g_kinematics(self, rng):
        sub = rng.choice(['d=vt', 'F=ma'])
        if sub == 'd=vt':
            v = float(rng.integers(1, 51))
            t = float(rng.integers(1, 21))
            return (v, t, 'd=vt'), v * t, self._make_state(v, t, 0, OP_KIN, 1000.0, 1000.0), 1000.0
        else:
            m = float(rng.integers(1, 101))
            a = float(rng.integers(1, 21))
            return (m, a, 'F=ma'), m * a, self._make_state(m, a, 0, OP_KIN + 0.1, 2000.0, 2000.0), 2000.0

    def _g_energy_ke(self, rng):
        m = float(rng.integers(1, 51))
        v = float(rng.integers(1, 21))
        return (m, v, 'KE'), 0.5 * m * v**2, self._make_state(m, v, 0, OP_ENRG, 10000.0, 10000.0), 10000.0

    def _g_potential_energy(self, rng):
        m = float(rng.integers(1, 101))
        h = float(rng.integers(1, 51))
        return (m, h, 'PE=mgh'), m * 9.81 * h, self._make_state(m, h, 0, OP_PE, 50000.0, 50000.0), 50000.0

    def _g_work(self, rng):
        f = float(rng.integers(1, 101))
        d = float(rng.integers(1, 51))
        return (f, d, 'W=Fd'), f * d, self._make_state(f, d, 0, OP_WORK, 5000.0, 5000.0), 5000.0

    def _g_momentum(self, rng):
        m = float(rng.integers(1, 101))
        v = float(rng.integers(1, 51))
        return (m, v, 'p=mv'), m * v, self._make_state(m, v, 0, OP_MOM, 5000.0, 5000.0), 5000.0

    def _g_mul_all_tables(self, rng):
        """Generalisation level: both factors free in [0, 12]."""
        a = float(rng.integers(0, 13))
        b = float(rng.integers(0, 13))
        return (a, b, '×'), a * b, self._make_state(a, b, 0, OP_MUL, 12.0, 144.0), 144.0

    # ── Per-table curriculum ──────────────────────────────────────────────────

    def _generate_per_table_problem(self, lvl, rng):
        """
        Generate a problem for the per-table multiplication curriculum (23L / 26L).

        Level mapping:
          L0–L3   : addition / subtraction — via standard dispatch table
          L4–L14  : ×2 through ×12 individual times tables (unique logic here)
          L15     : generalisation — both factors free in [0, 12]
          L16+    : via standard dispatch table (div, mixed, algebra, …)
        """
        if 4 <= lvl <= 14:
            # Individual times table: one factor is fixed (the table number)
            multiplier = lvl - 2        # L4→×2, L5→×3, …, L14→×12
            other = int(rng.integers(1, 13))
            # Randomly swap so the network sees both orderings equally
            if rng.random() < 0.5:
                a, b = float(multiplier), float(other)
            else:
                a, b = float(other), float(multiplier)
            # Normalise inputs by 12 (global max) for consistent encoding
            return (a, b, '×'), a * b, self._make_state(a, b, 0, OP_MUL, 12.0, 144.0), 144.0

        # All other levels (L0–L3, L15–L22 + beyond-energy) share generators
        # with the standard curriculum — dispatch by level name.
        level_name = self.level_names[lvl]
        gen = self._generators.get(level_name)
        if gen is None:
            raise ValueError(
                f"No generator for per-table level {lvl}: {level_name!r}"
            )
        return gen(rng)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _generate_problem_for_op(self, op, rng):
        """Generate a single mixed-arithmetic problem for the given operation."""
        if op == OP_ADD:
            a, b = float(rng.integers(0, 101)), float(rng.integers(0, 101))
            return (a, b, '+'), a + b, self._make_state(a, b, 0, op, 200.0), 200.0
        elif op == OP_SUB:
            a = float(rng.integers(0, 101))
            b = float(rng.integers(0, int(a) + 1))
            return (a, b, '-'), a - b, self._make_state(a, b, 0, op, 100.0), 100.0
        elif op == OP_MUL:
            a, b = float(rng.integers(0, 13)), float(rng.integers(0, 13))
            return (a, b, '*'), a * b, self._make_state(a, b, 0, op, 144.0), 144.0
        else:  # OP_DIV
            b = float(rng.integers(1, 13))
            a = b * float(rng.integers(1, 13))
            return (a, b, '/'), a / b, self._make_state(a, b, 0, op, 144.0), 144.0

    def _make_state(self, n1, n2, n3, op_code, input_scale, answer_scale=None):
        """Normalize all inputs to [-1, 1] range for the neural net.

        input_scale  — divides n1/n2/n3 so they land in [-1, 1].
        answer_scale — unused here; returned separately by generators
                       so the training loop can scale the agent's [-1,1] output.
        """
        max_level = len(self.level_names) - 1
        return np.array([
            float(n1) / input_scale,
            float(n2) / input_scale,
            float(n3) / input_scale,
            op_code / 8.0,                       # normalize op code
            float(self.level) / max_level         # normalize level index
        ], dtype=np.float32)

    def render(self):
        if self.current_problem:
            print(f"Level {self.level} ({self.level_names[self.level]}) | Problem: {self.current_problem}")

    def mastery_score(self):
        if not self.recent_rewards:
            return 0.0
        return float(np.mean(self.recent_rewards))

    def advance_level(self):
        """
        Advance to the next curriculum level.

        Called by the training loop after eval_deterministic_mastery() confirms
        the agent has truly mastered the current level — measured with noise-free
        deterministic actions, not the noisy stochastic ones used during PPO training.
        """
        if self.level < len(self.level_names) - 1:
            self.level += 1
            self.recent_rewards.clear()   # reset stochastic window for new level
            print(f"\n>>> LEVEL UP! Now at Level {self.level}: {self.level_names[self.level]}\n")
            return True
        return False  # already at max level
