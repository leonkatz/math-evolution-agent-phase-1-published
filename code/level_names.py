"""
Canonical semantic level naming for the Math Evolution Agent.

Semantic version format:  MAJOR.MINOR
  Major = math operation type
    1 = Addition        2 = Subtraction     3 = Multiplication
    4 = Division        5 = Mixed Arithmetic 6 = Linear Algebra
    7 = Quadratic       8 = Kinematics       9 = Energy
   10 = Momentum
  Minor = difficulty sub-step within that type (higher = harder)

The same semantic level means the same math, regardless of which raw
curriculum (13L / 15L / 17L / 16L / 23L etc.) produced it.

Standard curriculum (13L) skips 3.1 / 3.3 / 4.1 / 4.3 because it uses
coarser two-step progressions.  Reaching 3.4 or 4.4 always means the
same thing: "mastered that operation at full 0–12 / 1–12 range."

Per-table curriculum (23L) uses the multiplier directly as the minor for
levels ×2–×9 (3.02–3.09).  Due to Python float arithmetic, 3.10 == 3.1,
so ×10/×11/×12/generalization are encoded as 3.91–3.94.  These are looked
up with round(x, 2) precision.

Usage:
    from level_names import get_semantic, get_label, get_short_name, SEMANTIC_LEVELS

    sem = get_semantic(raw_level=11, refined_mul=False, refined_div=False)
    # → 8.0

    label = get_label(8.0)
    # → "Kinematics (d=vt, F=ma)"
"""

from __future__ import annotations

# ── Canonical level table ──────────────────────────────────────────────────────
# (semantic_float, short_operation_name, full_description)
SEMANTIC_LEVELS: list[tuple[float, str, str]] = [
    # ── Addition ──────────────────────────────────────────────────────────────
    (1.1, "Addition",       "Addition (0–10)"),
    (1.2, "Addition",       "Addition (0–100)"),
    # ── Subtraction ───────────────────────────────────────────────────────────
    (2.1, "Subtraction",    "Subtraction (0–10)"),
    (2.2, "Subtraction",    "Subtraction (0–100)"),
    # ── Multiplication: range-based (standard & refined curricula) ─────────────
    (3.1, "Multiplication", "Multiplication (0–3)"),      # refined_mul only
    (3.2, "Multiplication", "Multiplication (0–5/6)"),    # std=0-5, refined=0-6
    (3.3, "Multiplication", "Multiplication (0–9)"),      # refined_mul only
    (3.4, "Multiplication", "Multiplication (0–12)"),
    # ── Multiplication: per-table curriculum (23L) ─────────────────────────────
    # Encoded at 2 decimal places.  3.02–3.09 = ×2–×9 tables.
    # ×10/11/12 use 3.91–3.93 to avoid the 3.10==3.1 Python float collision.
    (3.02, "Multiplication", "Multiplication ×2 table"),   # per_table only
    (3.03, "Multiplication", "Multiplication ×3 table"),
    (3.04, "Multiplication", "Multiplication ×4 table"),
    (3.05, "Multiplication", "Multiplication ×5 table"),
    (3.06, "Multiplication", "Multiplication ×6 table"),
    (3.07, "Multiplication", "Multiplication ×7 table"),
    (3.08, "Multiplication", "Multiplication ×8 table"),
    (3.09, "Multiplication", "Multiplication ×9 table"),
    (3.91, "Multiplication", "Multiplication ×10 table"),  # per_table only
    (3.92, "Multiplication", "Multiplication ×11 table"),
    (3.93, "Multiplication", "Multiplication ×12 table"),
    (3.94, "Multiplication", "Multiplication (all tables)"),
    # ── Division ──────────────────────────────────────────────────────────────
    (4.1, "Division",       "Division (1–3)"),             # refined_div only
    (4.2, "Division",       "Division (1–5)"),
    (4.3, "Division",       "Division (1–9)"),             # refined_div only
    (4.4, "Division",       "Division (1–12)"),
    # ── Higher mathematics ────────────────────────────────────────────────────
    (5.0, "Mixed",          "Mixed Arithmetic"),
    (6.0, "Linear Algebra", "Linear Algebra (ax+b=c)"),
    (7.0, "Quadratic",      "Quadratic (ax²+bx+c=0)"),
    (8.0, "Kinematics",     "Kinematics (d=vt, F=ma)"),
    # ── Energy family (9.x) ───────────────────────────────────────────────────
    (9.0, "Energy",         "Energy — Kinetic (KE=½mv²)"),
    (9.1, "Energy",         "Energy — Potential (PE=mgh)"),
    (9.2, "Energy",         "Energy — Work (W=Fd)"),
    # ── Momentum (10.x) ───────────────────────────────────────────────────────
    (10.0, "Momentum",      "Momentum (p=mv)"),
]

# ── Curriculum maps: raw level index → semantic float ─────────────────────────
# Key: (per_table_mul, refined_mul, refined_div, beyond_energy)
# For backward compatibility, old 2-tuple keys (refined_mul, refined_div) also work.
_CURRICULUM_MAPS: dict = {
    # ── Original 13L (standard) ───────────────────────────────────────────────
    (False, False, False, False): [
        1.1, 1.2,               # Addition
        2.1, 2.2,               # Subtraction
        3.2, 3.4,               # Multiplication (coarse: 0-5, 0-12)
        4.2, 4.4,               # Division (coarse: 1-5, 1-12)
        5.0,                    # Mixed
        6.0, 7.0, 8.0, 9.0,    # Algebra, Quadratic, Kinematics, Energy
    ],
    # ── 16L: standard + beyond_energy ─────────────────────────────────────────
    (False, False, False, True): [
        1.1, 1.2,
        2.1, 2.2,
        3.2, 3.4,
        4.2, 4.4,
        5.0,
        6.0, 7.0, 8.0, 9.0,
        9.1, 9.2, 10.0,         # Potential Energy, Work, Momentum
    ],
    # ── 15L: refined multiplication ───────────────────────────────────────────
    (False, True, False, False): [
        1.1, 1.2,
        2.1, 2.2,
        3.1, 3.2, 3.3, 3.4,    # Multiplication (fine: 0-3, 0-6, 0-9, 0-12)
        4.2, 4.4,
        5.0,
        6.0, 7.0, 8.0, 9.0,
    ],
    # ── 18L: refined multiplication + beyond_energy ───────────────────────────
    (False, True, False, True): [
        1.1, 1.2,
        2.1, 2.2,
        3.1, 3.2, 3.3, 3.4,
        4.2, 4.4,
        5.0,
        6.0, 7.0, 8.0, 9.0,
        9.1, 9.2, 10.0,
    ],
    # ── 17L: refined multiplication + refined division ─────────────────────────
    (False, True, True, False): [
        1.1, 1.2,
        2.1, 2.2,
        3.1, 3.2, 3.3, 3.4,
        4.1, 4.2, 4.3, 4.4,    # Division (fine: 1-3, 1-5, 1-9, 1-12)
        5.0,
        6.0, 7.0, 8.0, 9.0,
    ],
    # ── 20L: refined multiplication + refined division + beyond_energy ─────────
    (False, True, True, True): [
        1.1, 1.2,
        2.1, 2.2,
        3.1, 3.2, 3.3, 3.4,
        4.1, 4.2, 4.3, 4.4,
        5.0,
        6.0, 7.0, 8.0, 9.0,
        9.1, 9.2, 10.0,
    ],
    # ── 23L: per-table multiplication ─────────────────────────────────────────
    (True, False, False, False): [
        1.1, 1.2,
        2.1, 2.2,
        3.02, 3.03, 3.04, 3.05, 3.06,   # ×2–×6 tables
        3.07, 3.08, 3.09,                # ×7–×9 tables
        3.91, 3.92, 3.93, 3.94,          # ×10–×12 tables + generalization
        4.2, 4.4,
        5.0,
        6.0, 7.0, 8.0, 9.0,
    ],
    # ── 26L: per-table multiplication + beyond_energy ─────────────────────────
    (True, False, False, True): [
        1.1, 1.2,
        2.1, 2.2,
        3.02, 3.03, 3.04, 3.05, 3.06,
        3.07, 3.08, 3.09,
        3.91, 3.92, 3.93, 3.94,
        4.2, 4.4,
        5.0,
        6.0, 7.0, 8.0, 9.0,
        9.1, 9.2, 10.0,
    ],
}

# ── Backward-compatible 2-tuple aliases ───────────────────────────────────────
_CURRICULUM_MAPS[(False, False)] = _CURRICULUM_MAPS[(False, False, False, False)]
_CURRICULUM_MAPS[(True,  False)] = _CURRICULUM_MAPS[(False, True, False, False)]
_CURRICULUM_MAPS[(True,  True)]  = _CURRICULUM_MAPS[(False, True, True,  False)]

# ── Quick lookup dict: semantic_float → (short_name, description) ─────────────
# Keyed by exact float; looked up with round(x, 2) to handle both 1- and
# 2-decimal-place semantic floats.
_SEM_LOOKUP: dict[float, tuple[str, str]] = {
    s: (short, desc) for s, short, desc in SEMANTIC_LEVELS
}


# ── Public API ─────────────────────────────────────────────────────────────────

def get_semantic(
    raw_level: int,
    refined_mul: bool = False,
    refined_div: bool = False,
    beyond_energy: bool = False,
    per_table_mul: bool = False,
    no_concept_discovery: bool = False,
) -> float:
    """Convert a raw curriculum level index to a canonical semantic float.

    Args:
        raw_level:           The integer level stored in the checkpoint / log.
        refined_mul:         True if the run used --refined_mul.
        refined_div:         True if the run used --refined_div (implies refined_mul).
        beyond_energy:       True if the run used --beyond_energy.
        per_table_mul:       True if the run used --per_table_mul.
        no_concept_discovery: True if the run used --no_concept_discovery (11L-ncd).

    Returns:
        Semantic level as float, e.g. 8.0 for Kinematics, 3.05 for ×5 table.
    """
    # 11L-ncd curriculum: skips Mul(0-5) and Div(1-5) warmups
    if no_concept_discovery:
        _ncd_sem = [1.1, 1.2, 2.1, 2.2, 3.4, 4.4, 5.0, 6.0, 7.0, 8.0, 9.0]
        if 0 <= raw_level < len(_ncd_sem):
            return _ncd_sem[raw_level]
        return float(raw_level)

    key = (
        bool(per_table_mul),
        bool(refined_mul or refined_div),
        bool(refined_div),
        bool(beyond_energy),
    )
    mapping = _CURRICULUM_MAPS.get(key, _CURRICULUM_MAPS[(False, False, False, False)])
    if 0 <= raw_level < len(mapping):
        return mapping[raw_level]
    return float(raw_level)  # out-of-range fallback


def get_label(semantic: float) -> str:
    """Full description for a semantic level, e.g. 'Kinematics (d=vt, F=ma)'.

    Tries 2-decimal precision first (for per-table levels like 3.02),
    then 1-decimal (for all standard levels like 3.4, 9.0).
    """
    entry = _SEM_LOOKUP.get(round(semantic, 2)) or _SEM_LOOKUP.get(round(semantic, 1))
    return entry[1] if entry else f"Level {semantic:.2f}"


def get_short_name(semantic: float) -> str:
    """Short operation name, e.g. 'Kinematics' or 'Multiplication'."""
    entry = _SEM_LOOKUP.get(round(semantic, 2)) or _SEM_LOOKUP.get(round(semantic, 1))
    return entry[0] if entry else f"L{semantic:.2f}"


def format_semantic(semantic: float) -> str:
    """Human-readable version string, e.g. '8.0 — Kinematics'."""
    label = get_short_name(semantic)
    return f"{semantic:.2f} — {label}"


def all_semantic_levels() -> list[tuple[float, str, str]]:
    """Return the full ordered list of (semantic, short_name, description)."""
    return list(SEMANTIC_LEVELS)


# ── Convenience: curriculum name from flags ────────────────────────────────────
def curriculum_name(
    refined_mul: bool = False,
    refined_div: bool = False,
    beyond_energy: bool = False,
    per_table_mul: bool = False,
    no_concept_discovery: bool = False,
) -> str:
    """Return a human-readable curriculum label like '13L', '17L', '23L', '11L-ncd'."""
    if no_concept_discovery:
        return "11L-ncd"
    if per_table_mul:
        return "26L" if beyond_energy else "23L"
    if refined_div:
        base = 17
    elif refined_mul:
        base = 15
    else:
        base = 13
    if beyond_energy:
        base += 3
    return f"{base}L"
