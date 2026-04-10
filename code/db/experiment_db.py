"""
ExperimentDB — PostgreSQL-backed database for Math Evolution Agent experiments.

Usage:
    from db.experiment_db import ExperimentDB

    db = ExperimentDB()  # reads DATABASE_URL from environment

    exp_id = db.create_experiment(
        run_id="seed42_rel_reset_13L_2026-03-15",
        seed=42, n_envs=1, total_steps=15_000_000,
        curriculum="13L", reward_mode="relative", entropy_mode="reset",
    )

    db.log_step(exp_id, step=1000, level=0, level_name="Addition (0-10)", ...)
    db.log_level_transition(exp_id, step=473088, from_level=0, to_level=1, ...)
    db.complete_experiment(exp_id, final_level=11, final_level_name="Kinematics")

Connection:
    Set DATABASE_URL environment variable:
      postgresql://math_agent:PASSWORD@HOST:5432/math_agent

    In Docker this is injected automatically by docker-compose.yml.
    For bare-metal runs export it before starting train.py / coordinator.py.
"""

import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import psycopg2
import psycopg2.extras
import psycopg2.extensions

from db.schema import get_semantic_level


# ── Schema path (used for bare-metal init) ────────────────────────────────────
_SCHEMA_FILE = Path(__file__).parent.parent.parent / "docker" / "postgres" / "init" / "01_schema.sql"


def _get_database_url() -> str:
    """Return the PostgreSQL connection URL.

    Priority:
      1. DATABASE_URL environment variable (always preferred — Docker injects this)
      2. Raise with a helpful message
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    raise RuntimeError(
        "DATABASE_URL environment variable is not set.\n"
        "In Docker this is set automatically by docker-compose.yml.\n"
        "For bare-metal runs export it first:\n"
        "  export DATABASE_URL=postgresql://math_agent:PASSWORD@192.168.1.141:5432/math_agent"
    )


class ExperimentDB:
    """
    PostgreSQL database for storing all experiment data.

    Thread-safety: each method opens its own connection and closes it on return,
    so multiple threads / processes can call this safely without sharing state.
    """

    def __init__(self, database_url: Optional[str] = None):
        self._database_url = database_url or _get_database_url()
        self._init_schema()
        host = self._database_url.split("@")[-1] if "@" in self._database_url else self._database_url
        print(f"ExperimentDB: connected to {host}", flush=True)

    # ── Connection + transaction helper ──────────────────────────────────────

    @contextmanager
    def _tx(self):
        """Open a connection, yield a RealDictCursor, commit on success, rollback + re-raise on error."""
        conn = psycopg2.connect(self._database_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Apply the Postgres schema (idempotent — all CREATE TABLE IF NOT EXISTS).

        In Docker this is redundant (postgres:16 applies it via docker-entrypoint-initdb.d
        on first start), but running it here makes bare-metal startup safe too.
        """
        if not _SCHEMA_FILE.exists():
            return
        schema_sql = _SCHEMA_FILE.read_text()
        conn = psycopg2.connect(self._database_url)
        try:
            with conn.cursor() as cur:
                # Execute each statement individually — psycopg2 doesn't support
                # multi-statement strings with parameters, and some PG versions
                # reject them even without parameters.
                for stmt in schema_sql.split(";"):
                    stmt = stmt.strip()
                    if stmt and not stmt.startswith("--"):
                        try:
                            cur.execute(stmt)
                        except psycopg2.errors.DuplicateTable:
                            pass  # CREATE TABLE IF NOT EXISTS handles most cases; belt-and-suspenders
                        except psycopg2.errors.DuplicateObject:
                            pass  # index or constraint already exists — safe to skip
            conn.commit()
        finally:
            conn.close()

    # ── Experiment lifecycle ─────────────────────────────────────────────────

    def create_experiment(
        self,
        run_id: str,
        seed: int,
        n_envs: int,
        total_steps: int,
        curriculum: str,
        reward_mode: str,
        entropy_mode: str,
        device: Optional[str] = None,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        rollout_size: int = 512,
        mastery_threshold: float = 0.85,
        notes: Optional[str] = None,
        imported_from: Optional[str] = None,
        params_json: Optional[dict] = None,
    ) -> int:
        """Create a new experiment record. Returns the experiment id.

        If a run with this run_id already exists and is incomplete (no
        completed_at), the stale record is deleted and replaced so a retry
        can start cleanly.  If it is already complete, a numeric suffix
        (_retry1, _retry2, …) is appended to avoid clobbering real data.
        """
        hostname = os.environ.get("MATH_AGENT_MACHINE_NAME") or socket.gethostname()
        started_at = datetime.now(timezone.utc).isoformat()
        with self._tx() as cur:
            cur.execute(
                "SELECT id, completed_at FROM experiments WHERE run_id=%s", (run_id,)
            )
            existing = cur.fetchone()
            if existing:
                exp_id = existing["id"]
                completed_at = existing["completed_at"]
                if completed_at is None:
                    print(f"[DB] Stale incomplete run '{run_id}' (id={exp_id}) — removing before retry.")
                    # Null out job_queue FK before deleting so the FK constraint
                    # introduced in the params_json migration doesn't block the delete.
                    cur.execute("UPDATE job_queue SET experiment_id = NULL WHERE experiment_id=%s", (exp_id,))
                    cur.execute("DELETE FROM training_log WHERE experiment_id=%s", (exp_id,))
                    cur.execute("DELETE FROM level_transitions WHERE experiment_id=%s", (exp_id,))
                    cur.execute("DELETE FROM checkpoints WHERE experiment_id=%s", (exp_id,))
                    cur.execute("DELETE FROM experiments WHERE id=%s", (exp_id,))
                else:
                    suffix = 1
                    while True:
                        cur.execute(
                            "SELECT 1 FROM experiments WHERE run_id=%s",
                            (f"{run_id}_retry{suffix}",),
                        )
                        if not cur.fetchone():
                            break
                        suffix += 1
                    run_id = f"{run_id}_retry{suffix}"
                    print(f"[DB] Completed run already exists — using run_id '{run_id}'.")
            params_json_str = json.dumps(params_json) if params_json is not None else None
            cur.execute(
                """
                INSERT INTO experiments
                    (run_id, seed, n_envs, total_steps, curriculum, reward_mode,
                     entropy_mode, device, hostname, hidden_dim, learning_rate,
                     rollout_size, mastery_threshold, started_at, notes, imported_from,
                     params_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (run_id, seed, n_envs, total_steps, curriculum, reward_mode,
                 entropy_mode, device, hostname, hidden_dim, learning_rate,
                 rollout_size, mastery_threshold, started_at, notes, imported_from,
                 params_json_str),
            )
            exp_id = cur.fetchone()["id"]

            # If spawned by the worker, link this experiment back to its job_queue
            # row so the queue and experiments table are connected.
            job_id_env = os.environ.get("MATH_AGENT_JOB_ID")
            if job_id_env:
                try:
                    cur.execute(
                        "UPDATE job_queue SET experiment_id = %s WHERE id = %s",
                        (exp_id, int(job_id_env)),
                    )
                except Exception as link_exc:
                    # Non-fatal: job may have been deleted or the table may not exist yet.
                    print(f"[DB] Could not link experiment {exp_id} to job {job_id_env}: {link_exc}")

            return exp_id

    def complete_experiment(
        self,
        experiment_id: int,
        final_level: int,
        final_level_name: str,
    ):
        """Mark an experiment as complete with final results."""
        sem = get_semantic_level(final_level_name)
        completed_at = datetime.now(timezone.utc).isoformat()
        with self._tx() as cur:
            cur.execute(
                """
                UPDATE experiments
                SET final_level=%s, final_level_name=%s, final_sem_level=%s, completed_at=%s
                WHERE id=%s
                """,
                (final_level, final_level_name, sem, completed_at, experiment_id),
            )

    def get_experiment(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return experiment record as a dict, or None if not found."""
        with self._tx() as cur:
            cur.execute("SELECT * FROM experiments WHERE run_id=%s", (run_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def get_experiment_by_id(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        with self._tx() as cur:
            cur.execute("SELECT * FROM experiments WHERE id=%s", (experiment_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Return all experiments ordered by start time."""
        with self._tx() as cur:
            cur.execute("SELECT * FROM experiments ORDER BY started_at")
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ── Training log ────────────────────────────────────────────────────────

    def log_step(
        self,
        experiment_id: int,
        step: int,
        level: int,
        level_name: str,
        avg_reward: float,
        det_mastery: float,
        stoch_mastery: float,
        entropy_coef: float,
        loss: float,
        elapsed_min: float,
        cpu_pct: float = 0.0,
        wall_time: Optional[str] = None,
    ):
        """Append one training log row.

        Silently skips on psycopg2.OperationalError (transient network/DB errors)
        so a DB hiccup never crashes a long-running train.py process.
        The CSV log is the authoritative record.
        """
        sem = get_semantic_level(level_name)
        wt = wall_time or datetime.now(timezone.utc).isoformat()
        try:
            with self._tx() as cur:
                cur.execute(
                    """
                    INSERT INTO training_log
                        (experiment_id, step, level, level_name, sem_level,
                         avg_reward, det_mastery, stoch_mastery, entropy_coef,
                         loss, elapsed_min, cpu_pct, wall_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (experiment_id, step, level, level_name, sem,
                     avg_reward, det_mastery, stoch_mastery, entropy_coef,
                     loss, elapsed_min, cpu_pct, wt),
                )
        except psycopg2.Error as e:
            print(f"[DB] log_step skipped (step {step}): {e}", flush=True)

    def bulk_log_steps(self, experiment_id: int, rows: List[Dict[str, Any]]):
        """Bulk-insert many training log rows (used by CSV importer)."""
        records = [
            (
                experiment_id,
                r["step"], r["level"], r["level_name"],
                get_semantic_level(r["level_name"]),
                r.get("avg_reward"), r.get("det_mastery"), r.get("stoch_mastery"),
                r.get("entropy_coef"), r.get("loss"), r.get("elapsed_min"),
                r.get("cpu_pct", 0.0), r.get("wall_time"),
            )
            for r in rows
        ]
        with self._tx() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO training_log
                    (experiment_id, step, level, level_name, sem_level,
                     avg_reward, det_mastery, stoch_mastery, entropy_coef,
                     loss, elapsed_min, cpu_pct, wall_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                records,
                page_size=500,
            )

    def get_training_log(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Return all training log rows for an experiment."""
        with self._tx() as cur:
            cur.execute(
                "SELECT * FROM training_log WHERE experiment_id=%s ORDER BY step",
                (experiment_id,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ── Level transitions ────────────────────────────────────────────────────

    def log_level_transition(
        self,
        experiment_id: int,
        step: int,
        from_level: int,
        to_level: int,
        from_level_name: str,
        to_level_name: str,
        det_mastery: float,
        stoch_mastery: float = 0.0,
        steps_on_level: int = 0,
        elapsed_min: float = 0.0,
        checkpoint_path: Optional[str] = None,
    ):
        """Record a level-up event."""
        from_sem = get_semantic_level(from_level_name)
        to_sem = get_semantic_level(to_level_name)
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO level_transitions
                    (experiment_id, step, from_level, to_level,
                     from_level_name, to_level_name, from_sem_level, to_sem_level,
                     det_mastery, stoch_mastery, steps_on_level, elapsed_min, checkpoint_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (experiment_id, step, from_level, to_level,
                 from_level_name, to_level_name, from_sem, to_sem,
                 det_mastery, stoch_mastery, steps_on_level, elapsed_min, checkpoint_path),
            )

    def get_level_transitions(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Return all level transitions for an experiment."""
        with self._tx() as cur:
            cur.execute(
                "SELECT * FROM level_transitions WHERE experiment_id=%s ORDER BY step",
                (experiment_id,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ── Checkpoints ─────────────────────────────────────────────────────────

    def log_checkpoint(
        self,
        experiment_id: int,
        step: int,
        level: int,
        checkpoint_path: str,
        meta_path: Optional[str] = None,
        checkpoint_type: str = "periodic",
        det_mastery: Optional[float] = None,
    ):
        """Record a saved checkpoint."""
        with self._tx() as cur:
            cur.execute(
                """
                INSERT INTO checkpoints
                    (experiment_id, step, level, checkpoint_path, meta_path,
                     checkpoint_type, det_mastery)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (experiment_id, step, level, checkpoint_path, meta_path,
                 checkpoint_type, det_mastery),
            )

    # ── Research finding queries ─────────────────────────────────────────────

    def query_finding1_det_vs_stoch(self) -> List[Dict[str, Any]]:
        """Finding 1: det_mastery vs stoch_mastery at every level transition."""
        with self._tx() as cur:
            cur.execute(
                """
                SELECT
                    e.run_id, e.seed, e.reward_mode, e.entropy_mode, e.curriculum,
                    lt.step, lt.from_level_name, lt.to_level_name,
                    lt.from_sem_level, lt.det_mastery, lt.stoch_mastery,
                    lt.det_mastery - lt.stoch_mastery AS mastery_gap,
                    lt.steps_on_level
                FROM level_transitions lt
                JOIN experiments e ON lt.experiment_id = e.id
                ORDER BY e.run_id, lt.step
                """
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def query_finding2_final_levels(self) -> List[Dict[str, Any]]:
        """Finding 2: Final semantic level by reward mode and entropy mode."""
        with self._tx() as cur:
            cur.execute(
                """
                SELECT
                    run_id, seed, reward_mode, entropy_mode, curriculum,
                    final_level, final_level_name, final_sem_level,
                    total_steps
                FROM experiments
                WHERE completed_at IS NOT NULL
                ORDER BY reward_mode, entropy_mode, seed
                """
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def query_finding5_curriculum_density(self) -> List[Dict[str, Any]]:
        """Finding 5: Compare final level by curriculum type."""
        with self._tx() as cur:
            cur.execute(
                """
                SELECT
                    curriculum, reward_mode, entropy_mode,
                    COUNT(*) as n_runs,
                    AVG(final_sem_level) as mean_sem,
                    MIN(final_sem_level) as min_sem,
                    MAX(final_sem_level) as max_sem
                FROM experiments
                WHERE completed_at IS NOT NULL
                GROUP BY curriculum, reward_mode, entropy_mode
                ORDER BY curriculum, reward_mode
                """
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def query_steps_to_mastery(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Return steps-to-mastery per level for a single experiment."""
        with self._tx() as cur:
            cur.execute(
                """
                SELECT
                    from_level, from_level_name, from_sem_level,
                    steps_on_level, det_mastery, step as step_of_transition
                FROM level_transitions
                WHERE experiment_id=%s
                ORDER BY step
                """,
                (experiment_id,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> Dict[str, Any]:
        """Return a quick summary of the database contents."""
        with self._tx() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM experiments")
            n_exp = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM training_log")
            n_log = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM level_transitions")
            n_trans = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM checkpoints")
            n_ckpt = cur.fetchone()["n"]
        host = self._database_url.split("@")[-1] if "@" in self._database_url else self._database_url
        return {
            "experiments": n_exp,
            "training_log_rows": n_log,
            "level_transitions": n_trans,
            "checkpoints": n_ckpt,
            "database": host,
        }
