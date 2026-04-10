#!/usr/bin/env python3
"""
Math Evolution Agent — Live Training Dashboard (Mac)
Reads run data from local CSVs + worker HTTP endpoints.
Reads queue and machine status from coordinator API.
Run: python3 dashboard_mac.py   →   http://localhost:8081
"""

import csv as csv_mod
import http.server
import json
import os
import re
import socketserver
import threading
import time
import urllib.request
import urllib.error
from io import StringIO
from pathlib import Path

# Load machine config — use defaults if not available
try:
    import os as _os
    import yaml as _yaml
    from config import get_config as _get_config
    _cfg = _get_config()
    LOGS_DIR        = Path(_cfg.data_path)
    COORDINATOR_URL = _cfg.coordinator_url
    PORT            = _cfg.dashboard_port or 8081
    # Build WORKER_URLS from machines.yaml: all worker machines
    _machines_path = _os.path.join(_os.path.dirname(__file__), "..", "config", "machines.yaml")
    with open(_machines_path) as _f:
        _registry = _yaml.safe_load(_f)
    WORKER_URLS = {
        _name: _m["worker_url"]
        for _name, _m in _registry.get("machines", {}).items()
        if "worker" in _m.get("role", "") and _m.get("worker_url")
    }
    WORKER_URL = next(iter(WORKER_URLS.values()), "http://192.168.1.141:8085")
    LOCAL_MACHINE  = _cfg.name
except Exception:
    LOGS_DIR        = Path(os.environ.get("DATA_PATH", str(Path(__file__).parent.parent / "data")))
    COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "")
    LOCAL_MACHINE   = os.environ.get("MATH_AGENT_MACHINE_NAME", "local")
    PORT            = int(os.environ.get("DASHBOARD_PORT", "8081"))
    WORKER_URLS     = {}
    WORKER_URL      = ""
WORKER_REFRESH = 60
MIN_STEPS      = 0
BUILD_VERSION  = os.environ.get("BUILD_VERSION", "dev")

# ── Coordinator / queue data ───────────────────────────────────────────────────
_queue_data        = []
_machines_data     = []
_experiments_data  = []
_coord_status      = {"ok": False, "last_ok": None, "error": "", "build_version": ""}
_coord_lock        = threading.Lock()

def _fetch_json(url: str, timeout: int = 8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _refresh_coordinator():
    try:
        q = _fetch_json(f"{COORDINATOR_URL}/api/queue")
        m = _fetch_json(f"{COORDINATOR_URL}/api/machines")
        e = _fetch_json(f"{COORDINATOR_URL}/api/experiments")
        h = _fetch_json(f"{COORDINATOR_URL}/health")
        with _coord_lock:
            _queue_data.clear()
            _queue_data.extend(q)
            _machines_data.clear()
            _machines_data.extend(m)
            _experiments_data.clear()
            _experiments_data.extend(e)
        _coord_status.update({
            "ok": True,
            "last_ok": time.time(),
            "error": "",
            "build_version": h.get("build_version", ""),
        })
    except Exception as ex:
        _coord_status.update({"ok": False, "error": str(ex)})

def _coordinator_loop():
    _refresh_coordinator()
    while True:
        time.sleep(30)
        _refresh_coordinator()

threading.Thread(target=_coordinator_loop, daemon=True).start()

# ── Colors by run tag ─────────────────────────────────────────────────────────
TAG_COLORS = {
    "cold_L4":                "#ff9da7",
    "cold_L2":                "#f28e2b",
    "entropy_ablation_reset": "#b07aa1",
    "entropy_reset":          "#b07aa1",
    "entropy_fixed":          "#9c755f",
    "nenv1_sweep":            "#59a14f",
    "full_refined":           "#76b7b2",
    "refined_mul":            "#edc948",
}
_PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
            "#76b7b2", "#b07aa1", "#ff9da7", "#9c755f", "#edc948"]
_used_colors: dict[str, str] = {}
_palette_idx = 0

def color_for_tag(tag: str) -> str:
    global _palette_idx
    for k, c in TAG_COLORS.items():
        if k in tag:
            return c
    if tag not in _used_colors:
        _used_colors[tag] = _PALETTE[_palette_idx % len(_PALETTE)]
        _palette_idx += 1
    return _used_colors[tag]

# ── Run labelling from filename stem ──────────────────────────────────────────
def tag_from_stem(stem: str) -> str:
    m = re.match(r"training_log_seed\d+_?(.*)", stem)
    return (m.group(1) or "baseline") if m else stem

def label_from_stem(stem: str, source: str) -> str:
    tag  = tag_from_stem(stem)
    sm   = re.search(r"seed(\d+)", stem)
    seed = f"s{sm.group(1)}" if sm else "?"
    src  = "W" if source == "worker" else "L"
    tag_str = tag.replace("_", " ") if tag != "baseline" else "base"
    return f"{tag_str} {seed} [{src}]"

# ── CSV parser — handles 7-col, 10-col, and 12-col formats ───────────────────
def parse_csv(text: str) -> list[dict]:
    from datetime import datetime, timezone
    data = []
    for row in csv_mod.reader(StringIO(text)):
        if not row or row[0].strip().lower() in ("step", ""):
            continue
        try:
            n = len(row)
            step    = int(row[0])
            level   = int(row[1])
            reward  = float(row[3])
            det_mastery  = float(row[4])
            stoch_mastery = float(row[5]) if n >= 6 else det_mastery
            if n >= 10:
                entropy = float(row[6])
                elapsed = float(row[8])
            else:
                # very old 7-col format: step,level,name,reward,mastery,loss,elapsed
                entropy = 0.0
                elapsed = float(row[6])
            # wall_time at col 10 — convert to Unix seconds for JS arithmetic
            wall_ts = None
            if n >= 11 and row[10].strip():
                try:
                    wall_ts = datetime.strptime(row[10].strip(),
                                                "%Y-%m-%d %H:%M:%S").replace(
                                                tzinfo=timezone.utc).timestamp()
                except ValueError:
                    pass
            data.append({"step": step, "level": level, "reward": reward,
                          "mastery": det_mastery, "stoch_mastery": stoch_mastery,
                          "entropy": entropy,
                          "elapsed_min": elapsed, "wall_ts": wall_ts})
        except (ValueError, IndexError):
            continue
    return data

def downsample(data: list, n: int = 500) -> list:
    if len(data) <= n:
        return data
    step = max(1, len(data) // n)
    out  = data[::step]
    if out[-1] is not data[-1]:
        out = out + [data[-1]]
    return out

# ── Local log discovery ───────────────────────────────────────────────────────
def discover_local() -> list[tuple]:
    """Returns (stem, source, machine, data) tuples.
    source is "local" for Mac runs, "worker" for synced remote runs.
    machine is the hostname (e.g. "dell") or "mac" for local.
    """
    runs = []
    # Mac-local runs (trained on this machine)
    for p in sorted(LOGS_DIR.glob("training_log*.csv")):
        try:
            data = parse_csv(p.read_text())
        except Exception:
            continue
        if data and data[-1]["step"] >= MIN_STEPS:
            runs.append((p.stem, "local", LOCAL_MACHINE, data))
    # Worker-synced runs — one subdirectory per machine: logs/worker/{machine}/
    worker_dir = LOGS_DIR / "worker"
    if worker_dir.is_dir():
        for machine_dir in sorted(worker_dir.iterdir()):
            if not machine_dir.is_dir():
                continue
            machine_name = machine_dir.name
            for p in sorted(machine_dir.glob("training_log*.csv")):
                try:
                    data = parse_csv(p.read_text())
                except Exception:
                    continue
                if data and data[-1]["step"] >= MIN_STEPS:
                    runs.append((p.stem, "worker", machine_name, data))
    return runs

# ── Training runs cache — fetched from coordinator /api/training_runs ─────────
# Falls back to local CSV scan if the coordinator is unreachable.
_db_runs:   list[dict] = []     # list of dicts returned by /api/training_runs
_db_lock    = threading.Lock()
_db_status  = {"ok": False, "last_ok": None, "error": "", "source": "none"}
DB_REFRESH  = 60                # seconds between refreshes

def _refresh_training_runs():
    try:
        runs = _fetch_json(f"{COORDINATOR_URL}/api/training_runs", timeout=30)
        with _db_lock:
            _db_runs.clear()
            _db_runs.extend(runs)
        _db_status.update({"ok": True, "last_ok": time.time(), "error": "",
                           "source": "postgres", "count": len(runs)})
        return
    except Exception as ex:
        _db_status.update({"ok": False, "error": str(ex), "source": "none"})

    # Fallback: local CSV scan (used during startup before coordinator is ready)
    try:
        csv_runs = discover_local()
        converted = [
            {
                "stem":    stem,
                "run_tag": tag_from_stem(stem),
                "seed":    int(__import__("re").search(r"seed(\d+)", stem).group(1))
                           if __import__("re").search(r"seed(\d+)", stem) else 0,
                "machine": machine,
                "data":    data,
                "source":  source,
            }
            for stem, source, machine, data in csv_runs
        ]
        with _db_lock:
            _db_runs.clear()
            _db_runs.extend(converted)
        _db_status.update({"ok": True, "last_ok": time.time(),
                           "source": "csv_fallback", "count": len(converted)})
    except Exception as ex2:
        _db_status.update({"ok": False, "error": str(ex2), "source": "none"})

def _training_runs_loop():
    _refresh_training_runs()        # immediate on startup
    while True:
        time.sleep(DB_REFRESH)
        _refresh_training_runs()

threading.Thread(target=_training_runs_loop, daemon=True).start()

def get_db_runs() -> list[dict]:
    with _db_lock:
        return list(_db_runs)

# ── Worker HTTP fetch ─────────────────────────────────────────────────────────
_worker_runs:  list[dict] = []
_worker_lock   = threading.Lock()
_worker_status = {"ok": False, "last_ok": None, "error": ""}

def _refresh_workers():
    runs = []
    any_ok = False
    last_error = ""
    for machine_name, url in WORKER_URLS.items():
        try:
            with urllib.request.urlopen(f"{url}/jobs", timeout=10) as r:
                payload = json.loads(r.read().decode())
            jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
            for j in jobs:
                if not isinstance(j, dict):
                    continue
                runs.append({
                    "run_tag": j.get("run_tag", "unknown"),
                    "stem":    f"training_log_seed{j.get('seed', 0)}_{j.get('run_tag', 'unknown')}",
                    "seed":    j.get("seed", 0),
                    "machine": machine_name,
                    "data":    [],   # live CSV history not available via worker API
                    "status":  {
                        "step":        j.get("step", 0),
                        "level":       j.get("level", 0),
                        "level_name":  j.get("level_name", ""),
                        "reward":      j.get("reward", 0),
                        "mastery":     j.get("mastery", 0),
                        "entropy":     j.get("entropy", 0),
                        "elapsed_min": j.get("elapsed_min", 0),
                    },
                })
            any_ok = True
        except Exception as e:
            last_error = f"{machine_name}: {e}"
    with _worker_lock:
        _worker_runs.clear()
        _worker_runs.extend(runs)
    if any_ok:
        _worker_status.update({"ok": True, "last_ok": time.time(), "error": last_error})
    else:
        _worker_status.update({"ok": False, "error": last_error})

def _worker_loop():
    _refresh_workers()          # immediate on startup
    while True:
        time.sleep(WORKER_REFRESH)
        _refresh_workers()

threading.Thread(target=_worker_loop, daemon=True).start()

def get_worker_runs() -> list[dict]:
    with _worker_lock:
        return list(_worker_runs)

# ── Combine & build API response ─────────────────────────────────────────────
def get_all_data() -> list[dict]:
    result = []

    # Build queue status lookup, job_id lookup, checkpoint offset, and queued-stems set.
    # Keyed by (run_tag, seed, machine) when machine is known, else (run_tag, seed).
    queue_status_map    = {}   # (run_tag, seed[, machine]) → status string
    job_id_map          = {}   # (run_tag, seed[, machine]) → job id int
    checkpoint_off_map  = {}   # (run_tag, seed) → step offset from checkpoint (0 if fresh start)
    queued_stems = set()
    # Status priority: running > done > failed > assigned/queued > cancelled
    # Higher number = wins when two jobs share the same (run_tag, seed, machine) key.
    _STATUS_PRIORITY = {"running": 4, "done": 3, "failed": 2, "assigned": 1, "queued": 1, "cancelled": 0}

    def _better_status(new_st: str, old_st: str) -> bool:
        return _STATUS_PRIORITY.get(new_st, 0) > _STATUS_PRIORITY.get(old_st, 0)

    import re as _re2
    with _coord_lock:
        for job in _queue_data:
            rt      = job.get("run_tag") or ""
            seed    = int(job.get("seed") or 0)
            st      = job.get("status") or ""
            machine = job.get("machine") or ""
            # Machine-specific key takes precedence; fallback key for lookups without machine
            key_m = (rt, seed, machine) if machine else (rt, seed)
            if key_m not in queue_status_map or _better_status(st, queue_status_map[key_m]):
                queue_status_map[key_m] = st
                job_id_map[key_m]       = job.get("id")
            # Also store under (rt, seed) as fallback
            if (rt, seed) not in queue_status_map or _better_status(st, queue_status_map[(rt, seed)]):
                queue_status_map[(rt, seed)] = st
                job_id_map[(rt, seed)]       = job.get("id")
            # Parse checkpoint step from flags so display can show true total steps
            try:
                flags = job.get("flags") or "{}"
                if isinstance(flags, str):
                    flags = __import__("json").loads(flags)
                ckpt = flags.get("checkpoint", "")
                cm = _re2.search(r"agent_step_(\d+)\.pt", ckpt)
                checkpoint_off_map[(rt, seed)] = int(cm.group(1)) if cm else 0
            except Exception:
                checkpoint_off_map[(rt, seed)] = 0
            if st == "queued":
                queued_stems.add(f"training_log_seed{seed}_{rt}")

    # Build runs indexed by (stem, machine) from Postgres (or CSV fallback)
    import re as _re
    csv_by_stem = {}
    for run in get_db_runs():
        stem    = run.get("stem") or ""
        machine = run.get("machine") or "unknown"
        source  = run.get("source") or "worker"
        data    = run.get("data") or []

        if not stem:
            continue
        if stem in queued_stems:
            continue   # not started yet — skip tile entirely

        tag = tag_from_stem(stem)
        m = _re.match(r'training_log_seed(\d+)_(.+)', stem)
        qstatus     = None
        job_id      = None
        ckpt_offset = 0
        if m:
            seed_int = int(m.group(1))
            run_tag  = m.group(2)
            qstatus     = queue_status_map.get((run_tag, seed_int, machine),
                          queue_status_map.get((run_tag, seed_int)))
            job_id      = job_id_map.get((run_tag, seed_int, machine),
                          job_id_map.get((run_tag, seed_int)))
            ckpt_offset = checkpoint_off_map.get((run_tag, seed_int), 0)

        # data from Postgres is already downsampled; CSV fallback data needs it
        display_data = data if _db_status.get("source") == "postgres" else downsample(data)

        csv_by_stem[(stem, machine)] = {
            "stem":              stem,
            "name":              label_from_stem(stem, source),
            "source":            source,
            "machine":           machine,
            "color":             color_for_tag(tag),
            "dash":              False,
            "data":              display_data,
            "status":            data[-1] if data else None,
            "queue_status":      qstatus,
            "job_id":            job_id,
            "checkpoint_offset": ckpt_offset,
            # Passed through from experiments table so the JS interrupted filter works.
            "final_level_name":  run.get("final_level_name"),
        }

    # Live worker API runs — merge with CSV if available, otherwise add standalone
    for r in get_worker_runs():
        tag     = r.get("run_tag", "baseline")
        stem    = r.get("stem", f"seed{r.get('seed',0)}_{tag}")
        machine = r.get("machine", "worker")
        key     = (stem, machine)
        if key in csv_by_stem:
            # Prefer live status (more current) but keep CSV history for charts
            csv_by_stem[key]["status"] = r.get("status") or csv_by_stem[key]["status"]
        else:
            # No CSV yet — show live-only tile with dashed style
            seed_live = int(r.get("seed") or 0)
            result.append({
                "stem":         stem,
                "name":         label_from_stem(stem, "worker"),
                "source":       "worker",
                "machine":      machine,
                "color":        color_for_tag(tag),
                "dash":         True,
                "data":         [],
                "status":       r.get("status"),
                "queue_status":      queue_status_map.get((tag, seed_live)),
                "job_id":            job_id_map.get((tag, seed_live)),
                "checkpoint_offset": checkpoint_off_map.get((tag, seed_live), 0),
            })

    result.extend(csv_by_stem.values())

    # Sort: active runs first, then by total step desc (checkpoint offset + csv step).
    # A run is "done" if queue_status says so OR if total steps hit the target.
    def _is_done_py(r):
        if r.get("queue_status") in ("done", "cancelled"):
            return True
        s = r.get("status")
        if not s:
            return True
        return (s["step"] + r.get("checkpoint_offset", 0)) >= 15_000_000

    result.sort(key=lambda r: (
        _is_done_py(r),
        -(r.get("job_id") or 0) if _is_done_py(r) else
        -((r["status"]["step"] + r.get("checkpoint_offset", 0)) if r.get("status") else 0)
    ))
    return result


def get_run_detail(stem: str, machine: str = None) -> dict:
    """Return full-resolution rows + raw log lines + checkpoints for one run.

    If machine is supplied the search is scoped to that machine's directory so
    two runs with the same stem on different machines (e.g. Dell #75 and Mac #76
    both producing training_log_seed42_abs_reset_fresh.csv) return the right file.
    """
    import csv as _csv

    worker_dir = LOGS_DIR / "worker"

    if machine and machine != LOCAL_MACHINE:
        # Worker machine explicitly requested — look only in logs/worker/<machine>/
        candidate_paths = []
        if worker_dir.is_dir():
            machine_dir = worker_dir / machine
            if machine_dir.is_dir():
                candidate_paths.append(machine_dir / f"{stem}.csv")
    elif machine and machine == LOCAL_MACHINE:
        # Local machine explicitly requested — skip worker subdirs
        candidate_paths = [LOGS_DIR / f"{stem}.csv"]
    else:
        # No machine specified — legacy fallback: local first, then all workers
        candidate_paths = [LOGS_DIR / f"{stem}.csv"]
        if worker_dir.is_dir():
            for md in worker_dir.iterdir():
                if md.is_dir():
                    candidate_paths.append(md / f"{stem}.csv")

    csv_path = next((p for p in candidate_paths if p.exists()), None)
    if csv_path is None:
        return {"found": False, "rows": [], "raw_rows": [], "checkpoints": []}

    # Parse full data (no downsample) for zoomed chart
    full_data = parse_csv(csv_path.read_text())

    # Raw rows for the log table — last 30, all columns
    raw_rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = _csv.reader(f)
            all_lines = [r for r in reader if r and r[0].strip().lower() not in ("step", "")]
        for row in all_lines[-30:]:
            if len(row) >= 8:
                raw_rows.append({
                    "step":         row[0],
                    "level":        row[1],
                    "level_name":   row[2] if len(row) > 2 else "",
                    "reward":       row[3] if len(row) > 3 else "",
                    "det_mastery":  row[4] if len(row) > 4 else "",
                    "stoch_mastery":row[5] if len(row) > 5 else "",
                    "entropy":      row[6] if len(row) > 6 else "",
                    "loss":         row[7] if len(row) > 7 else "",
                    "wall_time":    row[10] if len(row) > 10 else "",
                })
    except Exception:
        pass

    # Find matching checkpoint manifest
    checkpoints = []
    # Derive seed + tag from stem: training_log_seed42_rel_reset -> seed=42, tag=rel_reset
    import re as _re
    m = _re.match(r'training_log_seed(\d+)_(.+)', stem)
    if m:
        seed, tag = m.group(1), m.group(2)
        manifest_name = f"checkpoints_manifest_seed{seed}_{tag}.csv"
        manifest_candidates = [LOGS_DIR / manifest_name]
        if worker_dir.is_dir():
            for md in worker_dir.iterdir():
                if md.is_dir():
                    manifest_candidates.append(md / manifest_name)
        manifest_path = next((p for p in manifest_candidates if p.exists()), None)
        if manifest_path:
            try:
                with open(manifest_path, newline="") as f:
                    reader = _csv.DictReader(f)
                    for row in reader:
                        checkpoints.append({
                            "step":       row.get("step", ""),
                            "level_name": row.get("level_name", ""),
                            "type":       row.get("checkpoint_type", ""),
                            "path":       row.get("checkpoint_path", ""),
                            "mastery":    row.get("det_mastery", ""),
                            "wall_time":  row.get("wall_time", ""),
                        })
            except Exception:
                pass

    return {
        "found":       True,
        "rows":        full_data,      # full res for chart
        "raw_rows":    raw_rows,       # last 30 for table
        "checkpoints": checkpoints,
    }


# ── HTML / JS frontend ────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Math Evolution Agent — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e0e0e0; }
  /* ── Tab bar ── */
  .tab-bar { display: flex; gap: 0; border-bottom: 2px solid #1e2133;
             background: #13151f; padding: 0 16px; position: sticky; top: 0; z-index: 100; }
  .tab-btn { padding: 12px 20px; font-size: 0.85rem; font-weight: 600; color: #666;
             cursor: pointer; border: none; background: none; border-bottom: 3px solid transparent;
             margin-bottom: -2px; transition: color 0.15s; }
  .tab-btn:hover { color: #ccc; }
  .tab-btn.active { color: #fff; border-bottom-color: #4e79a7; }
  /* ── Content area ── */
  .tab-content { padding: 20px 16px; }
  h1   { font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; color: #fff; }
  .subtitle { font-size: 0.85rem; color: #888; margin-bottom: 20px; }
  /* ── Cards ── */
  .status-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                 gap: 12px; margin-bottom: 24px; }
  .card { background: #1a1d27; border-radius: 10px; padding: 14px 16px;
          border-left: 4px solid #4e79a7; cursor: pointer; transition: opacity 0.15s, box-shadow 0.15s; }
  .card:hover { box-shadow: 0 0 0 1px #444; }
  .card.selected { box-shadow: 0 0 0 2px var(--card-color, #4e79a7); opacity: 1; }
  .card.deselected { opacity: 0.3; }
  /* ── Run detail panel ── */
  #run-detail { background: #1a1d27; border-radius: 10px; padding: 20px 24px;
                margin: 0 0 24px 0; display: none; border: 1px solid #2a2d3e; }
  #run-detail h2 { font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
                   letter-spacing: 0.08em; color: #888; margin-bottom: 16px; }
  #run-detail .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 900px) { #run-detail .detail-grid { grid-template-columns: 1fr; } }
  #run-detail .detail-section h3 { font-size: 0.75rem; font-weight: 700; color: #666;
                                    text-transform: uppercase; letter-spacing: 0.06em;
                                    margin-bottom: 10px; }
  #run-detail .detail-chart-box { height: 220px; }
  .log-table { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  .log-table th { color: #555; font-weight: 600; text-align: right; padding: 3px 8px 6px;
                  border-bottom: 1px solid #2a2d3e; white-space: nowrap; }
  .log-table th:first-child { text-align: left; }
  .log-table td { color: #ccc; text-align: right; padding: 3px 8px;
                  border-bottom: 1px solid #1e2030; font-family: monospace; white-space: nowrap; }
  .log-table td:first-child { text-align: left; color: #888; }
  .log-table tr:last-child td { color: #fff; font-weight: 600; }
  .log-table tr:hover td { background: #21253a; }
  .ckpt-table { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
  .ckpt-table th { color: #555; font-weight: 600; text-align: left; padding: 3px 8px 6px;
                   border-bottom: 1px solid #2a2d3e; }
  .ckpt-table td { color: #ccc; padding: 3px 8px; border-bottom: 1px solid #1e2030;
                   font-family: monospace; font-size: 0.68rem; }
  .ckpt-table tr:last-child td { color: #7fba7f; }
  .ckpt-tag { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.65rem;
              font-weight: 700; background: #2a3a2a; color: #7fba7f; }
  .ckpt-tag.levelup { background: #3a2a1a; color: #c8874a; }
  .ckpt-tag.milestone { background: #1a2a3a; color: #4a8ac8; }
  .detail-close { float: right; background: none; border: none; color: #555; cursor: pointer;
                  font-size: 1.1rem; padding: 0; line-height: 1; }
  .detail-close:hover { color: #aaa; }
  .detail-stat-row { display: flex; gap: 20px; margin-bottom: 14px; flex-wrap: wrap; }
  .detail-stat { text-align: center; }
  .detail-stat .val { font-size: 1.1rem; font-weight: 700; color: #fff; }
  .detail-stat .lbl { font-size: 0.68rem; color: #555; text-transform: uppercase; }
  .card-title { font-size: 0.8rem; font-weight: 600; color: #aaa;
                text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
  .card-level { font-size: 1.05rem; font-weight: 700; color: #fff; margin-bottom: 4px; }
  .card-stats { display: flex; gap: 14px; font-size: 0.82rem; color: #bbb; flex-wrap: wrap; }
  .stat-label { color: #666; font-size: 0.75rem; }
  /* ── Charts ── */
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 700px) { .charts { grid-template-columns: 1fr; } }
  .chart-box { background: #1a1d27; border-radius: 10px; padding: 16px; }
  .chart-title { font-size: 0.9rem; font-weight: 600; color: #ccc; margin-bottom: 12px; }
  canvas { max-height: 240px; }
  .refresh { font-size: 0.75rem; color: #555; margin-top: 20px; text-align: center; }
  /* ── Badges ── */
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
           font-size: 0.7rem; font-weight: 600; }
  .badge-l0  { background: #2a2a3a; color: #888; }
  .badge-ok  { background: #1a3a1a; color: #59a14f; }
  .badge-hi  { background: #1a2a3a; color: #4e79a7; }
  .no-data   { color: #555; font-style: italic; font-size: 0.85rem; }
  .done-tag  { font-size: 0.7rem; color: #f28e2b; margin-left: 6px; }
  .seg-ctrl { display: inline-flex; border: 1px solid #333; border-radius: 6px; overflow: hidden; }
  .seg-btn  { background: #1a1d27; border: none; border-right: 1px solid #333; color: #666;
              padding: 6px 14px; font-size: 0.82rem; cursor: pointer; transition: background 0.15s, color 0.15s; }
  .seg-btn:last-child { border-right: none; }
  .seg-btn:hover  { background: #252836; color: #ccc; }
  .seg-btn.active { background: #252836; color: #fff; }
  .src-local { font-size: 0.65rem; color: #4e79a7; margin-left: 4px; }
  .src-worker  { font-size: 0.65rem; color: #ff9da7; margin-left: 4px; }
  /* ── Tables ── */
  .data-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 12px; }
  .data-table th { text-align: left; color: #444; font-weight: 600; font-size: 0.72rem;
                   text-transform: uppercase; letter-spacing: 0.04em;
                   padding: 6px 8px; border-bottom: 1px solid #222; }
  .data-table td { padding: 6px 8px; border-bottom: 1px solid #181a23; color: #bbb; }
  .data-table tr:last-child td { border-bottom: none; }
  .data-table tr:hover td { background: #1e2133; }
  /* ── Status badges ── */
  .sbadge { display: inline-block; padding: 1px 7px; border-radius: 10px;
            font-size: 0.68rem; font-weight: 700; }
  .sb-queued    { background: #2a2a3a; color: #888; }
  .sb-assigned  { background: #2a2800; color: #edc948; }
  .sb-running   { background: #0a1a2a; color: #4e79a7; }
  .sb-done      { background: #0a2a0a; color: #59a14f; }
  .sb-failed    { background: #2a0a0a; color: #e15759; }
  .sb-cancelled { background: #1a1a1a; color: #555; }
  /* ── Registry sem-level colors ── */
  .sl-low    { color: #666; }
  .sl-mid    { color: #edc948; }
  .sl-high   { color: #f28e2b; }
  .sl-top    { color: #59a14f; font-weight: 700; }
  /* ── Sortable table headers ── */
  th.sortable { cursor: pointer; user-select: none; white-space: nowrap; }
  th.sortable:hover { color: #fff; }
  th.sortable .sort-ind { margin-left: 4px; color: #4e79a7; font-size: 0.7rem; }
  th.sortable.sort-asc .sort-ind::after  { content: " ▲"; }
  th.sortable.sort-desc .sort-ind::after { content: " ▼"; }
  th.sortable:not(.sort-asc):not(.sort-desc) .sort-ind::after { content: " ⇅"; color: #444; }
  /* ── Filter buttons ── */
  .filter-bar { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
  .filter-btn { padding: 4px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
                border: 1px solid #333; background: #1a1d27; color: #888; cursor: pointer; }
  .filter-btn.active { background: #4e79a7; color: #fff; border-color: #4e79a7; }
  /* ── Machine cards (infra tab) ── */
  .machine-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                  gap: 12px; margin-bottom: 24px; }
  .machine-card { background: #1a1d27; border-radius: 10px; padding: 14px 16px; border-left: 4px solid #333; }
  .mcard-name { font-size: 1rem; font-weight: 700; color: #fff; }
  .mcard-role { font-size: 0.7rem; color: #555; margin-left: 8px; }
  .mcard-host { font-size: 0.75rem; color: #666; margin-top: 4px; }
  .mcard-status { font-size: 0.85rem; font-weight: 700; margin-top: 8px; }
  .mcard-meta { font-size: 0.72rem; color: #555; margin-top: 4px; }
  /* ── Legend (copied from queue_service.py) ── */
  .legend-grid  { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
  @media (max-width: 820px) { .legend-grid { grid-template-columns: 1fr; } }
  .legend-panel { background: #1a1d27; border-radius: 10px; padding: 18px 20px; }
  .legend-panel h3 { font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
                     letter-spacing: 0.06em; color: #aaa; margin-bottom: 12px; border-bottom: 1px solid #222; padding-bottom: 8px; }
  .legend-panel h3 span { font-weight: 400; text-transform: none; letter-spacing: 0;
                          color: #555; }
  .lvl-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  .lvl-table th { text-align: left; color: #444; font-weight: 600; font-size: 0.7rem;
                  text-transform: uppercase; padding: 5px 6px; border-bottom: 1px solid #222; }
  .lvl-table td { padding: 5px 6px; border-bottom: 1px solid #181a23; color: #bbb; }
  .lvl-table tr:last-child td { border-bottom: none; }
  .lvl-table tr:hover td { background: #1e2133; }
  .lsem { font-family: monospace; font-size: 0.77rem; color: #4e79a7; font-weight: 700; }
  .lsem.ceiling { color: #f28e2b; }
  .lsem.breakthrough { color: #59a14f; }
  .lsem.energy { color: #ffdd57; }
  .lraw { font-size: 0.72rem; color: #444; font-family: monospace; }
  .ldomain { font-weight: 600; color: #ccc; }
  .ldesc { font-size: 0.74rem; color: #666; }
  .ltag-general  { display: inline-block; font-size: 0.63rem; padding: 1px 5px;
                   border-radius: 8px; background: #1a2233; color: #4e79a7; white-space: nowrap; }
  .ltag-project  { display: inline-block; font-size: 0.63rem; padding: 1px 5px;
                   border-radius: 8px; background: #1e1a2a; color: #b07aa1; white-space: nowrap; }
  .ltag-infra    { display: inline-block; font-size: 0.63rem; padding: 1px 5px;
                   border-radius: 8px; background: #1a2a1a; color: #59a14f; white-space: nowrap; }
  .glos-row { display: grid; grid-template-columns: minmax(110px,auto) 1fr auto;
              gap: 8px 12px; padding: 6px 0; border-bottom: 1px solid #1a1a23; align-items: start; }
  .glos-row:last-child { border-bottom: none; }
  .glos-key  { font-family: monospace; color: #e0c080; font-size: 0.78rem; font-weight: 600; }
  .glos-val  { color: #999; line-height: 1.4; }
  .legend-note { font-size: 0.74rem; color: #444; margin-top: 10px; line-height: 1.5; }
  .curr-row  { display: grid; grid-template-columns: 60px 40px 1fr; gap: 8px 12px;
               padding: 6px 0; border-bottom: 1px solid #1a1a23; align-items: start; }
  .curr-row:last-child { border-bottom: none; }
  .curr-name { font-family: monospace; color: #4e79a7; font-weight: 700; }
  .curr-n    { color: #555; font-size: 0.74rem; }
  .curr-desc { color: #999; line-height: 1.4; }
  /* ── Infra section headers ── */
  .section-title { font-size: 0.75rem; font-weight: 700; color: #555; text-transform: uppercase;
                   letter-spacing: 0.08em; margin: 20px 0 10px; }
  /* Markdown rendered content */
  .md-content h1 { color: #fff; font-size: 1.6rem; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid #2a2d45; padding-bottom: 0.3rem; }
  .md-content h2 { color: #e0e0e0; font-size: 1.2rem; margin: 1.3rem 0 0.4rem; }
  .md-content h3 { color: #bbb; font-size: 1rem; margin: 1rem 0 0.3rem; }
  .md-content p  { margin: 0.6rem 0; }
  .md-content code { background: #1a1c2e; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; color: #a8d8a8; }
  .md-content pre { background: #1a1c2e; padding: 12px 16px; border-radius: 6px; overflow-x: auto; border: 1px solid #2a2d45; }
  .md-content pre code { background: none; padding: 0; color: #c8e6c9; }
  .md-content blockquote { border-left: 3px solid #4e79a7; margin: 0.8rem 0; padding: 4px 16px; color: #aaa; background: #141622; }
  .md-content hr { border: none; border-top: 1px solid #2a2d45; margin: 1.5rem 0; }
  .md-content a  { color: #6ab0de; }
  .md-content ul, .md-content ol { padding-left: 1.5rem; margin: 0.5rem 0; }
  .md-content li { margin: 0.25rem 0; }
  .md-content table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  .md-content th { background: #1e2133; color: #888; font-size: 0.8rem; text-align: left; padding: 6px 10px; border-bottom: 1px solid #2a2d45; }
  .md-content td { padding: 5px 10px; border-bottom: 1px solid #181a23; color: #bbb; font-size: 0.85rem; }
  /* Paper nav */
  #paper-nav a { display: block; color: #555; text-decoration: none; font-size: 0.78rem; padding: 3px 6px; border-radius: 3px; margin: 1px 0; line-height: 1.4; }
  #paper-nav a:hover { color: #ccc; background: #1e2133; }
  #paper-nav a.h1-link { color: #777; font-weight: 600; margin-top: 8px; }
  #paper-nav a.h2-link { padding-left: 14px; }
  /* ── Experiment cards ── */
  .exp-card { background: #13151f; border: 1px solid #1e2133; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .exp-card h3 { color: #e0e0e0; font-size: 1rem; margin: 0 0 4px; }
  .exp-card .exp-desc { color: #666; font-size: 0.8rem; margin-bottom: 12px; }
  .exp-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; margin-left: 8px; }
  .exp-badge.running { background: #1a3a1a; color: #59a14f; }
  .exp-badge.queued  { background: #1e1e2e; color: #888; }
  .exp-badge.done    { background: #2a2200; color: #ffdd57; }
  .exp-badge.failed  { background: #2a1010; color: #e15759; }
  .exp-badge.mixed   { background: #1a2a3a; color: #4e79a7; }
  /* ── Architecture tab ── */
  .arch-section { margin: 24px 0; }
  .arch-section h2 { color: #e0e0e0; font-size: 1.1rem; border-bottom: 1px solid #2a2d45; padding-bottom: 6px; margin-bottom: 12px; }
  .arch-flow { background: #0d0f1a; border: 1px solid #2a2d45; border-radius: 6px; padding: 16px 20px; font-family: monospace; font-size: 0.82rem; color: #8abedc; line-height: 1.7; overflow-x: auto; }
  .arch-params { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
  .arch-param-card { background: #131520; border: 1px solid #2a2d45; border-radius: 6px; padding: 10px 14px; }
  .arch-param-key { color: #666; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .arch-param-val { color: #e0e0e0; font-size: 1.1rem; font-weight: 600; margin-top: 3px; }
  .arch-formula { background: #0d0f1a; border: 1px solid #4e79a7; border-radius: 8px; padding: 20px; text-align: center; font-size: 1.2rem; color: #a8d8ff; font-family: monospace; margin: 12px 0; }
  .arch-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .arch-table th { text-align: left; color: #555; font-size: 0.72rem; padding: 5px 10px; border-bottom: 1px solid #2a2d45; }
  .arch-table td { padding: 5px 10px; border-bottom: 1px solid #181a23; color: #bbb; }
  .arch-table tr:hover td { background: #1e2133; }
  .arch-table .sem-cell { color: #4e79a7; font-weight: 600; }
  /* ── Experiments (run_tag grouped) tab ── */
  .exp-group-header { background: #1a1c2e; padding: 10px 12px; font-weight: 600; color: #ccc; font-size: 0.85rem; border-top: 2px solid #2a2d45; display: flex; justify-content: space-between; align-items: center; }
  .exp-desc { color: #666; font-size: 0.78rem; font-weight: 400; margin-left: 10px; }
  .exp-badge-counts { font-size: 0.72rem; display: flex; gap: 6px; }
  .flag-pill { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 0.68rem; font-weight: 600; margin: 0 2px; }
  .fp-rel   { background: #1a3a1a; color: #59a14f; }
  .fp-reset { background: #1a2a3a; color: #4e79a7; }
  .fp-anneal{ background: #3a3a1a; color: #e8c840; }
  .fp-fixed { background: #2a2a2a; color: #888; }
  .fp-ncd   { background: #3a2a1a; color: #f28e2b; }
  .fp-17L   { background: #2a1a3a; color: #b07fe0; }
</style>
</head>
<body>

<!-- ── Tab bar ── -->
<div class="tab-bar">
  <button class="tab-btn active" id="tb-training"  onclick="showTab('training')">&#128202; Training</button>
  <button class="tab-btn"        id="tb-infra"     onclick="showTab('infra')">&#128187; Infrastructure</button>
  <button class="tab-btn"        id="tb-queue"     onclick="showTab('queue')">&#9889; Queue</button>
  <button class="tab-btn"        id="tb-registry"  onclick="showTab('registry')">&#128203; Registry</button>
  <button class="tab-btn"        id="tb-legend"    onclick="showTab('legend')">&#128218; Legend</button>
  <button class="tab-btn" id="tb-hype"  onclick="showTab('hype')">&#x1F680; Hype</button>
  <button class="tab-btn" id="tb-paper" onclick="showTab('paper')">&#x1F4C4; Paper</button>
  <button class="tab-btn" id="tb-notes" onclick="showTab('notes')">&#x1F4DD; Notes</button>
  <button class="tab-btn" id="tb-exps"  onclick="showTab('exps')">&#x1F9EA; Experiments</button>
  <button class="tab-btn" id="tb-arch"  onclick="showTab('arch')">&#x2699;&#xFE0F; Architecture</button>
</div>

<!-- ════════════════════ TAB: TRAINING ════════════════════ -->
<div id="tab-training" class="tab-content">
  <h1>&#9889; Math Evolution Agent — Training</h1>
  <p class="subtitle" id="ts">Loading...</p>
  <div style="margin-bottom:14px; display:flex; align-items:center; gap:12px">
    <div class="seg-ctrl">
      <button class="seg-btn active" id="seg-all"    onclick="setDoneFilter('all')">All</button>
      <button class="seg-btn"        id="seg-active" onclick="setDoneFilter('active')">Active</button>
      <button class="seg-btn"        id="seg-done"   onclick="setDoneFilter('done')">Done</button>
    </div>
    <button class="filter-btn active" id="btn-smoke-filter" onclick="toggleSmokeFilter()">Hide Smoke</button>
    <button class="filter-btn active" id="btn-interrupted-filter" onclick="toggleInterruptedFilter()">Hide Interrupted</button>
    <a id="btn-clear-sel" href="#" onclick="clearSelection();return false;"
       style="display:none;font-size:0.8rem;color:#888;text-decoration:none">&#x2715; Clear selection</a>
  </div>
  <div class="status-grid" id="status-grid"></div>
  <div id="run-detail">
    <button class="detail-close" onclick="closeDetail()" title="Close">&#x2715;</button>
    <h2 id="detail-title">Run Detail</h2>
    <div class="detail-stat-row" id="detail-stats"></div>
    <div class="detail-grid">
      <div class="detail-section">
        <h3>Reward &amp; Mastery &mdash; full run</h3>
        <div class="detail-chart-box"><canvas id="detail-chart"></canvas></div>
      </div>
      <div class="detail-section">
        <h3>Last 30 log entries</h3>
        <div id="detail-log" style="overflow-x:auto"></div>
      </div>
    </div>
    <div class="detail-section" style="margin-top:18px">
      <h3>Checkpoints</h3>
      <div id="detail-ckpts"></div>
    </div>
  </div>
  <div class="charts">
    <div class="chart-box"><div class="chart-title">Advancement Score — deterministic (threshold: 0.85)</div><canvas id="chart-mastery"></canvas></div>
    <div class="chart-box"><div class="chart-title">Advancement Score — stochastic <span style="font-weight:400;font-size:0.75rem;color:#666">— gate score for stoch_gate runs; exploration noise drags this below det</span></div><canvas id="chart-mastery-stoch"></canvas></div>
    <div class="chart-box"><div class="chart-title">Level</div><canvas id="chart-level"></canvas></div>
    <div class="chart-box"><div class="chart-title">Reward (smoothed)</div><canvas id="chart-reward"></canvas></div>
    <div class="chart-box"><div class="chart-title">Entropy (exploration)</div><canvas id="chart-entropy"></canvas></div>
    <div class="chart-box"><div class="chart-title">Level vs Wall Time <span style="font-weight:400;font-size:0.75rem;color:#666">— slope = speed of progression; flat = stuck at a level</span></div><canvas id="chart-level-time"></canvas></div>
    <div class="chart-box"><div class="chart-title">Training Speed <span style="font-weight:400;font-size:0.75rem;color:#666">— steps/min · gradual decay normal (harder evals) · sudden drop = problem</span></div><canvas id="chart-speed"></canvas></div>
    <div class="chart-box"><div class="chart-title">Run Staleness <span style="font-weight:400;font-size:0.75rem;color:#666">— minutes since last log entry · green &lt;10m = active · orange = check it · red &gt;60m = stalled/done</span></div><canvas id="chart-health"></canvas></div>
  </div>
  <p class="refresh">Auto-refreshes every 30s &nbsp;|&nbsp; Worker poll every 60s &nbsp;|&nbsp; [L]=Mac local &nbsp;[W]=Dell worker (synced CSV) &nbsp;dashed=live worker poll</p>
</div>

<!-- ════════════════════ TAB: INFRASTRUCTURE ════════════════════ -->
<div id="tab-infra" class="tab-content" style="display:none">
  <h1>&#128187; Infrastructure</h1>
  <p class="subtitle" id="infra-ts">Coordinator and machine status.</p>
  <div style="margin-bottom:12px">
    <span id="coord-badge"></span>
    &nbsp;&nbsp;
    <button class="filter-btn active" onclick="refreshInfra()">Refresh</button>
  </div>
  <div class="section-title">Machines</div>
  <div class="machine-grid" id="machine-grid">
    <div style="color:#555;font-style:italic">Loading...</div>
  </div>
</div>

<!-- ════════════════════ TAB: QUEUE ════════════════════ -->
<div id="tab-queue" class="tab-content" style="display:none">
  <h1>&#9889; Job Queue</h1>
  <p class="subtitle" id="queue-ts">All jobs from coordinator. Auto-refreshes every 30s.</p>
  <div class="filter-bar">
    <button class="filter-btn active" id="qf-all"    onclick="setQueueFilter('all')">All</button>
    <button class="filter-btn"        id="qf-active" onclick="setQueueFilter('active')">Active</button>
    <button class="filter-btn"        id="qf-done"   onclick="setQueueFilter('done')">Done</button>
    <button class="filter-btn"        id="qf-failed" onclick="setQueueFilter('failed')">Failed</button>
  </div>
  <div id="queue-table-wrap">
    <div style="color:#555;font-style:italic">Loading...</div>
  </div>
</div>

<!-- ════════════════════ TAB: REGISTRY ════════════════════ -->
<div id="tab-registry" class="tab-content" style="display:none">
  <h1>&#128203; Experiment Registry</h1>
  <p class="subtitle" id="reg-ts">All experiments, ordered by start time. Auto-refreshes every 60s.</p>
  <div id="registry-table-wrap">
    <div style="color:#555;font-style:italic">Loading...</div>
  </div>
</div>

<!-- ════════════════════ TAB: LEGEND ════════════════════ -->
<div id="tab-legend" class="tab-content" style="display:none">
  <h1>&#128218; Reference &amp; Legend</h1>
  <p class="subtitle">Curriculum levels, run-name conventions, and RL terminology.</p>

  <!-- Level table (full width) -->
  <div class="legend-panel" style="margin-bottom:20px">
    <h3>Phase 1 Standard Curriculum <span>— 13 levels (L0–L12), with semantic scale</span></h3>
    <table class="lvl-table">
      <thead><tr><th>Sem</th><th>Raw L</th><th>Domain</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td><span class="lsem">1.1</span></td><td class="lraw">L0</td><td class="ldomain">Addition</td>      <td class="ldesc">Integers 0–10</td></tr>
        <tr><td><span class="lsem">1.2</span></td><td class="lraw">L1</td><td class="ldomain">Addition</td>      <td class="ldesc">Integers 0–100</td></tr>
        <tr><td><span class="lsem">2.1</span></td><td class="lraw">L2</td><td class="ldomain">Subtraction</td>   <td class="ldesc">Integers 0–10</td></tr>
        <tr><td><span class="lsem">2.2</span></td><td class="lraw">L3</td><td class="ldomain">Subtraction</td>   <td class="ldesc">Integers 0–100</td></tr>
        <tr><td><span class="lsem">3.2</span></td><td class="lraw">L4</td><td class="ldomain">Multiplication</td><td class="ldesc">Factors 0–5 (concept intro)</td></tr>
        <tr style="background:#1e1a10">
          <td><span class="lsem ceiling">3.4</span></td><td class="lraw">L5</td>
          <td class="ldomain" style="color:#f28e2b">Multiplication</td>
          <td class="ldesc" style="color:#f28e2b">Factors 0–12 — <strong>original baseline ceiling</strong> (all seeds stuck here)</td>
        </tr>
        <tr><td><span class="lsem">4.2</span></td><td class="lraw">L6</td><td class="ldomain">Division</td>      <td class="ldesc">Divisors 1–5</td></tr>
        <tr><td><span class="lsem">4.4</span></td><td class="lraw">L7</td><td class="ldomain">Division</td>      <td class="ldesc">Divisors 1–12</td></tr>
        <tr><td><span class="lsem">5.0</span></td><td class="lraw">L8</td><td class="ldomain">Mixed</td>         <td class="ldesc">All four operations</td></tr>
        <tr><td><span class="lsem">6.0</span></td><td class="lraw">L9</td><td class="ldomain">Linear Algebra</td><td class="ldesc">ax + b = c</td></tr>
        <tr><td><span class="lsem">7.0</span></td><td class="lraw">L10</td><td class="ldomain">Quadratic</td>    <td class="ldesc">ax² + bx + c = 0 (real roots only)</td></tr>
        <tr style="background:#0f1a10">
          <td><span class="lsem breakthrough">8.0</span></td><td class="lraw">L11</td>
          <td class="ldomain" style="color:#59a14f">Kinematics</td>
          <td class="ldesc" style="color:#59a14f">d = vt, F = ma — <strong>breakthrough: 14/20 seeds reach here (70%) at 15M steps</strong></td>
        </tr>
        <tr style="background:#181a0a">
          <td><span class="lsem energy">9.0</span></td><td class="lraw">L12</td>
          <td class="ldomain" style="color:#ffdd57">Energy</td>
          <td class="ldesc" style="color:#ffdd57">KE = ½mv² — frontier: confirmation runs at 30M steps in progress</td>
        </tr>
      </tbody>
    </table>
    <p class="legend-note">
      <strong>Semantic level</strong> = project-defined progress scale stable across curriculum variants.
      8.0 always means Kinematics mastered, whether the curriculum used 13, 17, or 23 levels to get there.
      The baseline configuration (abs reward + stochastic gating) never passed <span class="lsem ceiling" style="font-size:0.8rem">3.4</span>.
    </p>
  </div>

  <div class="legend-grid">
    <!-- Curriculum variants -->
    <div class="legend-panel">
      <h3>Curriculum Variants</h3>
      <div class="curr-row"><span class="curr-name">13-lvl</span><span class="curr-n">L0–L12</span><span class="curr-desc">Standard — default. Two Mul levels (0-5, 0-12), two Div levels (1-5, 1-12).</span></div>
      <div class="curr-row"><span class="curr-name">15-lvl</span><span class="curr-n">L0–L14</span><span class="curr-desc">Refined mul — four Mul sub-levels (0-3, 0-6, 0-9, 0-12) instead of two.</span></div>
      <div class="curr-row"><span class="curr-name">17-lvl</span><span class="curr-n">L0–L16</span><span class="curr-desc">Refined mul + refined div — four sub-levels each for Mul and Div.</span></div>
      <div class="curr-row"><span class="curr-name">11-lvl-ncd</span><span class="curr-n">L0–L10</span><span class="curr-desc">No concept discovery — skips Mul(0-5) and Div(1-5) warmup levels; jumps straight to full tables. (Finding 6 test)</span></div>
      <div class="curr-row"><span class="curr-name">23-lvl</span><span class="curr-n">L0–L22</span><span class="curr-desc">Per-table mul — each ×2 through ×12 times table as its own level, then generalization.</span></div>
      <div class="curr-row"><span class="curr-name">16-lvl</span><span class="curr-n">L0–L15</span><span class="curr-desc">Standard + beyond-energy — extends past KE to PE, Work (W=Fd), Momentum (p=mv).</span></div>
    </div>

    <!-- Run name conventions -->
    <div class="legend-panel">
      <h3>Run Name Components <span>— how names are assembled</span></h3>
      <div class="glos-row"><span class="glos-key">rel / abs</span>    <span class="glos-val">Reward mode: <em>relative</em> normalises by level scale; <em>absolute</em> by individual answer. Rel fixes flat gradients at high levels.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">anneal</span>        <span class="glos-val">Entropy coefficient decays linearly 0.01 → 0.001 over total training steps.</span><span class="ltag-general">general RL</span></div>
      <div class="glos-row"><span class="glos-key">reset</span>         <span class="glos-val">Entropy resets to 0.01 each time agent advances to a new curriculum level — prevents premature convergence.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">fixed</span>         <span class="glos-val">Entropy coefficient held constant at 0.01 throughout training.</span><span class="ltag-general">general RL</span></div>
      <div class="glos-row"><span class="glos-key">s42 / s7 / s123</span><span class="glos-val">Random seed number — controls weight init and environment sampling. Different seeds = independently trained agents.</span><span class="ltag-general">general ML</span></div>
      <div class="glos-row"><span class="glos-key">ncd</span>            <span class="glos-val">No concept discovery — uses the 11L-ncd curriculum (see left).</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">stoch_gate</span>    <span class="glos-val">Stochastic advancement evaluation — the original bug: uses noisy policy to check advancement, creating artificial ceilings.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">energy_30M</span>    <span class="glos-val">30M-step run targeting Energy (sem 9.0) — extended compute budget to test whether the Kinematics ceiling is compute-limited vs a fundamental barrier.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">cold_LN / fresh_lN</span><span class="glos-val">Cold-start ablation: fresh random weights, no checkpoint, training begins directly at level N (e.g. cold_L4 skips L0–L3; fresh_l12 starts at Energy). Two naming conventions for the same idea — cold_L* came first, fresh_l* used for later runs. Tests whether early concept-discovery levels are load-bearing.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">B4b</span>           <span class="glos-val">Experiment-series label from the paper's B-series (B1–B4 = controlled ablations for Findings 1–4). B4 = entropy mechanism; b = second variant (fixed vs anneal under rel reward).</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">worker / local</span><span class="glos-val">Machine the run executed on. worker = remote training machine polled via HTTP. local = coordinator machine running this dashboard.</span><span class="ltag-infra">infra</span></div>
    </div>

    <!-- General RL/ML terms -->
    <div class="legend-panel">
      <h3>General RL / ML Terms</h3>
      <div class="glos-row"><span class="glos-key">PPO</span>           <span class="glos-val">Proximal Policy Optimization — the RL algorithm. Clips the policy update ratio to prevent large destructive steps. Standard in continuous-action tasks.</span><span class="ltag-general">general RL</span></div>
      <div class="glos-row"><span class="glos-key">entropy</span>       <span class="glos-val">Measure of policy randomness. High entropy = more exploration. Low entropy = near-deterministic. The entropy <em>coefficient</em> weights the entropy bonus in the PPO loss.</span><span class="ltag-general">general RL</span></div>
      <div class="glos-row"><span class="glos-key">Actor-Critic</span> <span class="glos-val">Network architecture with two heads: actor (outputs action distribution) and critic (estimates expected return V(s) for advantage calculation).</span><span class="ltag-general">general RL</span></div>
      <div class="glos-row"><span class="glos-key">rollout</span>       <span class="glos-val">A batch of environment steps collected before each PPO update. This project uses 512 steps × N envs per rollout.</span><span class="ltag-general">general RL</span></div>
      <div class="glos-row"><span class="glos-key">n_envs</span>        <span class="glos-val">Number of parallel environment copies. Local runs typically use 1; worker machines may use more for higher CPU throughput.</span><span class="ltag-general">general RL</span></div>
      <div class="glos-row"><span class="glos-key">curriculum RL</span> <span class="glos-val">Training regime where task difficulty increases over time. The agent advances to harder problems only after mastering easier ones.</span><span class="ltag-general">general RL</span></div>
      <div class="glos-row"><span class="glos-key">cold start</span>    <span class="glos-val">Training begins from level 0 (Addition). Opposite: warm start from a mid-curriculum checkpoint. All standard runs are cold starts. See also: <em>cold_LN / fresh_lN</em> in Run Name Components above.</span><span class="ltag-general">general RL</span></div>
    </div>

    <!-- Project-specific terms -->
    <div class="legend-panel">
      <h3>Project-Specific Terms</h3>
      <div class="glos-row"><span class="glos-key">advancement score</span> <span class="glos-val">Mean reward over 200 test problems evaluated <em>deterministically</em> (actor mean, no noise). Curriculum advances when advancement score ≥ 0.85. Stored as <code>det_mastery</code> in training logs.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">det / stoch</span>  <span class="glos-val"><em>Deterministic</em>: evaluate using actor mean μ (no sampling) — accurate measure of learned behavior. <em>Stochastic</em>: sample from full distribution — introduces noise that can mask learning.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">semantic level</span><span class="glos-val">Project-defined difficulty scale (e.g. 8.0 = Kinematics). Stable across curriculum variants — 8.0 always means Kinematics regardless of whether the curriculum used 13 or 17 levels.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">answer_scale</span>  <span class="glos-val">The level's expected answer range (e.g. ~200 for Energy problems). Used by relative reward to normalise error consistently across all problems at a level.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">reward</span>        <span class="glos-val">exp(−|error| × sharpness). Smooth, max 1.0 for exact answer. Both absolute and relative reward use this form — they differ only in how <em>error</em> is normalised.</span><span class="ltag-project">project</span></div>
      <div class="glos-row"><span class="glos-key">concept disc.</span> <span class="glos-val">Warmup sub-levels that introduce a new operation gently (e.g. Mul 0-5 before Mul 0-12). The NCD curriculum tests whether these are redundant when relative reward is used.</span><span class="ltag-project">project</span></div>
    </div>
  </div><!-- end legend-grid -->
</div><!-- end tab-legend -->

<!-- ════════════════════ TAB: HYPE ════════════════════ -->
<div id="tab-hype"  class="tab-content" style="display:none">
  <div id="hype-body" style="max-width:800px;margin:0 auto;font-size:1.05rem;line-height:1.8;color:#ddd"></div>
</div>

<!-- ════════════════════ TAB: PAPER ════════════════════ -->
<div id="tab-paper" class="tab-content" style="display:none">
  <div style="display:flex;gap:32px">
    <nav id="paper-nav" style="width:220px;flex-shrink:0;position:sticky;top:20px;align-self:flex-start;max-height:80vh;overflow-y:auto"></nav>
    <div id="paper-body" style="max-width:720px;flex:1;font-size:0.95rem;line-height:1.75;color:#ccc"></div>
  </div>
</div>

<!-- ════════════════════ TAB: NOTES ════════════════════ -->
<div id="tab-notes" class="tab-content" style="display:none">
  <div id="notes-body" class="md-content" style="max-width:800px;margin:0 auto"></div>
</div>

<!-- ════════════════════ TAB: EXPERIMENTS ════════════════════ -->
<div id="tab-exps" class="tab-content" style="display:none">
  <h1>&#x1F9EA; Experiments</h1>
  <p style="color:#555;font-size:0.8rem;margin-bottom:16px">Jobs grouped by experiment — what each run_tag was testing and its current status.</p>
  <div id="exps-body"></div>
</div>

<!-- ════════════════════ TAB: ARCHITECTURE ════════════════════ -->
<div id="tab-arch" class="tab-content" style="display:none">
  <h1>&#x2699;&#xFE0F; Architecture</h1>
  <div style="max-width:900px">

    <!-- Section 1: Data Flow -->
    <div class="arch-section">
      <h2>Data Flow</h2>
      <pre class="arch-flow">MathEvolutionEnv          MathReasoningNetwork          PPO Update
─────────────────         ────────────────────          ──────────
 [num1, num2, num3]  →    4× ResidualBlock        →    clip_eps=0.2
 [op_code/8.0    ]        (Linear→LayerNorm→SiLU)      γ=0.99
 [level/12.0     ]             ↓           ↓            4 epochs
                          Actor Head   Critic Head
                          (tanh mean)  (value est.)
                               ↓
                         action ∈ [-1,1]
                               ↓
                     reward = exp(-|error|×1.5)</pre>
    </div>

    <!-- Section 2: Curriculum -->
    <div class="arch-section">
      <h2>Curriculum — 13 Levels</h2>
      <table class="arch-table">
        <thead><tr><th>L</th><th>Name</th><th>Sem</th><th>Range</th></tr></thead>
        <tbody>
          <tr><td>0</td><td>Addition (0-10)</td><td class="sem-cell">1.0</td><td>0-10</td></tr>
          <tr><td>1</td><td>Addition (0-100)</td><td class="sem-cell">2.0</td><td>0-100</td></tr>
          <tr><td>2</td><td>Subtraction (0-10)</td><td class="sem-cell">2.0</td><td>0-10</td></tr>
          <tr><td>3</td><td>Subtraction (0-100)</td><td class="sem-cell">3.0</td><td>0-100</td></tr>
          <tr><td>4</td><td>Multiplication (0-5)</td><td class="sem-cell">3.0</td><td>0-5</td></tr>
          <tr><td>5</td><td>Multiplication (0-12)</td><td class="sem-cell">3.4</td><td>0-12</td></tr>
          <tr><td>6</td><td>Division (÷1-5)</td><td class="sem-cell">4.0</td><td>÷1-5</td></tr>
          <tr><td>7</td><td>Division (÷1-12)</td><td class="sem-cell">4.5</td><td>÷1-12</td></tr>
          <tr><td>8</td><td>Mixed Arithmetic</td><td class="sem-cell">5.0</td><td>all ops</td></tr>
          <tr><td>9</td><td>Linear Algebra ax+b=c</td><td class="sem-cell">6.0</td><td>—</td></tr>
          <tr><td>10</td><td>Quadratic ax²+bx+c=0</td><td class="sem-cell">7.0</td><td>—</td></tr>
          <tr><td>11</td><td>Kinematics d=vt, F=ma</td><td class="sem-cell">8.0</td><td>—</td></tr>
          <tr><td>12</td><td>Energy KE, Work</td><td class="sem-cell">9.0</td><td>—</td></tr>
        </tbody>
      </table>
    </div>

    <!-- Section 3: Key Hyperparameters -->
    <div class="arch-section">
      <h2>Key Hyperparameters</h2>
      <div class="arch-params">
        <div class="arch-param-card"><div class="arch-param-key">hidden_dim</div><div class="arch-param-val">256</div></div>
        <div class="arch-param-card"><div class="arch-param-key">rollout_size</div><div class="arch-param-val">512</div></div>
        <div class="arch-param-card"><div class="arch-param-key">learning_rate</div><div class="arch-param-val">3e-4</div></div>
        <div class="arch-param-card"><div class="arch-param-key">mastery_threshold</div><div class="arch-param-val">0.85</div></div>
        <div class="arch-param-card"><div class="arch-param-key">mastery_check_interval</div><div class="arch-param-val">2,000 steps</div></div>
        <div class="arch-param-card"><div class="arch-param-key">mastery_episodes</div><div class="arch-param-val">200</div></div>
        <div class="arch-param-card"><div class="arch-param-key">entropy_start / end</div><div class="arch-param-val">0.01 → 0.001</div></div>
        <div class="arch-param-card"><div class="arch-param-key">clip_eps (PPO)</div><div class="arch-param-val">0.2</div></div>
        <div class="arch-param-card"><div class="arch-param-key">gamma</div><div class="arch-param-val">0.99</div></div>
        <div class="arch-param-card"><div class="arch-param-key">ppo_epochs</div><div class="arch-param-val">4</div></div>
      </div>
    </div>

    <!-- Section 4: Reward Function -->
    <div class="arch-section">
      <h2>Reward Function</h2>
      <div class="arch-formula">reward = exp( −|predicted − correct| × 1.5 )</div>
      <p style="color:#888;font-size:0.85rem;margin-top:10px">Smooth gradient, max 1.0 for perfect answer. Relative reward divides the error by the answer scale before applying this formula.</p>
    </div>

  </div>
</div>

<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
const TABS = ['training', 'infra', 'queue', 'registry', 'legend', 'hype', 'paper', 'notes', 'exps', 'arch'];
function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).style.display = '';
  document.getElementById('tb-' + name).classList.add('active');
  if (name === 'training')  refreshData();
  if (name === 'queue')     refreshQueue();
  if (name === 'registry')  refreshRegistry();
  if (name === 'infra')     refreshInfra();
  if (name === 'hype')  loadHype();
  if (name === 'paper') loadPaper();
  if (name === 'notes') loadNotes();
  if (name === 'exps')  { refreshExps(); if (!_expsInterval) _expsInterval = setInterval(refreshExps, 30000); }
  else { if (_expsInterval) { clearInterval(_expsInterval); _expsInterval = null; } }
}

// ── Training tab ─────────────────────────────────────────────────────────────
let charts         = {};
let _allRuns       = [];
let doneFilter     = "all";       // "all" | "active" | "done"
let smokeFilter    = "hide";      // "hide" (default) | "only" — controls ci_smoke run visibility
let interruptedFilter = "hide";  // "hide" (default) | "show" — hides runs with final_level_name=interrupted
let _selectedRuns  = new Set();   // names of selected runs; empty = show all

function totalStep(run) {
  return ((run.status && run.status.step) || 0) + (run.checkpoint_offset || 0);
}

function isDone(run) {
  if (run.queue_status === "done" || run.queue_status === "cancelled") return true;
  return run.status && totalStep(run) >= 14900000;
}

function isSmokeRun(run) {
  return (run.stem || "").includes("ci_smoke") || (run.run_tag || "").includes("ci_smoke");
}

function isInterruptedRun(run) {
  return run.final_level_name === "interrupted";
}

function filteredByDone() {
  let runs = _allRuns;
  if (doneFilter === "active") runs = runs.filter(r => !isDone(r));
  else if (doneFilter === "done") runs = runs.filter(r => isDone(r));
  if (smokeFilter === "hide") runs = runs.filter(r => !isSmokeRun(r));
  else if (smokeFilter === "only") runs = runs.filter(r => isSmokeRun(r));
  if (interruptedFilter === "hide") runs = runs.filter(r => !isInterruptedRun(r));
  return runs;
}

function toggleSmokeFilter() {
  smokeFilter = smokeFilter === "hide" ? "only" : "hide";
  const btn = document.getElementById("btn-smoke-filter");
  if (smokeFilter === "hide") {
    btn.textContent = "Hide Smoke";
    btn.classList.add("active");
  } else {
    btn.textContent = "Smoke Only";
    btn.classList.remove("active");
  }
  _selectedRuns.clear();
  document.getElementById("btn-clear-sel").style.display = "none";
  _applySelection();
}

function toggleInterruptedFilter() {
  interruptedFilter = interruptedFilter === "hide" ? "show" : "hide";
  const btn = document.getElementById("btn-interrupted-filter");
  if (interruptedFilter === "hide") {
    btn.textContent = "Hide Interrupted";
    btn.classList.add("active");
  } else {
    btn.textContent = "Show Interrupted";
    btn.classList.remove("active");
  }
  _selectedRuns.clear();
  document.getElementById("btn-clear-sel").style.display = "none";
  _applySelection();
}

function visibleRuns() {
  let runs = filteredByDone();
  if (_selectedRuns.size > 0) runs = runs.filter(r => _selectedRuns.has(r.name));
  return runs;
}

function setDoneFilter(f) {
  doneFilter = f;
  ["all", "active", "done"].forEach(id =>
    document.getElementById("seg-" + id).classList.toggle("active", id === f)
  );
  _selectedRuns.clear();
  document.getElementById("btn-clear-sel").style.display = "none";
  _applySelection();
}

function _applySelection() {
  document.getElementById("btn-clear-sel").style.display =
    _selectedRuns.size > 0 ? "inline" : "none";
  renderStatus(filteredByDone());   // cards reflect the done filter (with dim for selection)
  const runs = visibleRuns();
  updateChart("chart-mastery",       "mastery",       runs);
  updateChart("chart-mastery-stoch", "stoch_mastery", runs);
  updateChart("chart-level",         "level",         runs);
  updateChart("chart-reward",     "reward",  runs);
  updateChart("chart-entropy",    "entropy", runs);
  updateChart("chart-level-time", "level",   runs, "elapsed_min");
  updateSpeedChart("chart-speed",  runs);
  updateHealthChart("chart-health", runs);
}

function toggleRun(el) {
  const name = el.dataset.run;
  const stem = el.dataset.stem;
  const machine = el.dataset.machine;
  if (_selectedRuns.has(name)) _selectedRuns.delete(name);
  else _selectedRuns.add(name);
  if (_selectedRuns.size === 1 && stem) {
    openDetail(stem, machine);
  } else {
    closeDetail();
  }
  _applySelection();
}

function clearSelection() {
  _selectedRuns.clear();
  closeDetail();
  _applySelection();
}

let _detailChart   = null;
let _detailStem    = null;
let _detailMachine = null;

async function openDetail(stem, machine) {
  if (!stem) return;
  _detailStem    = stem;
  _detailMachine = machine || null;
  const panel = document.getElementById("run-detail");
  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  await refreshDetail();
}

function closeDetail() {
  document.getElementById("run-detail").style.display = "none";
  _detailStem    = null;
  _detailMachine = null;
  if (_detailChart) { _detailChart.destroy(); _detailChart = null; }
}

async function refreshDetail() {
  if (!_detailStem) return;
  let data;
  try {
    const url = _detailMachine
      ? "/api/run/" + encodeURIComponent(_detailMachine) + "/" + encodeURIComponent(_detailStem)
      : "/api/run/" + encodeURIComponent(_detailStem);
    const resp = await fetch(url);
    data = await resp.json();
  } catch(e) { data = {found: false}; }

  // Fallback: if no local CSV found, use data already fetched from Postgres via /api/data
  if (!data.found) {
    const run = _allRuns.find(r => r.stem === _detailStem);
    if (!run || !run.data || !run.data.length) return;
    // Build detail-compatible structure from the Postgres-backed run data
    const rows = run.data;
    const tail = rows.slice(-30);
    data = {
      found: true,
      rows: rows,
      raw_rows: tail.map(r => ({
        step:         r.step,
        level:        r.level,
        level_name:   r.level_name || ("L" + r.level),
        reward:       r.reward != null ? Number(r.reward).toFixed(4) : "",
        det_mastery:  r.mastery != null ? Number(r.mastery).toFixed(4) : "",
        stoch_mastery:r.stoch_mastery != null ? Number(r.stoch_mastery).toFixed(4) : "",
        entropy:      r.entropy != null ? Number(r.entropy).toFixed(4) : "",
        loss:         r.loss != null ? Number(r.loss).toFixed(4) : "",
        wall_time:    r.wall_ts ? new Date(r.wall_ts * 1000).toISOString().slice(0, 19) : "",
      })),
      checkpoints: [],
    };
  }

  // Header stat row
  const rows = data.rows;
  const last = rows.length ? rows[rows.length - 1] : null;
  const run = _allRuns.find(r => r.stem === _detailStem);
  const jobPrefix = run && run.job_id != null ? `#${run.job_id} — ` : "";
  const title = _detailStem.replace("training_log_", "").replace(/_/g, " ");
  document.getElementById("detail-title").textContent = jobPrefix + title;
  if (last) {
    const detOff = (run && run.checkpoint_offset) || 0;
    const stepM = ((last.step + detOff) / 1e6).toFixed(2);
    document.getElementById("detail-stats").innerHTML = `
      <div class="detail-stat"><div class="val">${stepM}M</div><div class="lbl">Step</div></div>
      <div class="detail-stat"><div class="val">L${last.level}</div><div class="lbl">Level</div></div>
      <div class="detail-stat"><div class="val">${last.reward.toFixed(3)}</div><div class="lbl">Reward</div></div>
      <div class="detail-stat"><div class="val">${last.mastery.toFixed(3)}</div><div class="lbl">Det Mastery</div></div>
      ${(last.stoch_mastery !== undefined && Math.abs(last.stoch_mastery - last.mastery) > 0.001) ? `<div class="detail-stat"><div class="val" style="color:#e15759">${last.stoch_mastery.toFixed(3)}</div><div class="lbl">Stoch Mastery</div></div>` : ''}
      <div class="detail-stat"><div class="val">${last.entropy.toFixed(4)}</div><div class="lbl">Entropy</div></div>
    `;
  }

  // Zoomed chart — reward + mastery
  if (_detailChart) { _detailChart.destroy(); _detailChart = null; }
  const ctx = document.getElementById("detail-chart").getContext("2d");
  const pts = rows;
  const isStochGate = _detailStem && _detailStem.includes("stoch");
  const hasStochDiff = pts.length > 0 && pts[0].stoch_mastery !== undefined
                       && Math.abs((pts[0].stoch_mastery||0) - (pts[0].mastery||0)) > 0.001;
  const maxStep = pts.length ? pts[pts.length-1].step : 15000000;
  const maxLevel = pts.length ? Math.max(...pts.map(r => r.level)) : 12;
  const datasets = [
    { label: "Reward",    data: pts.map(r => ({x: r.step, y: r.reward})),
      borderColor: "#4e79a7", borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
    { label: isStochGate ? "Det Mastery" : "Mastery",
      data: pts.map(r => ({x: r.step, y: r.mastery})),
      borderColor: "#59a14f", borderWidth: 1.5, pointRadius: 0, tension: 0.3,
      borderDash: [4,2] },
    { label: "Entropy\u00d710", data: pts.map(r => ({x: r.step, y: r.entropy * 10})),
      borderColor: "#e15759", borderWidth: 1, pointRadius: 0, tension: 0.2,
      borderDash: [2,3] },
    { label: "Level/" + (maxLevel||12),
      data: pts.map(r => ({x: r.step, y: r.level / (maxLevel||12)})),
      borderColor: "#00e5ff", borderWidth: 2, pointRadius: 0, tension: 0,
      borderDash: [4,3] },
    { label: "0.85",
      data: [{x: pts[0]?.step || 0, y: 0.85}, {x: maxStep, y: 0.85}],
      borderColor: "#ffdd57", borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false },
  ];
  if (isStochGate && hasStochDiff) {
    datasets.splice(2, 0, {
      label: "Stoch Mastery (gate)",
      data: pts.map(r => ({x: r.step, y: r.stoch_mastery})),
      borderColor: "#f28e2b", borderWidth: 1.5, pointRadius: 0, tension: 0.3,
      borderDash: [6,2]
    });
  }
  _detailChart = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#888", boxWidth: 20, font: {size: 10} } } },
      scales: {
        x: { type: "linear", ticks: { color: "#555",
               callback: v => v >= 1e6 ? (v/1e6).toFixed(1)+"M" : (v/1e3).toFixed(0)+"K" },
             grid: { color: "#1e2030" } },
        y: { min: 0, max: 1.05, ticks: { color: "#555" }, grid: { color: "#1e2030" } }
      }
    }
  });

  // Log table
  const logDiv = document.getElementById("detail-log");
  if (!data.raw_rows || data.raw_rows.length === 0) {
    logDiv.innerHTML = "<p style='color:#555;font-size:0.75rem'>No log data</p>";
  } else {
    const fmt = v => isNaN(parseFloat(v)) ? v : parseFloat(v).toFixed(4);
    const showStoch = isStochGate && data.raw_rows.some(r => r.stoch_mastery && r.stoch_mastery !== r.det_mastery);
    const rows_html = data.raw_rows.map(r => `
      <tr>
        <td>${Number(r.step).toLocaleString()}</td>
        <td style="text-align:left;color:#aaa">${r.level_name}</td>
        <td>${fmt(r.reward)}</td>
        <td>${fmt(r.det_mastery)}</td>
        ${showStoch ? `<td style="color:#f28e2b">${fmt(r.stoch_mastery)}</td>` : ""}
        <td>${fmt(r.entropy)}</td>
        <td>${fmt(r.loss)}</td>
        <td style="color:#555">${r.wall_time ? r.wall_time.replace("2026-","") : ""}</td>
      </tr>`).join("");
    logDiv.innerHTML = `<table class="log-table">
      <thead><tr>
        <th>Step</th><th style="text-align:left">Level</th>
        <th>Reward</th><th>Det Mastery</th>
        ${showStoch ? '<th style="color:#f28e2b">Stoch (gate)</th>' : ""}
        <th>Entropy</th><th>Loss</th><th>Wall time</th>
      </tr></thead><tbody>${rows_html}</tbody></table>`;
  }

  // Checkpoints table
  const ckDiv = document.getElementById("detail-ckpts");
  if (!data.checkpoints || data.checkpoints.length === 0) {
    ckDiv.innerHTML = "<p style='color:#555;font-size:0.75rem'>No checkpoint manifest found</p>";
  } else {
    const tagClass = t => t === "levelup" ? "ckpt-tag levelup" : t === "milestone" ? "ckpt-tag milestone" : "ckpt-tag";
    const ckRows = data.checkpoints.map(c => `
      <tr>
        <td>${Number(c.step).toLocaleString()}</td>
        <td>${c.level_name}</td>
        <td><span class="${tagClass(c.type)}">${c.type}</span></td>
        <td>${c.mastery ? parseFloat(c.mastery).toFixed(3) : ""}</td>
        <td style="color:#444">${c.wall_time ? c.wall_time.replace("2026-","") : ""}</td>
      </tr>`).join("");
    ckDiv.innerHTML = `<table class="ckpt-table">
      <thead><tr><th>Step</th><th>Level</th><th>Type</th><th>Mastery</th><th>Wall time</th></tr></thead>
      <tbody>${ckRows}</tbody></table>`;
  }
}

// Returns label with job ID prefix when available: "#23 rel anneal s7 [L]"
function runLabel(run) {
  return run.job_id != null ? `#${run.job_id} ${run.name}` : run.name;
}

function makeDataset(run, field, xField = "step") {
  const off = (xField === "step" && run.checkpoint_offset) ? run.checkpoint_offset : 0;
  return {
    label: runLabel(run),
    data: run.data.map(d => ({ x: d[xField] + off, y: d[field] })),
    borderColor: run.color,
    borderWidth: run.dash ? 1.5 : 2,
    borderDash: run.dash ? [5, 3] : [],
    pointRadius: 0,
    tension: 0.2,
    fill: false,
  };
}

// Compute steps/min from wall_ts (immune to elapsed_min resets on resume).
// Falls back to elapsed_min if wall_ts is missing.
function _computeSpeeds(data, W) {
  const pts = [];
  for (let i = W; i < data.length; i++) {
    const dStep = data[i].step - data[i - W].step;
    let dMin;
    if (data[i].wall_ts != null && data[i - W].wall_ts != null) {
      dMin = (data[i].wall_ts - data[i - W].wall_ts) / 60;  // seconds → minutes
    } else {
      dMin = data[i].elapsed_min - data[i - W].elapsed_min;
    }
    if (dMin > 0.5 && dStep > 0) {
      const spm = Math.round(dStep / dMin);
      // X = elapsed_min (consistent with other charts; wall_ts only used for dMin accuracy)
      pts.push({ x: data[i].elapsed_min, y: spm, raw: spm });
    }
  }
  return pts;
}

function makeSpeedDataset(run) {
  const W = 30;
  const pts = _computeSpeeds(run.data, W).filter(p => p.y < 30000);
  return {
    label: runLabel(run),
    data: pts,
    borderColor: run.color,
    borderWidth: run.dash ? 1.5 : 2,
    borderDash: run.dash ? [5, 3] : [],
    pointRadius: 0,
    tension: 0.3,
    fill: false,
  };
}

// Returns minutes since the last log entry for a run, using wall_ts when available.
function staleMins(run) {
  const data = run.data;
  if (!data || data.length === 0) return null;
  const last = data[data.length - 1];
  if (last.wall_ts != null) {
    return (Date.now() / 1000 - last.wall_ts) / 60;
  }
  return null;  // no wall_ts available
}

function _xScaleOpts(xField) {
  const isTime = xField === "elapsed_min";
  return {
    type: "linear",
    title: { display: true, text: isTime ? "Elapsed time" : "Steps", color: "#777" },
    ticks: { color: "#666", callback: isTime
      ? v => v >= 120 ? Math.round(v / 60) + "h" : Math.round(v) + "m"
      : v => v >= 1e6 ? (v/1e6).toFixed(1)+"M" : (v/1e3).toFixed(0)+"K" },
    grid: { color: "#222" },
  };
}

function buildChart(id, field, runs, yMin, yMax, xField = "step") {
  const ctx = document.getElementById(id).getContext("2d");
  const datasets = runs.map(r => makeDataset(r, field, xField));
  if (field === "mastery" || field === "stoch_mastery") {
    const maxX = runs.reduce((m, r) => {
      const pts = r.data || [];
      return pts.length ? Math.max(m, pts[pts.length-1][xField] || 0) : m;
    }, 0) || 15000000;
    datasets.push({
      label: "0.85 threshold",
      data: [{ x: 0, y: 0.85 }, { x: maxX, y: 0.85 }],
      borderColor: "#ffdd57", borderWidth: 1, borderDash: [4,4],
      pointRadius: 0, fill: false,
    });
  }
  charts[id] = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      animation: false,
      plugins: { legend: { labels: { color: "#ccc", font: { size: 10 },
                                      filter: i => i.datasetIndex < 20 } } },
      scales: {
        x: _xScaleOpts(xField),
        y: { min: yMin, max: yMax, ticks: { color: "#666" }, grid: { color: "#222" } },
      },
    },
  });
}

function buildSpeedChart(id, runs) {
  const ctx = document.getElementById(id).getContext("2d");
  const datasets = runs.map(r => makeSpeedDataset(r));
  charts[id] = new Chart(ctx, {
    type: "line",
    data: { datasets },
    options: {
      animation: false,
      plugins: { legend: { labels: { color: "#ccc", font: { size: 10 },
                                      filter: i => i.datasetIndex < 20 } } },
      scales: {
        x: _xScaleOpts("elapsed_min"),
        y: { min: 0, suggestedMax: 15000,
             ticks: { color: "#666",
             callback: v => v >= 1000 ? (v/1000).toFixed(1)+"K" : v },
             grid: { color: "#222" },
             title: { display: true, text: "steps / min", color: "#777" } },
      },
    },
  });
}

function updateChart(id, field, runs, xField = "step") {
  const chart = charts[id];
  const hasThreshold = field === "mastery" || field === "stoch_mastery";
  // Grab the threshold dataset before replacing (it's always the last element).
  const thresholdDs = hasThreshold
    ? chart.data.datasets[chart.data.datasets.length - 1]
    : null;
  // Replace the entire datasets array — avoids all splice index-math edge cases.
  const newDs = runs.map(run => makeDataset(run, field, xField));
  if (hasThreshold && thresholdDs) newDs.push(thresholdDs);
  chart.data.datasets = newDs;
  chart.update("none");
}

function updateSpeedChart(id, runs) {
  const chart = charts[id];
  chart.data.datasets = runs.map(r => makeSpeedDataset(r));
  chart.update("none");
}

function _healthBarData(runs) {
  // Only show jobs the coordinator considers "running" — done/queued/cancelled
  // belong in the Queue tab, not here. Goal: spot stalls, not track completion.
  const labels = [];
  const values = [];
  const colors = [];
  for (const run of runs) {
    // Only show jobs the coordinator explicitly knows are running.
    // null/undefined queue_status means the run has no job row (historical
    // import) — those would show arbitrarily stale times and clutter the chart.
    if (run.queue_status !== "running") continue;
    const m = staleMins(run);
    if (m === null) continue;
    labels.push(runLabel(run));
    const capped = Math.min(120, m);
    values.push(parseFloat(capped.toFixed(1)));
    // green <10m = actively logging, orange 10-60m = check it, red >60m = stalled
    colors.push(m < 10 ? "#59a14f" : m < 60 ? "#f28e2b" : "#e15759");
  }
  return { labels, values, colors };
}

function buildHealthChart(id, runs) {
  const ctx = document.getElementById(id).getContext("2d");
  const { labels, values, colors } = _healthBarData(runs);
  charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "min since last log", data: values,
                   backgroundColor: colors, borderWidth: 0 }],
    },
    options: {
      indexAxis: "y",
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { min: 0, suggestedMax: 30,
             title: { display: true, text: "minutes since last log entry", color: "#777" },
             ticks: { color: "#666", callback: v => v + "m" },
             grid: { color: "#222" } },
        y: { ticks: { color: "#ccc", font: { size: 9 } }, grid: { color: "#1a1a1a" } },
      },
    },
  });
}

function updateHealthChart(id, runs) {
  const chart = charts[id];
  const { labels, values, colors } = _healthBarData(runs);
  chart.data.labels = labels;
  chart.data.datasets[0].data = values;
  chart.data.datasets[0].backgroundColor = colors;
  chart.update("none");
}

function formatStep(n) {
  if (!n) return "—";
  if (n >= 1e6) return (n/1e6).toFixed(2)+"M";
  if (n >= 1e3) return (n/1e3).toFixed(0)+"K";
  return n;
}

function renderStatus(runs) {
  const grid = document.getElementById("status-grid");
  grid.innerHTML = runs.map(run => {
    const m = run.machine || (run.source === "worker" ? "worker" : "mac");
    const srcTag = run.dash
      ? `<span class="src-worker">[${m} live]</span>`
      : run.source === "worker"
        ? `<span class="src-worker">[${m}]</span>`
        : `<span class="src-local">[${m}]</span>`;
    const selClass = _selectedRuns.size === 0 ? ""
      : _selectedRuns.has(run.name) ? " selected" : " deselected";
    const jobTag = run.job_id != null
      ? `<span style="font-size:0.72rem;color:#6a9fc0;font-family:monospace;font-weight:700;margin-right:5px">#${run.job_id}</span>`
      : "";
    if (!run.status) {
      return `<div class="card${selClass}" style="border-color:${run.color};--card-color:${run.color}"
               data-run="${run.name}" data-stem="${run.stem || ''}" data-machine="${run.machine || ''}" onclick="toggleRun(this)">
        <div class="card-title">${jobTag}${run.name} ${srcTag}</div>
        <div class="no-data">No data yet</div>
      </div>`;
    }
    const s = run.status;
    const done = run.queue_status === "done" || run.queue_status === "cancelled" || totalStep(run) >= 14900000;
    const badge = s.level >= 8
      ? `<span class="badge badge-hi">L${s.level}</span>`
      : s.level > 0
        ? `<span class="badge badge-ok">L${s.level}</span>`
        : `<span class="badge badge-l0">L0</span>`;
    const doneTag = done ? `<span class="done-tag">&#10003; DONE</span>` : "";
    return `<div class="card${selClass}" style="border-color:${run.color};--card-color:${run.color}"
             data-run="${run.name}" data-stem="${run.stem || ''}" data-machine="${run.machine || ''}" onclick="toggleRun(this)">
      <div class="card-title">${jobTag}${run.name} ${badge}${doneTag} ${srcTag}</div>
      <div class="card-level">Level ${s.level}</div>
      <div class="card-stats">
        <div><div class="stat-label">Steps</div>${formatStep(totalStep(run))}</div>
        <div><div class="stat-label">${run.name.toLowerCase().includes('stoch') ? 'Adv.(stoch)' : 'Adv. Score'}</div>${(run.name.toLowerCase().includes('stoch') ? (s.stoch_mastery ?? s.mastery) : s.mastery)?.toFixed(3) ?? '—'}</div>
        <div><div class="stat-label">Reward</div>${s.reward != null ? s.reward.toFixed(3) : '—'}</div>
        <div><div class="stat-label">Time</div>${Math.round(s.elapsed_min)}m</div>
      </div>
    </div>`;
  }).join("");
}

async function refreshData() {
  try {
    const resp = await fetch("/api/data");
    _allRuns = await resp.json();
    const doneCount    = _allRuns.filter(isDone).length;
    const activeCount  = _allRuns.length - doneCount;
    document.getElementById("ts").textContent =
      "Last updated: " + new Date().toLocaleTimeString() +
      "  (" + activeCount + " active, " + doneCount + " done)";
    const cardRuns  = filteredByDone();
    const chartRuns = visibleRuns();
    renderStatus(cardRuns);
    document.getElementById("btn-clear-sel").style.display =
      _selectedRuns.size > 0 ? "inline" : "none";
    const hasCharts = Object.keys(charts).length > 0;
    if (!hasCharts) {
      buildChart("chart-mastery",       "mastery",       chartRuns, 0, 1);
      buildChart("chart-mastery-stoch", "stoch_mastery", chartRuns, 0, 1);
      buildChart("chart-level",         "level",         chartRuns, 0, null);
      buildChart("chart-reward",     "reward",  chartRuns, 0, 1);
      buildChart("chart-entropy",    "entropy", chartRuns, 0, 0.011);
      buildChart("chart-level-time", "level",   chartRuns, 0, null, "elapsed_min");
      buildSpeedChart("chart-speed",   chartRuns);
      buildHealthChart("chart-health", chartRuns);
    } else {
      updateChart("chart-mastery",       "mastery",       chartRuns);
      updateChart("chart-mastery-stoch", "stoch_mastery", chartRuns);
      updateChart("chart-level",         "level",         chartRuns);
      updateChart("chart-reward",     "reward",  chartRuns);
      updateChart("chart-entropy",    "entropy", chartRuns);
      updateChart("chart-level-time", "level",   chartRuns, "elapsed_min");
      updateSpeedChart("chart-speed",  chartRuns);
      updateHealthChart("chart-health", chartRuns);
    }
    if (_detailStem) refreshDetail();
  } catch (e) {
    document.getElementById("ts").textContent = "Error fetching data — retrying... (" + e.message + ")";
    console.error("refreshData error:", e);
  }
}

// ── Infrastructure tab ────────────────────────────────────────────────────────
const STATUS_COLORS = {online:"#59a14f",training:"#4e79a7",idle:"#888",offline:"#e15759"};

async function refreshInfra() {
  try {
    const [machines, cs] = await Promise.all([
      fetch("/api/machines").then(r=>r.json()),
      fetch("/api/coord_status").then(r=>r.json()),
    ]);
    document.getElementById("infra-ts").textContent =
      "Last updated: " + new Date().toLocaleTimeString();
    const verTag = cs.build_version ? `<span style="color:#444;font-size:0.72rem;margin-left:8px">${cs.build_version}</span>` : "";
    const coordBadge = cs.ok
      ? `<span style="color:#59a14f;font-weight:600">● coordinator online</span>${verTag} <span style="color:#555;font-size:0.75rem">${cs.error||""}</span>`
      : `<span style="color:#e15759;font-weight:600">● coordinator offline</span>${verTag} <span style="color:#e15759;font-size:0.75rem">${cs.error}</span>`;
    document.getElementById("coord-badge").innerHTML = coordBadge;

    const grid = document.getElementById("machine-grid");
    if (!machines.length) {
      grid.innerHTML = `<div style="color:#555;font-style:italic">No machines registered.</div>`;
      return;
    }
    grid.innerHTML = machines.map(m => {
      const sc   = STATUS_COLORS[m.status] || "#888";
      const bclr = m.status === "offline" ? "#e15759"
                 : m.status === "training" ? "#4e79a7"
                 : m.status === "idle"     ? "#555"
                 : "#59a14f";
      const ago = m.last_heartbeat
        ? Math.round((Date.now() - new Date(m.last_heartbeat.endsWith("Z")?m.last_heartbeat:m.last_heartbeat+"Z").getTime())/1000) + "s ago"
        : "never";
      const jobBadge = m.current_job_id
        ? `<span style="font-size:0.72rem;color:#4e79a7">job #${m.current_job_id}</span>` : "";
      return `<div class="machine-card" style="border-left-color:${bclr}">
        <div><span class="mcard-name">${m.name}</span><span class="mcard-role">${m.role||""}</span> ${jobBadge}</div>
        <div class="mcard-host">${m.hostname||"—"}</div>
        <div class="mcard-status" style="color:${sc}">${m.status||"?"}</div>
        <div class="mcard-meta">Last heartbeat: ${ago}</div>
      </div>`;
    }).join("");
  } catch(e) {
    document.getElementById("machine-grid").innerHTML =
      `<div style="color:#555;font-size:0.8rem">Error: ${e.message}</div>`;
  }
}

// ── Queue tab ─────────────────────────────────────────────────────────────────
const JOB_COLORS = {queued:"#888",assigned:"#edc948",running:"#4e79a7",done:"#59a14f",failed:"#e15759",cancelled:"#555"};
let _queueFilter = 'all';
let _queueAll    = [];
let _queueSort   = { col: "id", dir: -1 };  // default: newest first

const QUEUE_COLS = [
  { key: "id",              label: "ID" },
  { key: "run_tag",         label: "Run Tag" },
  { key: "machine",         label: "Machine" },
  { key: "phase",           label: "Phase" },
  { key: "seed",            label: "Seed" },
  { key: "steps",           label: "Steps" },
  { key: "n_envs",          label: "N Envs" },
  { key: "starting_level",  label: "Start Lvl" },
  { key: "status",          label: "Status" },
  { key: "priority",        label: "Priority" },
  { key: "created_at",      label: "Created" },
  { key: "started_at",      label: "Started" },
];

function sortQueue(col) {
  if (_queueSort.col === col) {
    _queueSort.dir *= -1;
  } else {
    _queueSort.col = col;
    // dates default descending, everything else ascending
    _queueSort.dir = (col === "created_at" || col === "started_at") ? -1 : 1;
  }
  renderQueueTable();
}

function setQueueFilter(f) {
  _queueFilter = f;
  ['all','active','done','failed'].forEach(x => {
    document.getElementById('qf-'+x).classList.toggle('active', x===f);
  });
  renderQueueTable();
}

function renderQueueTable() {
  const { col, dir } = _queueSort;
  let rows = [..._queueAll].sort((a, b) => {
    let av = a[col], bv = b[col];
    if (av === null || av === undefined) av = typeof av === "number" ? -Infinity : "";
    if (bv === null || bv === undefined) bv = typeof bv === "number" ? -Infinity : "";
    if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
    return String(av).localeCompare(String(bv)) * dir;
  });
  if (_queueFilter === 'active')  rows = rows.filter(j => ['queued','assigned','running'].includes(j.status));
  if (_queueFilter === 'done')    rows = rows.filter(j => j.status === 'done');
  if (_queueFilter === 'failed')  rows = rows.filter(j => j.status === 'failed');
  const wrap = document.getElementById("queue-table-wrap");
  if (!rows.length) {
    wrap.innerHTML = `<div style="color:#555;font-style:italic">No jobs match this filter.</div>`;
    return;
  }
  const headers = QUEUE_COLS.map(c => {
    const isActive = _queueSort.col === c.key;
    const cls = `sortable${isActive ? (_queueSort.dir === 1 ? " sort-asc" : " sort-desc") : ""}`;
    return `<th class="${cls}" onclick="sortQueue('${c.key}')">${c.label}<span class="sort-ind"></span></th>`;
  }).join("") + `<th>Actions</th>`;
  wrap.innerHTML = `<table class="data-table">
    <thead><tr>${headers}</tr></thead>
    <tbody>${rows.map(j => {
      const sc = j.status in JOB_COLORS ? `sb-${j.status}` : "sb-queued";
      // For resumed jobs, add the checkpoint start step so total reflects full run
      let totalSteps = j.steps;
      let resumeNote = "";
      try {
        const flags = typeof j.flags === "string" ? JSON.parse(j.flags) : (j.flags || {});
        const ckpt = flags.checkpoint || "";
        const m = ckpt.match(/agent_step_(\d+)\.pt/);
        if (m) {
          const ckptStep = parseInt(m[1]);
          totalSteps = ckptStep + j.steps;
          resumeNote = `<span style="color:#555;font-size:0.68rem"> +resume</span>`;
        }
      } catch(e) {}
      const steps = totalSteps >= 1e6 ? (totalSteps/1e6).toFixed(1)+"M" : (totalSteps/1e3).toFixed(0)+"K";
      const startLvl = j.starting_level != null
        ? `<span style="color:#e8b84b;font-weight:700">L${j.starting_level}</span>`
        : `<span style="color:#444">0</span>`;
      // Action buttons
      let actions = "—";
      if (j.status === "failed") {
        actions = `<button class="filter-btn" style="font-size:0.7rem;padding:2px 7px;margin-right:4px"
             onclick="requeueJob(${j.id},'resume')" title="Re-run from last checkpoint">&#8635; Resume</button>
           <button class="filter-btn" style="font-size:0.7rem;padding:2px 7px"
             onclick="requeueJob(${j.id},'fresh')" title="Start over from scratch (deletes checkpoints)">&#8635; Fresh</button>`;
      } else if (j.status === "cancelled") {
        actions = `<button class="filter-btn" style="font-size:0.7rem;padding:2px 7px"
             onclick="requeueJob(${j.id},'fresh')" title="Start over from scratch (deletes checkpoints)">&#8635; Fresh</button>`;
      } else if (['queued','assigned','running'].includes(j.status)) {
        actions = `<button class="filter-btn" style="font-size:0.7rem;padding:2px 7px;color:#e15759"
             onclick="cancelJob(${j.id})" title="Cancel this job">&#10005; Cancel</button>`;
      }
      return `<tr>
        <td style="color:#666">#${j.id}</td>
        <td style="color:#ccc;font-weight:600">${j.run_tag||"—"}</td>
        <td style="color:#888">${j.machine||"any"}</td>
        <td>${j.phase||"—"}</td>
        <td>${j.seed}</td>
        <td>${steps}${resumeNote}</td>
        <td>${j.n_envs||"—"}</td>
        <td>${startLvl}</td>
        <td><span class="sbadge ${sc}">${j.status}</span></td>
        <td>${j.priority}</td>
        <td style="color:#555;font-size:0.72rem">${(j.created_at||"").replace("T"," ").replace("Z","")}</td>
        <td style="color:#555;font-size:0.72rem">${(j.started_at||"—").replace("T"," ").replace("Z","")}</td>
        <td style="white-space:nowrap">${actions}</td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

async function refreshQueue() {
  try {
    const jobs = await fetch("/api/queue").then(r=>r.json());
    _queueAll = jobs;
    document.getElementById("queue-ts").textContent =
      "Last updated: " + new Date().toLocaleTimeString() +
      " — " + jobs.length + " total jobs";
    renderQueueTable();
  } catch(e) {
    document.getElementById("queue-table-wrap").innerHTML =
      `<div style="color:#555;font-size:0.8rem">Error: ${e.message}</div>`;
  }
}

async function cancelJob(jobId) {
  if (!confirm(`Cancel job #${jobId}? The worker will finish its current step and stop.`)) return;
  try {
    const resp = await fetch(`/api/queue/${jobId}`, { method: "DELETE" });
    if (!resp.ok) { alert(`Cancel failed: ${await resp.text()}`); return; }
    await refreshQueue();
  } catch(e) {
    alert(`Cancel error: ${e.message}`);
  }
}

async function requeueJob(jobId, mode) {
  const label = mode === "fresh" ? "fresh start" : "resume from checkpoint";
  if (!confirm(`Requeue job #${jobId} as a ${label}?`)) return;
  try {
    const resp = await fetch(`/api/queue/${jobId}/requeue`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode}),
    });
    if (!resp.ok) {
      const err = await resp.text();
      alert(`Requeue failed: ${err}`);
      return;
    }
    const newJob = await resp.json();
    alert(`Job #${jobId} requeued as job #${newJob.id} (${label}).`);
    await refreshQueue();
  } catch(e) {
    alert(`Requeue error: ${e.message}`);
  }
}

// ── Registry tab ─────────────────────────────────────────────────────────────
function fmtCurriculum(c) {
  if (!c) return "—";
  // "13L" → "13-lvl", "17L" → "17-lvl", etc.; unknown values pass through
  return c.replace(/^(\\d+)L(-.*)?$/, (_, n, suffix) => n + "-lvl" + (suffix || ""));
}
let _regData = [];
let _regSort = { col: "started_at", dir: -1 };  // -1 = desc, 1 = asc

const REG_COLS = [
  { key: "run_id",           label: "Run ID" },
  { key: "seed",             label: "Seed" },
  { key: "curriculum",       label: "Curriculum" },
  { key: "reward_mode",      label: "Reward" },
  { key: "entropy_mode",     label: "Entropy" },
  { key: "final_level_name", label: "Final Level" },
  { key: "final_sem_level",  label: "Sem Level" },
  { key: "started_at",       label: "Started" },
  { key: "completed_at",     label: "Completed" },
];

function sortRegistry(col) {
  if (_regSort.col === col) {
    _regSort.dir *= -1;
  } else {
    _regSort.col = col;
    _regSort.dir = col === "started_at" || col === "completed_at" ? -1 : 1;
  }
  renderRegistry();
}

function renderRegistry() {
  const wrap = document.getElementById("registry-table-wrap");
  if (!_regData.length) {
    wrap.innerHTML = `<div style="color:#555;font-style:italic">No experiments recorded yet.</div>`;
    return;
  }
  const { col, dir } = _regSort;
  const sorted = [..._regData].sort((a, b) => {
    let av = a[col], bv = b[col];
    if (av === null || av === undefined) av = col === "final_sem_level" ? -1 : "";
    if (bv === null || bv === undefined) bv = col === "final_sem_level" ? -1 : "";
    if (typeof av === "number") return (av - bv) * dir;
    return String(av).localeCompare(String(bv)) * dir;
  });
  const headers = REG_COLS.map(c => {
    const isActive = _regSort.col === c.key;
    const cls = `sortable${isActive ? (_regSort.dir === 1 ? " sort-asc" : " sort-desc") : ""}`;
    return `<th class="${cls}" onclick="sortRegistry('${c.key}')">${c.label}<span class="sort-ind"></span></th>`;
  }).join("");
  wrap.innerHTML = `<table class="data-table">
    <thead><tr>${headers}</tr></thead>
    <tbody>${sorted.map(e => {
      const sl = e.final_sem_level;
      const slCls = sl === null || sl === undefined ? "" :
                    sl < 4 ? "sl-low" : sl < 6 ? "sl-mid" : sl < 8 ? "sl-high" : "sl-top";
      const slTxt = sl !== null && sl !== undefined ? sl.toFixed(1) : "—";
      const isImported = !!e.imported_from;
      const fmtTs = ts => ts ? ts.replace("T"," ").replace("Z","").slice(0,16) : null;
      const startedTxt = e.started_at
        ? `<span style="color:#555;font-size:0.72rem">${fmtTs(e.started_at)}</span>`
        : `<span style="color:#444;font-size:0.72rem;font-style:italic">unknown</span>`;
      const completedTxt = e.completed_at
        ? `<span style="font-size:0.72rem;color:#555">${fmtTs(e.completed_at)}</span>`
        : isImported
          ? `<span style="color:#444;font-size:0.72rem;font-style:italic">unknown</span>`
          : `<span style="color:#4e79a7;font-size:0.72rem">running</span>`;
      return `<tr>
        <td style="font-family:monospace;color:#ccc;font-size:0.75rem">${e.run_id||"—"}</td>
        <td>${e.seed??'—'}</td>
        <td>${fmtCurriculum(e.curriculum)}</td>
        <td>${e.reward_mode||"—"}</td>
        <td>${e.entropy_mode||"—"}</td>
        <td style="color:#bbb">${e.final_level_name||"—"}</td>
        <td class="${slCls}" style="font-family:monospace;font-weight:700">${slTxt}</td>
        <td>${startedTxt}</td>
        <td>${completedTxt}</td>
      </tr>`;
    }).join("")}</tbody>
  </table>`;
}

async function refreshRegistry() {
  try {
    const exps = await fetch("/api/experiments").then(r=>r.json());
    _regData = exps;
    document.getElementById("reg-ts").textContent =
      "Last updated: " + new Date().toLocaleTimeString() + " — " + exps.length + " experiments";
    renderRegistry();
  } catch(e) {
    document.getElementById("registry-table-wrap").innerHTML =
      `<div style="color:#555;font-size:0.8rem">Error: ${e.message}</div>`;
  }
}

// ── Hype tab ──────────────────────────────────────────────────────────────────
let _hypeLoaded = false;
async function loadHype() {
  if (_hypeLoaded) return;
  const resp = await fetch("/api/markdown?file=hype");
  const data = await resp.json();
  // Hide the tab button entirely when the file isn't present (public deployments).
  if (!data.content) { document.getElementById("tb-hype").style.display = "none"; return; }
  const html = marked.parse(data.content);
  const el = document.getElementById("hype-body");
  el.innerHTML = html;
  el.classList.add("md-content");
  _hypeLoaded = true;
}

// ── Paper tab ─────────────────────────────────────────────────────────────────
let _paperLoaded = false;
async function loadPaper() {
  if (_paperLoaded) return;
  const resp = await fetch("/api/markdown?file=paper");
  const data = await resp.json();
  if (!data.content) { document.getElementById("tb-paper").style.display = "none"; return; }
  const html = marked.parse(data.content);
  const body = document.getElementById("paper-body");
  body.innerHTML = html;
  body.classList.add("md-content");

  // Build section nav from headings
  const nav = document.getElementById("paper-nav");
  const headings = body.querySelectorAll("h1, h2");
  headings.forEach((h, i) => {
    const anchor = "sec-" + i;
    h.id = anchor;
    const a = document.createElement("a");
    a.href = "#" + anchor;
    a.textContent = h.textContent;
    a.className = h.tagName === "H1" ? "h1-link" : "h2-link";
    a.onclick = e => { e.preventDefault(); h.scrollIntoView({behavior:"smooth",block:"start"}); };
    nav.appendChild(a);
  });
  _paperLoaded = true;
}

// ── Notes tab ─────────────────────────────────────────────────────────────────
let _notesLoaded = false;
async function loadNotes() {
  if (_notesLoaded) return;
  const resp = await fetch("/api/markdown?file=notes");
  const data = await resp.json();
  if (!data.content) { document.getElementById("tb-notes").style.display = "none"; return; }
  const html = marked.parse(data.content);
  document.getElementById("notes-body").innerHTML = html;
  _notesLoaded = true;
}

// ── Experiments tab (grouped by run_tag) ──────────────────────────────────────
const RUN_TAG_DESC = {
  'abs_anneal':      'Baseline — absolute reward, annealing entropy (original config)',
  'rel_reset':       'Finding 2 — relative reward + entropy reset (main result)',
  'abs_reset':       'Ablation — absolute reward + entropy reset (isolates entropy effect)',
  'rel_fixed':       'Ablation — relative reward + fixed entropy (isolates reward effect)',
  'rel_anneal':      'Ablation — relative reward + annealing entropy',
  'stoch_gate':      'Finding 1 — stochastic mastery gate (original bug demonstration)',
  'cold_L2':         'Finding 3 seed — cold start from L2 Subtraction',
  'cold_L4':         'Finding 3 seed — cold start from L4 Multiplication',
  'cold_L6':         'Finding 3 seed — cold start from L6 Division',
  'cold_L8':         'Finding 3 seed — cold start from L8 Mixed Arithmetic',
  'no_concept_disc': 'Finding 6 — no concept-discovery warmup levels (11L curriculum)',
  'refined_17L':     'Finding 5 — refined 17L curriculum with reset entropy',
  'energy_30M':      'Finding 3 — 30M step runs to test Energy barrier',
  'fresh_l12':       'Finding 4 — resume from L12 Energy checkpoint',
};

let _expsInterval = null;

function _flagPills(flags) {
  if (!flags) return '';
  let f = flags;
  if (typeof f === 'string') {
    try { f = JSON.parse(f); } catch(e) { return ''; }
  }
  const pills = [];
  if (f.relative_reward === true)    pills.push('<span class="flag-pill fp-rel">rel</span>');
  if (f.entropy_mode === 'reset')    pills.push('<span class="flag-pill fp-reset">reset</span>');
  else if (f.entropy_mode === 'anneal') pills.push('<span class="flag-pill fp-anneal">anneal</span>');
  else if (f.entropy_mode === 'fixed')  pills.push('<span class="flag-pill fp-fixed">fixed</span>');
  if (f.no_concept_discovery === true) pills.push('<span class="flag-pill fp-ncd">ncd</span>');
  if (f.refined_mul === true && f.refined_div === true) pills.push('<span class="flag-pill fp-17L">17L</span>');
  return pills.join('');
}

function _statusBadge(st) {
  const sc = {queued:'sb-queued',assigned:'sb-assigned',running:'sb-running',done:'sb-done',failed:'sb-failed',cancelled:'sb-cancelled'}[st] || 'sb-queued';
  return `<span class="sbadge ${sc}">${st}</span>`;
}

function _countBadges(jobs) {
  const counts = {done:0, running:0, queued:0, failed:0};
  for (const j of jobs) {
    if (j.status === 'done')                            counts.done++;
    else if (j.status === 'running' || j.status === 'assigned') counts.running++;
    else if (j.status === 'queued')                     counts.queued++;
    else if (j.status === 'failed' || j.status === 'cancelled') counts.failed++;
  }
  const parts = [];
  if (counts.done)    parts.push(`<span style="color:#59a14f">${counts.done} done</span>`);
  if (counts.running) parts.push(`<span style="color:#4e79a7">${counts.running} running</span>`);
  if (counts.queued)  parts.push(`<span style="color:#888">${counts.queued} queued</span>`);
  if (counts.failed)  parts.push(`<span style="color:#e15759">${counts.failed} failed</span>`);
  return parts.join(' · ');
}

async function refreshExps() {
  const body = document.getElementById("exps-body");
  body.innerHTML = `<div style="color:#555;font-style:italic">Loading...</div>`;
  try {
    const [jobs, runs] = await Promise.all([
      fetch("/api/queue").then(r => r.json()),
      fetch("/api/data").then(r => r.json()),
    ]);

    // Build level lookup: (run_tag, seed) → {level, level_name, step}
    const levelMap = new Map();
    for (const run of runs) {
      const m = run.stem.match(/seed(\d+)_(.+)$/);
      if (!m) continue;
      const seed = parseInt(m[1]), tag = m[2];
      const pts = run.data || [];
      if (!pts.length) continue;
      const last = pts[pts.length - 1];
      levelMap.set(`${tag}|${seed}`, { level: last.level, step: last.step });
    }

    // Group by run_tag
    const groups = new Map();
    for (const job of jobs) {
      const tag = job.run_tag || '(unknown)';
      if (!groups.has(tag)) groups.set(tag, []);
      groups.get(tag).push(job);
    }

    if (groups.size === 0) {
      body.innerHTML = `<div style="color:#555;font-style:italic">No jobs found.</div>`;
      return;
    }

    const LEVEL_NAMES = ['Add 0-10','Add 0-100','Sub 0-10','Sub 0-100','Mul 0-5','Mul 0-12',
                         'Div ÷1-5','Div ÷1-12','Mixed','Linear','Quadratic','Kinematics','Energy'];

    let html = '';
    for (const [tag, tagJobs] of groups) {
      const desc = RUN_TAG_DESC[tag] || '';
      const countHtml = _countBadges(tagJobs);
      html += `<div style="margin-bottom:2px">
        <div class="exp-group-header">
          <span>${tag}${desc ? `<span class="exp-desc">${desc}</span>` : ''}</span>
          <span class="exp-badge-counts">${countHtml}</span>
        </div>
        <table class="data-table" style="font-size:0.78rem;margin-top:0;border-top:none">
          <thead><tr><th>ID</th><th>Seed</th><th>Steps (M)</th><th>Status</th><th>Level</th><th>Flags</th></tr></thead>
          <tbody>${tagJobs.map(j => {
            const stepsM = j.steps >= 1e6 ? (j.steps/1e6).toFixed(1)+'M' : (j.steps >= 1e3 ? (j.steps/1e3).toFixed(0)+'K' : (j.steps||'—'));
            const lvlInfo = levelMap.get(`${j.run_tag}|${j.seed}`);
            const lvlHtml = lvlInfo != null
              ? `<span style="color:#00e5ff;font-weight:600">L${lvlInfo.level}</span> <span style="color:#555;font-size:0.7rem">${LEVEL_NAMES[lvlInfo.level]||''}</span>`
              : `<span style="color:#333">—</span>`;
            return `<tr>
              <td style="color:#666">#${j.id}</td>
              <td>${j.seed??'—'}</td>
              <td style="font-family:monospace">${stepsM}</td>
              <td>${_statusBadge(j.status)}</td>
              <td>${lvlHtml}</td>
              <td>${_flagPills(j.flags)}</td>
            </tr>`;
          }).join('')}</tbody>
        </table>
      </div>`;
    }
    body.innerHTML = html;
  } catch(e) {
    body.innerHTML = `<div style="color:#555;font-size:0.8rem">Error: ${e.message}</div>`;
  }
}

// ── Boot ─────────────────────────────────────────────────────────────────────
refreshData();
refreshInfra();
refreshQueue();
refreshRegistry();
setInterval(refreshData,     30000);
setInterval(refreshQueue,    30000);
setInterval(refreshRegistry, 60000);
setInterval(refreshInfra,    60000);

// Probe markdown availability on load so tabs hide immediately if files are
// absent (e.g. public deployment without paper/ mounted).
loadHype();
loadPaper();
loadNotes();
</script>
</body>
</html>
"""

# ── HTTP server ───────────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({
                "status":        "ok",
                "build_version": BUILD_VERSION,
                "service":       "dashboard",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/data":
            body = json.dumps(get_all_data()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/worker_status":
            body = json.dumps(_worker_status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/refresh_workers":
            threading.Thread(target=_refresh_workers, daemon=True).start()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Worker refresh triggered")
        elif self.path == "/api/queue":
            with _coord_lock:
                body = json.dumps(_queue_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/machines":
            with _coord_lock:
                body = json.dumps(_machines_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/coord_status":
            body = json.dumps(_coord_status).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/experiments":
            with _coord_lock:
                body = json.dumps(_experiments_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/run/"):
            rest = self.path[len("/api/run/"):]
            # Support both /api/run/<stem> and /api/run/<machine>/<stem>
            parts = rest.split("/", 1)
            if len(parts) == 2:
                machine, stem = parts[0], parts[1]
            else:
                machine, stem = None, parts[0]
            body = json.dumps(get_run_detail(stem, machine)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/markdown"):
            import urllib.parse as _urlparse
            qs = _urlparse.parse_qs(_urlparse.urlparse(self.path).query)
            file_key = (qs.get("file") or [""])[0]
            _base = Path(__file__).parent.parent / "paper"
            _file_map = {
                "hype": None,   # concatenation, handled below
                "paper": _base / "paper.md",
            }
            content = ""
            if file_key == "hype":
                pitch_path = _base / "ELEVATOR_PITCH_v2.md"
                narrative_path = _base / "LEARNING_WITHOUT_SURVIVAL_HEURISTICS_v2.md"
                parts = []
                for p in (pitch_path, narrative_path):
                    try:
                        parts.append(p.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                content = "\n\n---\n\n".join(parts)
            elif file_key == "paper":
                paper_path = _base / "paper.md"
                try:
                    content = paper_path.read_text(encoding="utf-8")
                except Exception:
                    content = ""
            elif file_key == "notes":
                notes_path = _base / "RESEARCH_NOTES.md"
                try:
                    content = notes_path.read_text(encoding="utf-8")
                except Exception:
                    content = ""
            body = json.dumps({"content": content}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

    def do_DELETE(self):
        import re as _re
        import urllib.request as _ureq
        import urllib.error  as _uerr

        m = _re.match(r"^/api/queue/(\d+)$", self.path)
        if m:
            coord_url = f"{COORDINATOR_URL}/api/queue/{m.group(1)}"
            try:
                req = _ureq.Request(coord_url, method="DELETE")
                with _ureq.urlopen(req, timeout=10) as resp:
                    status = resp.status
                    body   = resp.read()
            except _uerr.HTTPError as exc:
                status = exc.code
                body   = exc.read()
            except Exception as exc:
                status = 502
                body   = json.dumps({"error": str(exc)}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        import re as _re
        import urllib.request as _ureq
        import urllib.error  as _uerr

        # Proxy requeue requests to the coordinator so the browser doesn't need
        # to know the coordinator's port.
        m = _re.match(r"^/api/queue/(\d+)/requeue$", self.path)
        if m:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length else b"{}"
            coord_url = f"{COORDINATOR_URL}/api/queue/{m.group(1)}/requeue"
            try:
                req = _ureq.Request(
                    coord_url,
                    data=raw_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with _ureq.urlopen(req, timeout=10) as resp:
                    status = resp.status
                    body   = resp.read()
            except _uerr.HTTPError as exc:
                status = exc.code
                body   = exc.read()
            except Exception as exc:
                status = 502
                body   = json.dumps({"error": str(exc)}).encode()

            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    print(f"Dashboard running → http://localhost:{PORT}")
    print(f"Local logs: {LOGS_DIR}")
    print(f"Worker URL: {WORKER_URL}  (refresh every {WORKER_REFRESH}s)")
    print("Ctrl+C to stop")
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()
