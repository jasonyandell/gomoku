#!/usr/bin/env python3
"""Delo Derby — a standalone autonomous race scheduler for training recipes.

A race of "ideas" (fresh-start self-play training recipes, each differing from a
baseline by ONE lever). Ideas are raced to a 140-epoch budget in 10-epoch chunks,
scored by the model's anchored elo (`eval/model_elo`). Compute is allocated by a
Delta-elo-rate priority queue: at each step the next chunk goes to the not-yet-capped
idea whose most recent chunk gained the most elo. Everyone reaches the cap (no
elimination) — leaders just get there first, so the standings always show the best
available model.

Designed to run UNATTENDED OVERNIGHT: robustness > features. One bad chunk never
kills the race; state is written atomically after every chunk and the loop is fully
crash-resumable via --resume.

TWO CHUNK ENGINES (selected by board `global.engine`):

  "single_process" (v1, DEFAULT — unchanged):
    `gomoku.train` ALONE does NOT write `eval/model_elo` to eval_results.jsonl. With
    in-trainer eval it logs only per-baseline winrates; the maximum-likelihood
    `model_elo` + the jsonl line are produced by a SEPARATE `gomoku.eval_worker`
    process. So each v1 chunk is TWO sequential subprocess steps:
      1. `python -m gomoku.train --no-eval --epochs <chunk> [--resume latest.pt]
         --save-buffer-every <chunk> ...`  → advances the model, writes latest.pt.
      2. `python -m gomoku.eval_worker --checkpoint-path <dir>/latest.pt
         --max-cycles 1 ...`  → appends ONE eval_results.jsonl line carrying model_elo.
    `--save-buffer-every <chunk_epochs>` is REQUIRED: train.py only rewrites latest.pt
    every save_buffer_every epochs (default 20), so a 10-epoch chunk would otherwise
    leave NO fresh latest.pt to resume from. Budget is EPOCH-native (cap_epochs).

  "run_sweep_wall_slice" (v2 — production multiprocess, WALL-NATIVE):
    v1 ran single-process (one `gomoku.train`; GPU ~30%, machine mostly idle), so its
    wall-clock / Δelo-per-hour were unrepresentative of production. v2 races on the
    PRODUCTION recipe — `run_sweep.py` runs 1 trainer + N selfplay_workers in wave-mode
    and SATURATES the GPU — so Δelo/hr is finally honest. Each v2 chunk is ONE command:
      `python scripts/run_sweep.py --cell <CELL> --resume <idea>/checkpoints/latest.pt
       --max-wall-secs <slice_secs> --final-eval`
    That single command runs the bundle for the slice, self-caps with a clean resumable
    save (latest.pt with buffer embedded), AND ends with a fresh `eval/model_elo` line
    in `<cell>/checkpoints/eval_results.jsonl` (run_sweep --final-eval does eval INSIDE
    the bundle — no separate eval_worker step). Budget is WALL-NATIVE: a chunk = a wall
    slice; the per-idea cap is wall-seconds (cap_wall_secs), NOT epochs.

Both engines share: never-run-first then Δelo/hour "hill-climb" priority, atomic state writes, one
failed chunk never kills the race, crash-resume via --resume, peak-checkpoint
snapshotting (best-elo latest.pt copied to a peak path so keep-last-n pruning can't
discard the champion).

Usage:
  python scripts/delo_derby.py --board scripts/derby_board.json            # v1 (default)
  python scripts/delo_derby.py --board scripts/derby_v2_board.json         # v2 wall-slice
  python scripts/delo_derby.py --board scripts/derby_v2_board.json --resume
  python scripts/delo_derby.py --board scripts/derby_v2_board.json --dry-run
  python scripts/delo_derby.py --max-chunks 4 --chunk-timeout 1800
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
STANDINGS_SENTINEL = "<!-- STANDINGS:AUTO — delo_derby.py rewrites everything below this line -->"
RESEARCH_BOARD_REL = "wiki/ops/research-board.md"

# The eval_worker score key, verified against a real WL5 eval_results.jsonl line:
#   {"ts": ..., "eval_worker/epoch_evaluated": 10200, "eval/vs_random_winrate": 1.0,
#    ..., "eval/model_elo": 1592.4224853515625}
ELO_KEY = "eval/model_elo"
EPOCH_KEY = "eval_worker/epoch_evaluated"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_line(log_path: Path, msg: str) -> None:
    """Append a timestamped line to a log file (best-effort; never raises)."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"{now_iso()} {msg}\n")
    except OSError:
        pass


def derby_print(msg: str) -> None:
    print(msg, flush=True)


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically via tmp+rename so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def resolve_path(p: str | Path) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(p)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Board / state model
# ---------------------------------------------------------------------------

SINGLE_PROCESS = "single_process"
RUN_SWEEP_WALL_SLICE = "run_sweep_wall_slice"


def engine_of(board: dict) -> str:
    """The board's chunk engine. Absent => v1 single_process (backward compat)."""
    return board["global"].get("engine", SINGLE_PROCESS)


def load_board(board_path: Path) -> dict:
    with open(board_path) as f:
        board = json.load(f)
    if "global" not in board or "ideas" not in board:
        raise ValueError(f"board {board_path} missing 'global' or 'ideas'")
    g = board["global"]
    engine = g.get("engine", SINGLE_PROCESS)
    if engine == SINGLE_PROCESS:
        req_keys = ("chunk_epochs", "cap_epochs", "size", "seed", "base_out_dir", "device")
    elif engine == RUN_SWEEP_WALL_SLICE:
        # WALL-native: slice_secs (per chunk) + cap_wall_secs (per idea). No epochs.
        req_keys = ("slice_secs", "cap_wall_secs", "base_out_dir")
    else:
        raise ValueError(f"board global.engine={engine!r} unknown "
                         f"(expected {SINGLE_PROCESS!r} or {RUN_SWEEP_WALL_SLICE!r})")
    for req in req_keys:
        if req not in g:
            raise ValueError(f"board global (engine={engine}) missing required key: {req!r}")
    names = [idea["name"] for idea in board["ideas"]]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate idea names in board: {names}")
    if engine == RUN_SWEEP_WALL_SLICE:
        for idea in board["ideas"]:
            for req in ("cell", "cell_name"):
                if req not in idea:
                    raise ValueError(
                        f"run_sweep idea {idea.get('name')!r} missing required key "
                        f"{req!r} ('cell' = run_sweep.CELLS key; 'cell_name' = that "
                        f"Cell's .name, which fixes the sweep_runs/<cell_name>/ path)")
    return board


def fresh_state(board: dict) -> dict:
    return {
        "board_path": None,  # filled by caller
        "round0_done": False,
        "created": now_iso(),
        "updated": now_iso(),
        "total_chunks_run": 0,
        "ideas": {
            idea["name"]: {
                "name": idea["name"],
                "epochs_done": 0,            # v1: epoch progress (v2: best-effort epoch tag)
                "elo_history": [],          # list of [epoch_or_chunk, elo, chunk_wall_secs]
                "last_delo": 0.0,
                "wall_secs_total": 0.0,      # v2 budget axis (also tracked by v1)
                "chunks_done": 0,            # v2: # wall slices completed
                "peak_elo": None,            # best elo ever seen (peak-checkpoint tracking)
                "peak_path": None,           # snapshot path of the best-elo latest.pt
                "status": "queued",          # queued | running | capped | errored
                "wandb_run_id": None,
                "retries": 0,
            }
            for idea in board["ideas"]
        },
    }


def state_path_for(board: dict) -> Path:
    base = resolve_path(board["global"]["base_out_dir"])
    return base / "derby_state.json"


def board_md_path_for(board: dict) -> Path:
    """Where standings are written. Default (v1) = the shared research-board.md.
    A board may override via global.board_md_path (relative to repo root) — the v2
    board points at a DERBY-OWNED standings file in base_out_dir so it never clobbers
    the wiki page another agent owns."""
    override = board["global"].get("board_md_path")
    if override:
        return resolve_path(override)
    return resolve_path(RESEARCH_BOARD_REL)


def load_state(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Command composition
# ---------------------------------------------------------------------------

def idea_checkpoint_dir(board: dict, idea_name: str) -> Path:
    """Checkpoint dir for an idea.

    v1 (single_process): <base_out_dir>/<idea_name>/checkpoints — derby owns the dir.
    v2 (run_sweep_wall_slice): run_sweep OWNS the layout. It always writes to
    REPO_ROOT/sweep_runs/<cell_name>/checkpoints (see run_sweep.cell_dirs); the derby
    must read latest.pt / eval_results.jsonl FROM THERE, not from base_out_dir. We
    locate it via the idea's `cell_name` (the run_sweep Cell.name).
    """
    if engine_of(board) == RUN_SWEEP_WALL_SLICE:
        idea = next(i for i in board["ideas"] if i["name"] == idea_name)
        return REPO_ROOT / "sweep_runs" / idea["cell_name"] / "checkpoints"
    base = resolve_path(board["global"]["base_out_dir"])
    return base / idea_name / "checkpoints"


def idea_peak_dir(board: dict, idea_name: str) -> Path:
    """Derby-owned dir for peak-checkpoint snapshots (survives keep-last-n pruning)."""
    base = resolve_path(board["global"]["base_out_dir"])
    return base / "_peaks" / idea_name


def idea_chunk_log(board: dict, idea_name: str) -> Path:
    base = resolve_path(board["global"]["base_out_dir"])
    return base / idea_name / "chunk.log"


def build_train_cmd(board: dict, idea: dict, resume: bool) -> list[str]:
    """Compose the trainer chunk command.

    common_args + idea.args + global eval/size/seed/device/wandb flags +
    checkpoint-dir + run-name. Always `--epochs <chunk_epochs>` (an INCREMENT).
    `--no-eval`: the trainer never writes model_elo; we run eval_worker after.
    """
    g = board["global"]
    chunk_epochs = int(g["chunk_epochs"])
    ckpt_dir = idea_checkpoint_dir(board, idea["name"])

    cmd: list[str] = [PYTHON, "-u", "-m", "gomoku.train"]
    cmd += ["--size", str(g["size"])]
    cmd += ["--seed", str(g["seed"])]
    cmd += ["--device", str(g["device"])]
    cmd += ["--epochs", str(chunk_epochs)]
    cmd += ["--checkpoint-dir", str(ckpt_dir)]
    cmd += ["--run-name", f"derby-{idea['name']}"]
    # Eval is done out-of-process by eval_worker (in-trainer eval lacks model_elo).
    cmd += ["--no-eval"]
    # latest.pt must be rewritten at the chunk boundary so the next chunk can
    # resume from it (train.py default save_buffer_every=20 would skip a 10-ep chunk).
    cmd += ["--save-buffer-every", str(chunk_epochs)]
    cmd += ["--save-every", "1"]
    # wandb
    if g.get("wandb", False):
        cmd += ["--wandb"]
        if g.get("wandb_project"):
            cmd += ["--wandb-project", str(g["wandb_project"])]
    else:
        cmd += ["--no-wandb"]
    # shared training hyperparameters
    cmd += [str(x) for x in g.get("common_args", [])]
    # the idea's one-lever overrides (later args win in argparse)
    cmd += [str(x) for x in idea.get("args", [])]
    if resume:
        latest = ckpt_dir / "latest.pt"
        cmd += ["--resume", str(latest)]
    return cmd


def build_eval_cmd(board: dict, idea: dict) -> list[str]:
    """Compose the single-pass eval command (writes one eval_results.jsonl line)."""
    g = board["global"]
    ckpt_dir = idea_checkpoint_dir(board, idea["name"])
    latest = ckpt_dir / "latest.pt"

    baselines = g.get("eval_baselines", "random,heuristic,lookahead:depth=2")
    slow = g.get("eval_baselines_slow", "")
    # Fold the slow baseline(s) into the single eval pass so model_elo is anchored
    # against the strongest baseline too (eval_worker does one pass over all of them).
    all_baselines = baselines
    if slow:
        all_baselines = f"{baselines},{slow}"

    cmd: list[str] = [PYTHON, "-u", "-m", "gomoku.eval_worker"]
    cmd += ["--checkpoint-path", str(latest)]
    cmd += ["--output-dir", str(ckpt_dir)]
    cmd += ["--baselines", all_baselines]
    cmd += ["--n-games", str(g.get("eval_baseline_games", 20))]
    cmd += ["--sims", str(g.get("eval_sims", 100))]
    cmd += ["--device", "cpu"]            # keep eval off MPS so it doesn't fight a future trainer
    cmd += ["--n-workers", "4"]
    cmd += ["--max-cycles", "1"]
    cmd += ["--seed", str(g["seed"])]
    return cmd


def build_sweep_slice_cmd(board: dict, idea: dict, resume: bool) -> list[str]:
    """Compose the v2 (run_sweep_wall_slice) chunk command — ONE process.

    `python scripts/run_sweep.py --cell <CELL> [--resume <idea>/checkpoints/latest.pt]
     --max-wall-secs <slice_secs> --final-eval`

    run_sweep launches the full production bundle (1 trainer + N selfplay_workers in
    wave-mode), self-caps the trainer at slice_secs on an epoch boundary (clean
    resumable latest.pt with buffer embedded), tears down workers, then --final-eval
    runs one eval_worker cycle that appends a fresh eval/model_elo line. So a v2 chunk
    needs NO separate eval step — eval is inside the bundle.
    """
    g = board["global"]
    slice_secs = float(g["slice_secs"])
    run_sweep = REPO_ROOT / "scripts" / "run_sweep.py"
    cmd: list[str] = [PYTHON, "-u", str(run_sweep), "--cell", str(idea["cell"])]
    if resume:
        latest = idea_checkpoint_dir(board, idea["name"]) / "latest.pt"
        cmd += ["--resume", str(latest)]
    cmd += ["--max-wall-secs", str(slice_secs)]
    cmd += ["--final-eval"]
    return cmd


# ---------------------------------------------------------------------------
# eval_results.jsonl parsing
# ---------------------------------------------------------------------------

def read_last_elo(board: dict, idea_name: str) -> Optional[tuple[float, Optional[int]]]:
    """Return (model_elo, epoch_evaluated) from the LAST jsonl line, or None.

    Robust to partial/garbage trailing lines: scans backwards for the last line
    that parses and carries ELO_KEY.
    """
    ckpt_dir = idea_checkpoint_dir(board, idea_name)
    jsonl = ckpt_dir / "eval_results.jsonl"
    if not jsonl.exists():
        return None
    try:
        with open(jsonl) as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ELO_KEY in rec:
            elo = float(rec[ELO_KEY])
            epoch = rec.get(EPOCH_KEY)
            epoch = int(epoch) if epoch is not None else None
            return elo, epoch
    return None


def count_jsonl_lines(board: dict, idea_name: str) -> int:
    jsonl = idea_checkpoint_dir(board, idea_name) / "eval_results.jsonl"
    if not jsonl.exists():
        return 0
    try:
        with open(jsonl) as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def extract_wandb_run_id(board: dict, idea_name: str) -> Optional[str]:
    """Best-effort: read wandb_run_id from latest.pt's metadata, if present."""
    latest = idea_checkpoint_dir(board, idea_name) / "latest.pt"
    if not latest.exists():
        return None
    try:
        import torch  # local import; heavy
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        rid = payload.get("wandb_run_id") if isinstance(payload, dict) else None
        return str(rid) if rid else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------

def run_subprocess(cmd: list[str], log_path: Path, timeout: float) -> tuple[int, str]:
    """Run a subprocess, streaming combined stdout/stderr (appended) to log_path.

    Returns (returncode, status) where status is 'ok' | 'timeout' | 'error'.
    Never raises — all failure modes are mapped to a returncode/status pair.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"\n===== {now_iso()} :: {' '.join(cmd)} =====\n"
    try:
        with open(log_path, "a") as logf:
            logf.write(header)
            logf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                logf.write(f"\n[derby] TIMEOUT after {timeout}s; killed.\n")
                logf.flush()
                return -1, "timeout"
            rc = proc.returncode
            if rc != 0:
                logf.write(f"\n[derby] exited nonzero rc={rc}\n")
                logf.flush()
                return rc, "error"
            return 0, "ok"
    except Exception as e:  # OSError launching, etc.
        log_line(log_path, f"[derby] subprocess launch failed: {e}")
        return -2, "error"


# ---------------------------------------------------------------------------
# Peak-checkpoint snapshotting
# ---------------------------------------------------------------------------

def snapshot_peak(board: dict, idea_name: str, elo: float) -> Optional[str]:
    """If `elo` is a new high for the idea, copy its current latest.pt to a derby-owned
    peak path so keep-last-n pruning can't discard the champion. Returns the peak path
    (str) on success, else None. Best-effort: never raises (a failed copy is logged but
    must not kill the race)."""
    src = idea_checkpoint_dir(board, idea_name) / "latest.pt"
    if not src.exists():
        return None
    peak_dir = idea_peak_dir(board, idea_name)
    dst = peak_dir / "peak.pt"
    try:
        import shutil
        peak_dir.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".pt.tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        return str(dst)
    except OSError as e:
        log_line(milestones_log_path(board), f"WARN peak-snapshot {idea_name} failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Standings / research-board output
# ---------------------------------------------------------------------------

def write_research_board(board: dict, state: dict, board_md_path: Path) -> None:
    """Rewrite the standings section below the sentinel; preserve prose above it."""
    g = board["global"]
    beat_elo = float(g.get("beat_heuristic_elo", 800))
    is_v2 = engine_of(board) == RUN_SWEEP_WALL_SLICE
    cap = int(g["cap_epochs"]) if not is_v2 else 0
    cap_wall = float(g["cap_wall_secs"]) if is_v2 else 0.0
    progress_col = "Wall (of cap)" if is_v2 else "Epochs"
    floor = 389.0  # "beats random, nothing above" — the no-skill elo floor

    # Build rows sorted by current elo desc.
    rows = []
    for name, st in state["ideas"].items():
        hist = st.get("elo_history", [])
        elo = hist[-1][1] if hist else None
        # peak elo + cumulative wall-seconds AT the peak (entries are [ep,elo,wall]).
        peak = None; wall_to_peak = 0.0; cum = 0.0
        for entry in hist:
            w = entry[2] if len(entry) >= 3 else 0.0
            cum += w
            if peak is None or entry[1] > peak:
                peak = entry[1]; wall_to_peak = cum
        wall_total = st.get("wall_secs_total", cum)
        # Δelo/hr = real-strength gain (above the no-skill floor) per wall-hour to peak.
        delo_per_hr = ((peak - floor) / (wall_to_peak / 3600.0)) if (peak is not None and wall_to_peak > 0) else None
        beat = "✓" if (peak is not None and peak >= beat_elo) else ""
        if is_v2:
            progress = (f"{wall_total/3600.0:.2f}/{cap_wall/3600.0:.2f}h"
                        if cap_wall > 0 else f"{wall_total/3600.0:.2f}h")
        else:
            progress = f"{st.get('epochs_done', 0)}/{cap}"
        rows.append({
            "name": name,
            "progress": progress,
            "epochs_done": st.get("epochs_done", 0),
            "elo": elo,
            "peak": peak,
            "delo": st.get("last_delo", 0.0),
            "wall_min": wall_total / 60.0,
            "delo_per_hr": delo_per_hr,
            "beat": beat,
            "status": st.get("status", "queued"),
        })

    def sort_key(r):
        # Ideas with no elo yet sort last; among rated, highest elo first.
        return (0 if r["elo"] is None else 1, r["elo"] if r["elo"] is not None else 0.0)

    rows.sort(key=sort_key, reverse=True)

    lines: list[str] = []
    lines.append("")
    lines.append("## Standings")
    lines.append("")
    lines.append(f"_Last updated: {now_iso()} — {state.get('total_chunks_run', 0)} chunks run._")
    lines.append("")

    # leader / champion summary
    rated = [r for r in rows if r["elo"] is not None]
    if rated:
        leader = rated[0]
        prog_word = "wall" if is_v2 else "epochs"
        lines.append(
            f"**Champion so far:** `{leader['name']}` at "
            f"{leader['elo']:.0f} elo ({leader['progress']} {prog_word})."
        )
    else:
        lines.append("**Champion so far:** _(no rated ideas yet)_")
    lines.append("")

    # table — north-star is Δelo/hr (real-strength gain above the floor per wall-hour to peak)
    lines.append(f"| Rank | Idea | {progress_col} | Elo | Peak | Wall (min) | Δelo/hr | Beat heuristic? | Status |")
    lines.append("|-----:|------|:------:|----:|-----:|-----------:|--------:|:---------------:|--------|")
    for i, r in enumerate(rows, start=1):
        elo_s = f"{r['elo']:.0f}" if r["elo"] is not None else "—"
        peak_s = f"{r['peak']:.0f}" if r["peak"] is not None else "—"
        wall_s = f"{r['wall_min']:.1f}" if r["wall_min"] else "—"
        dph_s = f"{r['delo_per_hr']:.0f}" if r["delo_per_hr"] is not None else "—"
        lines.append(
            f"| {i} | {r['name']} | {r['progress']} | {elo_s} | {peak_s} | "
            f"{wall_s} | {dph_s} | {r['beat']} | {r['status']} |"
        )
    lines.append("")
    lines.append("_Δelo/hr = (peak elo − 389 floor) ÷ wall-hours-to-peak: real-strength gain "
                 "per wall-clock hour, the north-star. Beat-heuristic ✓ = peak ≥ 800._")
    standings_block = "\n".join(lines)

    # Preserve everything up to and including the sentinel; rewrite below.
    if board_md_path.exists():
        try:
            existing = board_md_path.read_text()
        except OSError:
            existing = ""
        if STANDINGS_SENTINEL in existing:
            head = existing.split(STANDINGS_SENTINEL)[0]
            new_text = head + STANDINGS_SENTINEL + "\n" + standings_block + "\n"
        else:
            # No sentinel found: keep existing prose, append sentinel + standings.
            new_text = existing.rstrip() + "\n\n" + STANDINGS_SENTINEL + "\n" + standings_block + "\n"
    else:
        header = (
            "# Research Board — Δelo Derby\n\n"
            "_Title-card prose is owned by the orchestrator and lives above the "
            "sentinel line below. The standings section is auto-generated by "
            "`scripts/delo_derby.py`._\n\n"
        )
        new_text = header + STANDINGS_SENTINEL + "\n" + standings_block + "\n"

    atomic_write_text(board_md_path, new_text)


# ---------------------------------------------------------------------------
# Milestone tracking
# ---------------------------------------------------------------------------

def milestones_log_path(board: dict) -> Path:
    return resolve_path(board["global"]["base_out_dir"]) / "derby_milestones.log"


def current_leader(state: dict) -> Optional[str]:
    best_name, best_elo = None, None
    for name, st in state["ideas"].items():
        hist = st.get("elo_history", [])
        if not hist:
            continue
        elo = hist[-1][1]
        if best_elo is None or elo > best_elo:
            best_elo, best_name = elo, name
    return best_name


# ---------------------------------------------------------------------------
# Core chunk execution
# ---------------------------------------------------------------------------

def run_chunk(board: dict, state: dict, idea_name: str, args: argparse.Namespace) -> None:
    """Dispatch one chunk to the board's engine. Never raises (see per-engine impls)."""
    if engine_of(board) == RUN_SWEEP_WALL_SLICE:
        run_sweep_chunk(board, state, idea_name, args)
    else:
        run_single_process_chunk(board, state, idea_name, args)


def run_single_process_chunk(board: dict, state: dict, idea_name: str,
                             args: argparse.Namespace) -> None:
    """v1 engine: run ONE chunk for an idea: train (--no-eval) then a single eval pass.

    Updates state in-place and writes it atomically. Never raises; on any failure
    the idea is marked 'errored' (with one optional retry) and the race continues.
    """
    g = board["global"]
    chunk = int(g["chunk_epochs"])
    cap = int(g["cap_epochs"])
    beat_elo = float(g.get("beat_heuristic_elo", 800))
    st = state["ideas"][idea_name]
    idea_spec = next(i for i in board["ideas"] if i["name"] == idea_name)

    epochs_before = st["epochs_done"]
    resume = epochs_before > 0
    elo_prev = st["elo_history"][-1][1] if st["elo_history"] else 0.0
    chunk_log = idea_chunk_log(board, idea_name)
    milestones = milestones_log_path(board)
    state_p = state_path_for(board)

    leader_before = current_leader(state)
    beat_before = any(
        s.get("elo_history") and s["elo_history"][-1][1] >= beat_elo
        for n, s in state["ideas"].items() if n == idea_name
    )

    st["status"] = "running"
    state["updated"] = now_iso()
    atomic_write_json(state_p, state)

    target = epochs_before + chunk
    t0 = time.time()
    derby_print(f"[derby] {idea_name} chunk @epoch {epochs_before}->{target} starting "
                f"(resume={resume})")

    # --- step 1: trainer ---
    train_cmd = build_train_cmd(board, idea_spec, resume=resume)
    rc, status = run_subprocess(train_cmd, chunk_log, args.chunk_timeout)
    if status != "ok":
        _handle_chunk_failure(board, state, idea_name, args,
                              f"train {status} (rc={rc})", milestones, state_p)
        return

    # --- step 2: single eval pass ---
    lines_before = count_jsonl_lines(board, idea_name)
    eval_cmd = build_eval_cmd(board, idea_spec)
    rc, status = run_subprocess(eval_cmd, chunk_log, args.chunk_timeout)
    if status != "ok":
        # Train succeeded; eval failed. Advance epochs (model DID train) but carry
        # previous elo so we don't lose the chunk's training. Don't mark errored —
        # the model is fine; eval is just missing this round.
        dt = time.time() - t0
        st["epochs_done"] = target
        st["last_delo"] = 0.0
        st["elo_history"].append([target, elo_prev, round(dt, 1)])
        st["wall_secs_total"] = st.get("wall_secs_total", 0.0) + dt
        st["wandb_run_id"] = st.get("wandb_run_id") or extract_wandb_run_id(board, idea_name)
        if st["epochs_done"] >= cap:
            st["status"] = "capped"
        else:
            st["status"] = "queued"
        state["total_chunks_run"] += 1
        state["updated"] = now_iso()
        atomic_write_json(state_p, state)
        msg = (f"[derby] {idea_name} eval {status} (rc={rc}); carried elo "
               f"{elo_prev:.0f}, epochs->{target}")
        derby_print(msg)
        log_line(milestones, f"WARN {idea_name} eval failed; carried elo {elo_prev:.0f} @epoch {target}")
        write_research_board(board, state, board_md_path_for(board))
        return

    # --- parse new elo ---
    lines_after = count_jsonl_lines(board, idea_name)
    parsed = read_last_elo(board, idea_name)
    if parsed is None or lines_after <= lines_before:
        # Eval ran but produced no new parseable elo line: carry previous elo.
        dt = time.time() - t0
        st["epochs_done"] = target
        st["last_delo"] = 0.0
        st["elo_history"].append([target, elo_prev, round(dt, 1)])
        st["wall_secs_total"] = st.get("wall_secs_total", 0.0) + dt
        if st["epochs_done"] >= cap:
            st["status"] = "capped"
        else:
            st["status"] = "queued"
        state["total_chunks_run"] += 1
        state["updated"] = now_iso()
        atomic_write_json(state_p, state)
        derby_print(f"[derby] {idea_name} WARNING: no new elo line; carried {elo_prev:.0f}")
        log_line(milestones, f"WARN {idea_name} no new elo line @epoch {target}; carried {elo_prev:.0f}")
        write_research_board(board, state, board_md_path_for(board))
        return

    elo_now, _epoch_eval = parsed
    last_delo = elo_now - elo_prev
    dt = time.time() - t0

    st["epochs_done"] = target
    st["last_delo"] = last_delo
    st["elo_history"].append([target, elo_now, round(dt, 1)])
    st["wall_secs_total"] = st.get("wall_secs_total", 0.0) + dt
    st["wandb_run_id"] = st.get("wandb_run_id") or extract_wandb_run_id(board, idea_name)
    # Peak-checkpoint snapshot: if this is a new elo high, copy latest.pt to a
    # derby-owned peak path so keep-last-n pruning can't lose the champion.
    if st.get("peak_elo") is None or elo_now > st["peak_elo"]:
        peak_path = snapshot_peak(board, idea_name, elo_now)
        st["peak_elo"] = elo_now
        if peak_path:
            st["peak_path"] = peak_path
            log_line(milestones, f"PEAK {idea_name} new high elo {elo_now:.0f} "
                                 f"@epoch {target}; snapshot -> {peak_path}")
    if st["epochs_done"] >= cap:
        st["status"] = "capped"
    else:
        st["status"] = "queued"
    state["total_chunks_run"] += 1
    state["updated"] = now_iso()
    atomic_write_json(state_p, state)

    derby_print(
        f"[derby] {idea_name} chunk @epoch {epochs_before}->{target} ({dt:.0f}s): "
        f"elo {elo_prev:.0f}->{elo_now:.0f} (Δ{last_delo:+.0f})"
    )

    # --- milestones ---
    if (not beat_before) and elo_now >= beat_elo:
        log_line(milestones, f"BEAT-HEURISTIC {idea_name} crossed {beat_elo:.0f} "
                             f"(elo {elo_now:.0f}) @epoch {target}")
    leader_after = current_leader(state)
    if leader_after != leader_before and leader_after is not None:
        led_elo = state["ideas"][leader_after]["elo_history"][-1][1]
        log_line(milestones, f"LEAD-CHANGE new leader {leader_after} @ {led_elo:.0f} elo "
                             f"(was {leader_before})")
    if st["status"] == "capped":
        log_line(milestones, f"CAP {idea_name} reached {cap} epochs @ elo {elo_now:.0f}")

    write_research_board(board, state, board_md_path_for(board))


def _handle_chunk_failure(board: dict, state: dict, idea_name: str,
                          args: argparse.Namespace, reason: str,
                          milestones: Path, state_p: Path) -> None:
    """Mark an idea errored (with one optional retry) and persist; race continues."""
    st = state["ideas"][idea_name]
    st["retries"] = st.get("retries", 0)
    if st["retries"] < args.max_retries:
        st["retries"] += 1
        st["status"] = "queued"  # leave epochs_done unchanged; it will be re-picked
        state["updated"] = now_iso()
        atomic_write_json(state_p, state)
        derby_print(f"[derby] {idea_name} chunk FAILED ({reason}); retry "
                    f"{st['retries']}/{args.max_retries} queued")
        log_line(milestones, f"RETRY {idea_name} chunk failed ({reason}); "
                             f"retry {st['retries']}/{args.max_retries}")
    else:
        st["status"] = "errored"
        state["updated"] = now_iso()
        atomic_write_json(state_p, state)
        derby_print(f"[derby] {idea_name} chunk FAILED ({reason}); ERRORED out "
                    f"(retries exhausted)")
        log_line(milestones, f"ERROR {idea_name} errored ({reason}) after "
                             f"{st['retries']} retries @epoch {st['epochs_done']}")
    write_research_board(board, state, board_md_path_for(board))


def run_sweep_chunk(board: dict, state: dict, idea_name: str,
                    args: argparse.Namespace) -> None:
    """v2 engine: advance an idea by ONE WALL-TIME SLICE via the production recipe.

    ONE subprocess: `run_sweep.py --cell <CELL> [--resume latest.pt] --max-wall-secs
    <slice_secs> --final-eval`. That command runs the multiprocess bundle for the
    slice, self-caps with a clean resumable latest.pt, AND ends with a fresh
    eval/model_elo line in eval_results.jsonl — so there is NO separate eval step.

    Budget is WALL-NATIVE: an idea caps when wall_secs_total >= cap_wall_secs. We bill
    actual measured wall-time per chunk (which includes the slice + teardown + final
    eval) so Δelo/hr is honest against the saturated machine.

    Updates state in-place, writes atomically, never raises. On launch/non-zero failure
    the idea is retried/errored (mirrors v1). On a missing/unparseable elo line the
    chunk is still billed (the model DID train + save) and previous elo is carried.
    """
    g = board["global"]
    slice_secs = float(g["slice_secs"])
    cap_wall = float(g["cap_wall_secs"])
    beat_elo = float(g.get("beat_heuristic_elo", 800))
    st = state["ideas"][idea_name]
    idea_spec = next(i for i in board["ideas"] if i["name"] == idea_name)

    resume = st["wall_secs_total"] > 0  # first chunk fresh; later chunks --resume latest.pt
    elo_prev = st["elo_history"][-1][1] if st["elo_history"] else 0.0
    chunk_log = idea_chunk_log(board, idea_name)
    milestones = milestones_log_path(board)
    state_p = state_path_for(board)

    leader_before = current_leader(state)
    beat_before = st.get("peak_elo") is not None and st["peak_elo"] >= beat_elo

    st["status"] = "running"
    state["updated"] = now_iso()
    atomic_write_json(state_p, state)

    t0 = time.time()
    derby_print(f"[derby/v2] {idea_name} (cell={idea_spec['cell']}) wall-slice "
                f"{slice_secs:g}s starting (resume={resume}, "
                f"{st['wall_secs_total']:.0f}/{cap_wall:.0f}s spent)")

    # --- the single multiprocess chunk: run_sweep slice WITH --final-eval ---
    # Give run_sweep enough wall room to self-cap + teardown + final eval, then a
    # safety margin on top so a clean cap is never preempted by our own timeout.
    slice_timeout = slice_secs + 600.0
    if args.chunk_timeout and args.chunk_timeout > slice_timeout:
        slice_timeout = args.chunk_timeout
    lines_before = count_jsonl_lines(board, idea_name)
    cmd = build_sweep_slice_cmd(board, idea_spec, resume=resume)
    rc, status = run_subprocess(cmd, chunk_log, slice_timeout)
    dt = time.time() - t0
    # A COMPLETED slice bills at least its requested wall slice toward the per-idea
    # cap, even if the subprocess returned faster than slice_secs. This guarantees
    # forward progress toward cap_wall_secs so the leader-priority loop can't pin
    # forever on an idea whose slices happen to exit early. We record the TRUE dt in
    # elo_history (so Δelo/hr stays honest), but bill billed_dt against the budget.
    billed_dt = max(dt, slice_secs)

    if status != "ok":
        # Even a non-zero/timeout slice usually leaves a clean resumable latest.pt
        # (the trainer's SIGTERM handler force-saves). But to mirror v1's failure
        # discipline we DON'T bill it — we retry/error so a genuinely broken cell
        # (e.g. cell not yet defined in run_sweep.CELLS) can't burn the whole budget.
        _handle_chunk_failure(board, state, idea_name, args,
                              f"run_sweep {status} (rc={rc})", milestones, state_p)
        return

    # --- bill the wall time (the chunk completed; the model trained + saved) ---
    st["wall_secs_total"] = st.get("wall_secs_total", 0.0) + billed_dt
    st["chunks_done"] = st.get("chunks_done", 0) + 1
    st["wandb_run_id"] = st.get("wandb_run_id") or extract_wandb_run_id(board, idea_name)

    # --- parse the fresh elo line written by --final-eval ---
    lines_after = count_jsonl_lines(board, idea_name)
    parsed = read_last_elo(board, idea_name)
    if parsed is None or lines_after <= lines_before:
        # Slice ran + billed, but --final-eval produced no new parseable elo line.
        # Carry previous elo; the training progress is NOT lost (latest.pt is fresh).
        st["last_delo"] = 0.0
        epoch_tag = st["epochs_done"]
        st["elo_history"].append([epoch_tag, elo_prev, round(dt, 1)])
        st["status"] = "capped" if st["wall_secs_total"] >= cap_wall else "queued"
        state["total_chunks_run"] += 1
        state["updated"] = now_iso()
        atomic_write_json(state_p, state)
        derby_print(f"[derby/v2] {idea_name} WARNING: no new elo line; carried "
                    f"{elo_prev:.0f} ({dt:.0f}s billed)")
        log_line(milestones, f"WARN {idea_name} no new elo line after slice; "
                             f"carried {elo_prev:.0f}")
        write_research_board(board, state, board_md_path_for(board))
        return

    elo_now, epoch_eval = parsed
    last_delo = elo_now - elo_prev
    # Track real epoch progress from eval_worker's epoch tag, when available.
    if epoch_eval is not None:
        st["epochs_done"] = epoch_eval
    st["last_delo"] = last_delo
    st["elo_history"].append([epoch_eval if epoch_eval is not None else st["epochs_done"],
                              elo_now, round(dt, 1)])

    # --- peak-checkpoint snapshot (survives keep-last-n pruning) ---
    if st.get("peak_elo") is None or elo_now > st["peak_elo"]:
        peak_path = snapshot_peak(board, idea_name, elo_now)
        st["peak_elo"] = elo_now
        if peak_path:
            st["peak_path"] = peak_path
            log_line(milestones, f"PEAK {idea_name} new high elo {elo_now:.0f}; "
                                 f"snapshot -> {peak_path}")

    st["status"] = "capped" if st["wall_secs_total"] >= cap_wall else "queued"
    state["total_chunks_run"] += 1
    state["updated"] = now_iso()
    atomic_write_json(state_p, state)

    derby_print(
        f"[derby/v2] {idea_name} wall-slice done ({dt:.0f}s, "
        f"{st['wall_secs_total']:.0f}/{cap_wall:.0f}s): "
        f"elo {elo_prev:.0f}->{elo_now:.0f} (Δ{last_delo:+.0f})"
    )

    # --- milestones ---
    if (not beat_before) and elo_now >= beat_elo:
        log_line(milestones, f"BEAT-HEURISTIC {idea_name} crossed {beat_elo:.0f} "
                             f"(elo {elo_now:.0f}) @ {st['wall_secs_total']:.0f}s")
    leader_after = current_leader(state)
    if leader_after != leader_before and leader_after is not None:
        led_elo = state["ideas"][leader_after]["elo_history"][-1][1]
        log_line(milestones, f"LEAD-CHANGE new leader {leader_after} @ {led_elo:.0f} elo "
                             f"(was {leader_before})")
    if st["status"] == "capped":
        log_line(milestones, f"CAP {idea_name} reached {cap_wall:.0f}s wall budget "
                             f"@ elo {elo_now:.0f} (peak {st['peak_elo']:.0f})")

    write_research_board(board, state, board_md_path_for(board))


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def active_ideas(board: dict, state: dict) -> list[str]:
    """Ideas eligible for more chunks: not capped, not errored."""
    out = []
    for idea in board["ideas"]:
        name = idea["name"]
        st = state["ideas"][name]
        if st["status"] == "errored":
            continue
        if idea_is_capped(board, st):
            # ensure marked capped
            if st["status"] != "capped":
                st["status"] = "capped"
            continue
        out.append(name)
    return out


def idea_is_capped(board: dict, st: dict) -> bool:
    """Engine-aware cap test. v1: epochs_done >= cap_epochs. v2: wall_secs_total
    >= cap_wall_secs."""
    if engine_of(board) == RUN_SWEEP_WALL_SLICE:
        return st.get("wall_secs_total", 0.0) >= float(board["global"]["cap_wall_secs"])
    return st["epochs_done"] >= int(board["global"]["cap_epochs"])


def idea_started(board: dict, st: dict) -> bool:
    """Has the idea run its first (entry-fee) chunk? Engine-aware: v1 keys on
    epochs_done, v2 keys on chunks_done (epochs_done is only a best-effort tag in v2)."""
    if engine_of(board) == RUN_SWEEP_WALL_SLICE:
        return st.get("chunks_done", 0) > 0
    return st["epochs_done"] > 0


FLOOR_ELO = 389.0  # "beats random, nothing above" — the no-skill floor (matches render_standings)


def delo_per_hr(state: dict, name: str) -> float | None:
    """Δelo/HOUR — the recent elo-climb RATE (last-chunk slope) for an idea.

    Returns None when the rate is NOT YET MEASURABLE (fewer than 2 elo points):
    you cannot compute a climb rate from a single measurement, and at round-0
    every fresh idea sits at the same no-skill floor (~389), indistinguishable.
    Such ideas are "entry fee" — they need another chunk to establish a slope
    before the hill-climb can rank them (handled in pick_priority).

    With >=2 points: (elo[-1] - elo[-2]) / last_chunk_wall_hours — the most recent
    gradient, so compute follows where strength is currently rising fastest.
    """
    hist = state["ideas"][name].get("elo_history") or []
    if len(hist) < 2:
        return None  # never-run OR measured-once: rate not yet defined
    last_wall = hist[-1][2] if len(hist[-1]) >= 3 else 0.0
    wall_hr = last_wall / 3600.0
    if wall_hr <= 0:
        return 0.0
    return (hist[-1][1] - hist[-2][1]) / wall_hr


def pick_priority(board: dict, state: dict, candidates: list[str]) -> str:
    """Hill-climb by Δelo/HOUR (Jason's directive 2026-05-24: "priority by never-run,
    then by delta elo/hour — hill climb elo"). Priority order:

      1. NEVER-RUN ideas first — the entry fee. (Round-0 normally drains these before
         the priority loop even starts; this also guards an idea whose round-0 errored
         and left it with no elo.)
      2. Then by Δelo/HOUR over the most recent chunk, DESCENDING — feed the steepest
         elo gradient. Compute goes where strength is rising fastest, NOT to the
         highest absolute elo (that was the v2 "feed-the-leader" rule this replaces:
         it over-fed an already-peaked champion and starved a faster-climbing
         challenger). Tie-break: higher current elo, fewer chunks, then name.

    On the oscillation tradeoff (why v1 once moved OFF a Δ-rule): ranking by the
    RATE is not the v1 pathology. v1 ranked by last-chunk *raw Δelo* and fed the
    WORST idea (a floored idea at Δ0 outranked a strong idea whose chunk dipped). Here
    a floored idea sits at Δ0/hr and ANY genuine climber (Δ>0/hr) outranks it; a strong
    idea that dips for one chunk is correctly paused (it stopped climbing) and the
    cap-guarantee (everyone caps eventually) prevents permanent starvation."""
    def key(name: str):
        st = state["ideas"][name]
        hist = st.get("elo_history") or []
        npts = len(hist)
        rate = delo_per_hr(state, name)            # None until 2 elo points exist
        chunks = st.get("chunks_done", st.get("epochs_done", 0))
        if rate is None:
            # ENTRY-FEE group: needs (another) chunk to establish a Δelo/hr slope.
            # Fewest elo points first → never-run (0) before measured-once (1);
            # then fewest chunks, then name. This drains round-0 then round-1.
            return (0, npts, chunks, name)
        # HILL-CLIMB group: steepest recent Δelo/hr first; TIES broken by PEAK elo
        # (demonstrated potential — Jason 2026-05-24: two equal Δelo/hr → the one
        # that reached higher wins), then current elo, fewer chunks, name.
        peak = max((h[1] for h in hist), default=float("-inf"))
        return (1, -rate, -peak, -hist[-1][1], chunks, name)
    return sorted(candidates, key=key)[0]


def round0_remaining(board: dict, state: dict) -> list[str]:
    """Ideas that have not yet run their first (entry-fee) chunk, in board order."""
    out = []
    for idea in board["ideas"]:
        name = idea["name"]
        st = state["ideas"][name]
        if st["status"] == "errored":
            continue
        if not idea_started(board, st):
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def do_dry_run(board: dict) -> None:
    if engine_of(board) == RUN_SWEEP_WALL_SLICE:
        do_dry_run_v2(board)
    else:
        do_dry_run_v1(board)


def do_dry_run_v2(board: dict) -> None:
    g = board["global"]
    base = resolve_path(g["base_out_dir"])
    slice_secs = float(g["slice_secs"])
    cap_wall = float(g["cap_wall_secs"])
    chunks_per_idea = int(cap_wall // slice_secs) if slice_secs > 0 else 0
    derby_print("=" * 78)
    derby_print("Δelo DERBY v2 — DRY RUN (engine: run_sweep_wall_slice, WALL-NATIVE)")
    derby_print("=" * 78)
    derby_print(f"base_out_dir : {base}  (derby state + peak snapshots)")
    derby_print(f"run_sweep checkpoints/eval live in: sweep_runs/<cell_name>/checkpoints/")
    derby_print(f"slice_secs   : {slice_secs:g}s/chunk   cap_wall_secs: {cap_wall:g}s/idea "
                f"=> ~{chunks_per_idea} chunks/idea ({cap_wall/3600.0:.2f} wall-hr/idea)")
    derby_print(f"beat_heuristic_elo: {g.get('beat_heuristic_elo')}")
    derby_print(f"ideas: {len(board['ideas'])}   (eval is INSIDE the bundle via --final-eval)")
    derby_print("")
    derby_print(f"Total chunks to run (all ideas to cap): ~{chunks_per_idea * len(board['ideas'])} "
                f"({cap_wall * len(board['ideas']) / 3600.0:.2f} wall-hr total)")
    derby_print("")
    derby_print("Plan:")
    derby_print("  1. Round 0: every idea runs its first wall-slice FRESH (no --resume).")
    derby_print("  2. Priority loop: NEVER-RUN first, then highest Δelo/HOUR among active")
    derby_print("     ideas (hill-climb the elo gradient); tie-break elo, chunks, name. Everyone caps.")
    derby_print("  3. Each chunk = ONE `run_sweep --max-wall-secs --final-eval` command:")
    derby_print("     multiprocess bundle (1 trainer + N selfplay_workers, wave-mode,")
    derby_print("     GPU-saturated) self-caps + final eval writes a fresh model_elo line.")
    derby_print("  4. Peak-checkpoint snapshot: new elo high copies latest.pt -> _peaks/<idea>/peak.pt.")
    derby_print("")
    for idea in board["ideas"]:
        derby_print("-" * 78)
        derby_print(f"IDEA: {idea['name']}   cell={idea['cell']}  "
                    f"lever: {idea.get('lever', '')}")
        ckpt = idea_checkpoint_dir(board, idea["name"])
        round0 = build_sweep_slice_cmd(board, idea, resume=False)
        later = build_sweep_slice_cmd(board, idea, resume=True)
        derby_print("  round-0 chunk cmd (fresh):")
        derby_print("    " + " ".join(round0))
        derby_print("  later chunk cmd (resumed):")
        derby_print("    " + " ".join(later))
        derby_print(f"  reads model_elo from: {ckpt}/eval_results.jsonl")
        derby_print(f"  resumes from:         {ckpt}/latest.pt")
        derby_print(f"  peak snapshot to:     {idea_peak_dir(board, idea['name'])}/peak.pt")
    derby_print("-" * 78)
    derby_print("NOTE: cells must be added to run_sweep.CELLS before a REAL run "
                "(orchestrator owns run_sweep.py).")
    derby_print("Dry run complete; nothing executed.")


def do_dry_run_v1(board: dict) -> None:
    g = board["global"]
    base = resolve_path(g["base_out_dir"])
    derby_print("=" * 78)
    derby_print("Δelo DERBY — DRY RUN")
    derby_print("=" * 78)
    derby_print(f"base_out_dir : {base}")
    derby_print(f"chunk_epochs : {g['chunk_epochs']}   cap_epochs: {g['cap_epochs']}   "
                f"=> {g['cap_epochs'] // g['chunk_epochs']} chunks/idea")
    derby_print(f"size={g['size']} seed={g['seed']} device={g['device']} "
                f"wandb={g.get('wandb')}")
    derby_print(f"beat_heuristic_elo: {g.get('beat_heuristic_elo')}")
    derby_print(f"eval: baselines={g.get('eval_baselines')!r} "
                f"slow={g.get('eval_baselines_slow')!r} "
                f"games={g.get('eval_baseline_games')} sims={g.get('eval_sims')}")
    derby_print(f"ideas: {len(board['ideas'])}")
    derby_print("")
    total_chunks = sum(g["cap_epochs"] // g["chunk_epochs"] for _ in board["ideas"])
    derby_print(f"Total chunks to run (all ideas to cap): {total_chunks}")
    derby_print("")
    derby_print("Plan:")
    derby_print("  1. Round 0: every idea runs its first chunk FRESH (no --resume).")
    derby_print("  2. Priority loop: pick max last_delo among active ideas, run next chunk.")
    derby_print("     Tie-break: fewest epochs_done, then name.")
    derby_print("  3. Each chunk = train(--no-eval, --epochs chunk) THEN eval_worker(--max-cycles 1).")
    derby_print("")
    for idea in board["ideas"]:
        derby_print("-" * 78)
        derby_print(f"IDEA: {idea['name']}   lever: {idea.get('lever', '')}")
        train_cmd = build_train_cmd(board, idea, resume=False)
        eval_cmd = build_eval_cmd(board, idea)
        derby_print("  round-0 train cmd:")
        derby_print("    " + " ".join(train_cmd))
        derby_print("  round-0 eval cmd:")
        derby_print("    " + " ".join(eval_cmd))
        derby_print("  (later chunks add: --resume "
                    f"{idea_checkpoint_dir(board, idea['name'])}/latest.pt)")
    derby_print("-" * 78)
    derby_print("Dry run complete; nothing executed.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Δelo Derby — autonomous training-recipe race scheduler")
    ap.add_argument("--board", default="scripts/derby_board.json",
                    help="Path to the board spec JSON (default: scripts/derby_board.json)")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from derby_state.json (re-derive next pick)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the full plan + exact round-0 commands; run nothing")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="Stop after N total chunks this invocation (0 = run to completion)")
    ap.add_argument("--chunk-timeout", type=float, default=1800.0,
                    help="Per-subprocess timeout in seconds (default 1800)")
    ap.add_argument("--max-retries", type=int, default=1,
                    help="Retry a failed chunk up to N times before marking errored")
    args = ap.parse_args()

    board_path = resolve_path(args.board)
    board = load_board(board_path)
    base = resolve_path(board["global"]["base_out_dir"])
    base.mkdir(parents=True, exist_ok=True)
    derby_log = base / "derby.log"

    if args.dry_run:
        do_dry_run(board)
        return 0

    state_p = state_path_for(board)
    if args.resume:
        state = load_state(state_p)
        if state is None:
            derby_print(f"[derby] --resume given but no state at {state_p}; starting fresh")
            state = fresh_state(board)
        else:
            derby_print(f"[derby] resumed from {state_p}")
            # Reconcile board ideas with state (a new idea added to the board joins fresh).
            for idea in board["ideas"]:
                if idea["name"] not in state["ideas"]:
                    state["ideas"][idea["name"]] = fresh_state(board)["ideas"][idea["name"]]
            # Reset any idea left 'running' (crash mid-chunk): re-queue it. Its
            # epochs_done reflects only COMPLETED chunks, so re-picking redoes the
            # interrupted chunk cleanly (train --resume from the last good latest.pt).
            for name, st in state["ideas"].items():
                if st["status"] == "running":
                    st["status"] = "queued"
                    derby_print(f"[derby] {name} was 'running' at crash; re-queued")
    else:
        state = fresh_state(board)
    state["board_path"] = str(board_path)
    atomic_write_json(state_p, state)
    write_research_board(board, state, board_md_path_for(board))

    chunks_this_run = 0

    def budget_left() -> bool:
        return args.max_chunks <= 0 or chunks_this_run < args.max_chunks

    try:
        # ----- Round 0: entry fee -----
        if not state.get("round0_done", False):
            for name in round0_remaining(board, state):
                if not budget_left():
                    derby_print(f"[derby] hit --max-chunks={args.max_chunks}; stopping (round 0)")
                    return 0
                try:
                    run_chunk(board, state, name, args)
                except Exception as e:
                    # Defensive: run_chunk shouldn't raise, but never let the loop die.
                    log_line(derby_log, f"UNEXPECTED in round0 {name}: {e}\n{traceback.format_exc()}")
                    state["ideas"][name]["status"] = "errored"
                    atomic_write_json(state_p, state)
                chunks_this_run += 1
            state["round0_done"] = True
            atomic_write_json(state_p, state)
            derby_print("[derby] round 0 complete.")

        # ----- Priority loop -----
        while budget_left():
            candidates = active_ideas(board, state)
            atomic_write_json(state_p, state)  # persist any capped-status reconciliation
            if not candidates:
                derby_print("[derby] all ideas capped or errored — race complete.")
                break
            pick = pick_priority(board, state, candidates)
            try:
                run_chunk(board, state, pick, args)
            except Exception as e:
                log_line(derby_log, f"UNEXPECTED in loop {pick}: {e}\n{traceback.format_exc()}")
                state["ideas"][pick]["status"] = "errored"
                atomic_write_json(state_p, state)
            chunks_this_run += 1

        if not budget_left():
            derby_print(f"[derby] hit --max-chunks={args.max_chunks}; stopping.")
    except KeyboardInterrupt:
        derby_print("[derby] interrupted; state is consistent. Re-run with --resume.")
        atomic_write_json(state_p, state)
        return 130
    except Exception as e:
        log_line(derby_log, f"FATAL main loop: {e}\n{traceback.format_exc()}")
        derby_print(f"[derby] FATAL: {e} (logged to {derby_log}); state left consistent.")
        atomic_write_json(state_p, state)
        return 1

    # final standings refresh
    write_research_board(board, state, board_md_path_for(board))
    derby_print(f"[derby] done. {state['total_chunks_run']} total chunks. "
                f"leader={current_leader(state)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
