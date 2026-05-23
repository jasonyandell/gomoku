"""Canonical 5-axis perf sweep for M5 Max gomoku self-play.

Implements step 4 of `wiki/topics/m5-max-as-mainframe.md` (the contour
plot "on the wall"): map the production-shape self-play throughput
contour over workers x games-per-worker x n-simulations x wave-size x
model-size. Each cell spawns N selfplay_worker subprocesses against a
fresh pre-saved fused checkpoint, bounds wall time, then counts records
and computes aug-positions/sec.

Per `wiki/topics/perf-lab-session-runbook.md`:
  - default --max-plies 16, --wave-mode on, --device mps
  - fused_eval on, native_mcts on, stem_padding=1
  - per-cell wall = --secs-per-cell (default 300s)

Run-tree under --out-dir:
  metadata.txt          (one block appended per session)
  cells.csv             (cell list snapshot)
  summary.tsv           (one row per completed cell; resumable source of truth)
  .sweep.lock           (PID lock; prevents concurrent invocations)
  checkpoints/{tiny,small,medium}.pt
  cell_<id>/
    records/v.../worker.../game*.pt
    logs/w*.log

Also writes/refreshes `sweep_logs/canonical-sweep-latest` symlink so
future invocations can resume the most recent sweep by passing
`--out-dir latest` (or by reading the symlink target manually).

Resumability contract (also documented in
`wiki/topics/perf-lab-session-runbook.md`):
  - cell_id is derived purely from cell params; the list can grow over
    time without renumbering existing rows.
  - cells with a row in summary.tsv (status=ok) are skipped on resume.
  - cells with status=failed are skipped by default; pass --retry-failed
    to re-run them (their old row is removed and cell_dir wiped).
  - --status reports done/failed/remaining/ETA and exits without running.
  - --max-wall-secs N exits cleanly after the budget; resume picks up
    the next cell.
  - SIGINT/SIGTERM kills live workers and exits cleanly mid-cell.
  - Lock file prevents two invocations from racing on the same out-dir.
"""

from __future__ import annotations

import argparse
import csv
import errno
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# Flush each print line so background `nohup ... > driver.log` stays
# readable mid-run instead of buffering until the buffer fills.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gomoku.model import build_model, save_checkpoint  # noqa: E402


# --- Cell design --------------------------------------------------------------

# Default (canonical) cell. Matches the production-contour-20260522
# winner: native small 8w x 8g, sims=400, wave=64. All single-axis
# sweeps below pivot off this point.
DEFAULT = dict(
    model="small",
    workers=8,
    games_per_batch=8,
    n_simulations=400,
    wave_size=64,
)

CELL_PARAM_FIELDS = ["model", "workers", "games_per_batch", "n_simulations", "wave_size"]


def cell_id_of(cell: dict) -> str:
    """Deterministic cell_id from params only — never includes an index.

    Stable across additions to the cell list, so old summary.tsv rows
    remain valid when build_cells() grows."""
    return (
        f"{cell['model']}_W{cell['workers']:02d}"
        f"_G{cell['games_per_batch']:02d}_S{cell['n_simulations']:03d}"
        f"_V{cell['wave_size']:03d}"
    )


def _axis(name: str, values: list) -> list[dict]:
    """Generate cells that vary `name` over `values`, holding others at DEFAULT."""
    cells = []
    for v in values:
        c = dict(DEFAULT)
        c[name] = v
        cells.append(c)
    return cells


def load_cells_csv(path: Path) -> list[dict]:
    """Read an ad-hoc cell list from a csv. Expected columns: model,
    workers, games_per_batch, n_simulations, wave_size. cell_id is
    re-derived; any cell_id in the csv is ignored. Used by --cells-from."""
    if not path.exists():
        raise SystemExit(f"--cells-from {path} not found")
    cells: list[dict] = []
    with path.open() as f:
        r = csv.DictReader(f)
        missing = [c for c in CELL_PARAM_FIELDS if c not in (r.fieldnames or [])]
        if missing:
            raise SystemExit(f"--cells-from {path} missing columns: {missing}")
        for row in r:
            cell = {
                "model": row["model"].strip(),
                "workers": int(row["workers"]),
                "games_per_batch": int(row["games_per_batch"]),
                "n_simulations": int(row["n_simulations"]),
                "wave_size": int(row["wave_size"]),
            }
            cell["cell_id"] = cell_id_of(cell)
            cells.append(cell)
    if not cells:
        raise SystemExit(f"--cells-from {path} is empty")
    return cells


def build_cells() -> list[dict]:
    """Build the canonical sweep cell list. Order: default first, then axes,
    then corners/diagonals. cell_id derives from params alone (stable)."""
    out: list[dict] = []
    seen: set[tuple] = set()

    def add(cell: dict) -> None:
        key = tuple((k, cell[k]) for k in CELL_PARAM_FIELDS)
        if key in seen:
            return
        seen.add(key)
        out.append(cell)

    # 1) default
    add(dict(DEFAULT))

    # 2) workers axis
    for c in _axis("workers", [1, 2, 4, 12, 16]):
        add(c)

    # 3) games-per-batch axis
    for c in _axis("games_per_batch", [4, 16]):
        add(c)

    # 4) n-simulations axis
    for c in _axis("n_simulations", [100, 200, 800]):
        add(c)

    # 5) wave-size axis
    for c in _axis("wave_size", [32, 128, 256]):
        add(c)

    # 6) model axis
    for c in _axis("model", ["tiny", "medium"]):
        add(c)

    # 7) corners / diagonals
    add(dict(model="small",  workers=1,  games_per_batch=16, n_simulations=800, wave_size=128))
    add(dict(model="small",  workers=16, games_per_batch=4,  n_simulations=100, wave_size=32))
    add(dict(model="tiny",   workers=16, games_per_batch=16, n_simulations=100, wave_size=32))
    add(dict(model="tiny",   workers=1,  games_per_batch=4,  n_simulations=100, wave_size=32))
    add(dict(model="medium", workers=4,  games_per_batch=16, n_simulations=800, wave_size=128))
    add(dict(model="medium", workers=12, games_per_batch=8,  n_simulations=400, wave_size=64))
    add(dict(model="medium", workers=2,  games_per_batch=8,  n_simulations=400, wave_size=64))

    for c in out:
        c["cell_id"] = cell_id_of(c)
    return out


# --- Checkpoint staging -------------------------------------------------------

def stage_checkpoint(model_size: str, out_path: Path, stem_padding: int = 1) -> None:
    """Build a fresh model, save once. Workers fuse after load themselves
    (gomoku.selfplay_worker._load_model -> fuse_model_for_inference)."""
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = build_model(model_size, stem_padding=stem_padding)
    m.eval()
    save_checkpoint(str(out_path), m, epoch=0)


# --- Lock file ----------------------------------------------------------------

LOCK_NAME = ".sweep.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        if e.errno == errno.ESRCH:
            return False
        if e.errno == errno.EPERM:
            # Process exists but isn't ours; treat as alive to be safe.
            return True
        return False
    return True


def acquire_lock(out_dir: Path) -> Path:
    """Write a PID lock; abort if another live process holds it."""
    lock_path = out_dir / LOCK_NAME
    if lock_path.exists():
        try:
            other = int(lock_path.read_text().strip())
        except (ValueError, OSError):
            other = -1
        if _pid_alive(other) and other != os.getpid():
            raise SystemExit(
                f"another canonical_sweep is running on this dir (PID {other}). "
                f"If you are sure it's dead, `rm {lock_path}` and retry."
            )
    lock_path.write_text(str(os.getpid()))
    return lock_path


def release_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists() and lock_path.read_text().strip() == str(os.getpid()):
            lock_path.unlink()
    except OSError:
        pass


# --- Signal handling ----------------------------------------------------------

# Globals so the signal handler can reach across stack frames.
_ACTIVE_PROCS: list = []
_INTERRUPTED = False


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _INTERRUPTED
        _INTERRUPTED = True
        # Kill every worker we spawned in the current cell so the box clears.
        for entry in list(_ACTIVE_PROCS):
            try:
                p = entry[1]
                if p.poll() is None:
                    p.terminate()
            except (ProcessLookupError, PermissionError, IndexError):
                pass
        sys.stderr.write(
            f"\n[signal] received {signum}; terminating workers, "
            "current cell will not be recorded\n"
        )
        sys.stderr.flush()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# --- Cell runner --------------------------------------------------------------

def run_cell(
    cell: dict,
    sweep_dir: Path,
    secs_per_cell: int,
    max_plies: int,
    device: str,
    compile_model: bool = False,
    fp16_eval: bool = False,
) -> dict:
    """Spawn workers for one cell, bound wall time, return measured row.
    Mutates _ACTIVE_PROCS so the signal handler can reach the workers."""
    cell_id = cell["cell_id"]
    cell_dir = sweep_dir / f"cell_{cell_id}"
    records_dir = cell_dir / "records"
    logs_dir = cell_dir / "logs"
    records_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ckpt = sweep_dir / "checkpoints" / f"{cell['model']}.pt"

    base_cmd = [
        sys.executable, "-u", "-m", "gomoku.selfplay_worker",
        "--weights-path", str(ckpt),
        "--output-dir", str(records_dir),
        "--device", device,
        "--games-per-batch", str(cell["games_per_batch"]),
        "--n-simulations", str(cell["n_simulations"]),
        "--wave-size", str(cell["wave_size"]),
        "--max-plies", str(max_plies),
        "--wave-mode",
    ]
    if compile_model:
        # L05-torch-compile-mps: pass-through to selfplay_worker's existing
        # --compile flag. Worker falls back to uncompiled if torch.compile
        # raises, so this is safe to leave on across diverse cell shapes.
        base_cmd.append("--compile")
    if fp16_eval:
        base_cmd.append("--fp16-eval")

    procs: list[tuple[str, subprocess.Popen, object]] = []
    started = time.perf_counter()
    try:
        for w in range(cell["workers"]):
            wid = f"w{w}"
            cmd = base_cmd + ["--worker-id", wid, "--seed", str(1000 + w)]
            log_path = logs_dir / f"{wid}.log"
            log_f = open(log_path, "w")
            p = subprocess.Popen(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                start_new_session=True,
            )
            procs.append((wid, p, log_f))
            _ACTIVE_PROCS.append((wid, p, log_f))

        # Let them run. Track early exits so we can mark a cell failed.
        deadline = time.perf_counter() + secs_per_cell
        while time.perf_counter() < deadline and not _INTERRUPTED:
            alive = sum(1 for _, p, _ in procs if p.poll() is None)
            if alive == 0:
                break
            time.sleep(1.0)

        # Stop survivors, give them 15s to flush. Skip already-dead procs;
        # killing a zombie raises EPERM on macOS.
        for _, p, _ in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except (ProcessLookupError, PermissionError):
                    pass
        grace_deadline = time.time() + 15
        for _, p, _ in procs:
            remaining = max(0.1, grace_deadline - time.time())
            try:
                p.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        wall_secs = time.perf_counter() - started
        for _, _, log_f in procs:
            try:
                log_f.close()
            except OSError:
                pass
        # Remove entries we own from the global tracker.
        for entry in procs:
            try:
                _ACTIVE_PROCS.remove(entry)
            except ValueError:
                pass

    # If all workers exited before the wall budget, the cell crashed
    # (typical cause: MPS contention with another tenant). Mark it.
    early_exits = sum(
        1 for _, p, _ in procs
        if (p.returncode or 0) != 0 and wall_secs < secs_per_cell - 5
    )
    cell_status = "failed" if early_exits == cell["workers"] else "ok"

    # Count + measure. Worker writes D4-augmented examples (8x per raw ply),
    # so n_examples == record.plies * 8. We track both: total_raw_plies for
    # game-shape diagnostics, total_aug_examples as the throughput numerator.
    total_games = 0
    total_raw_plies = 0
    total_aug_examples = 0
    for game_path in records_dir.rglob("game*.pt"):
        if game_path.suffix != ".pt":
            continue
        try:
            payload = torch.load(game_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        total_games += 1
        total_aug_examples += int(payload.get("n_examples", 0))
        for r in payload.get("records", []):
            total_raw_plies += int(getattr(r, "plies", 0))

    plies_mean = (total_raw_plies / total_games) if total_games else 0.0
    games_per_sec = total_games / wall_secs if wall_secs else 0.0
    aug_pos_per_sec = total_aug_examples / wall_secs if wall_secs else 0.0

    return dict(
        cell_id=cell_id,
        model=cell["model"],
        workers=cell["workers"],
        games_per_batch=cell["games_per_batch"],
        n_simulations=cell["n_simulations"],
        wave_size=cell["wave_size"],
        wall_secs=round(wall_secs, 3),
        total_games=total_games,
        total_aug_examples=total_aug_examples,
        total_raw_plies=total_raw_plies,
        aug_pos_per_sec=round(aug_pos_per_sec, 1),
        games_per_sec=round(games_per_sec, 3),
        plies_mean=round(plies_mean, 2),
        cell_status=cell_status,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# --- Summary I/O --------------------------------------------------------------

SUMMARY_COLS = [
    "cell_id", "model", "workers", "games_per_batch", "n_simulations",
    "wave_size", "wall_secs", "total_games", "total_aug_examples",
    "total_raw_plies", "aug_pos_per_sec", "games_per_sec", "plies_mean",
    "cell_status", "started_at",
]


def load_summary(summary_path: Path) -> list[dict]:
    if not summary_path.exists():
        return []
    rows: list[dict] = []
    with summary_path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append(row)
    return rows


def existing_status_by_id(rows: list[dict]) -> dict[str, str]:
    """cell_id -> cell_status (last write wins)."""
    out: dict[str, str] = {}
    for row in rows:
        out[row["cell_id"]] = row.get("cell_status", "ok")
    return out


def rewrite_summary(summary_path: Path, rows: list[dict]) -> None:
    """Atomic rewrite: write to .tmp then rename."""
    tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in SUMMARY_COLS})
    os.replace(tmp, summary_path)


def append_summary(summary_path: Path, row: dict) -> None:
    is_new = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, delimiter="\t")
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SUMMARY_COLS})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def drop_rows(summary_path: Path, cell_ids: set[str]) -> int:
    """Remove rows with cell_id in cell_ids from summary.tsv. Returns count removed."""
    rows = load_summary(summary_path)
    keep = [r for r in rows if r["cell_id"] not in cell_ids]
    n_removed = len(rows) - len(keep)
    if n_removed:
        rewrite_summary(summary_path, keep)
    return n_removed


def wipe_cell_dir(sweep_dir: Path, cell_id: str) -> None:
    cell_dir = sweep_dir / f"cell_{cell_id}"
    if cell_dir.exists():
        shutil.rmtree(cell_dir, ignore_errors=True)


# --- Snapshots / metadata -----------------------------------------------------

def write_cells_csv(cells_path: Path, cells: list[dict]) -> None:
    fields = ["cell_id"] + CELL_PARAM_FIELDS
    with cells_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow({k: c[k] for k in fields})


def append_metadata(meta_path: Path, args: argparse.Namespace, cells: list[dict]) -> None:
    """Append one metadata block per session; never overwrites previous blocks."""
    import platform
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        porcelain = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
        ).strip()
    except Exception:
        porcelain = ""
    block = [
        f"--- session start {datetime.now(timezone.utc).isoformat(timespec='seconds')} ---",
        f"git commit:    {commit}",
        f"git status:    {'clean' if not porcelain else porcelain}",
        f"hostname:      {platform.node()}",
        f"hardware:      {platform.platform()}; arch={platform.machine()}",
        f"python:        {sys.version.split()[0]}",
        f"torch:         {torch.__version__}",
        f"device:        {args.device}",
        f"compile:       {bool(getattr(args, 'compile_model', False))}",
        f"fp16_eval:     {bool(getattr(args, 'fp16_eval', False))}",
        f"secs_per_cell: {args.secs_per_cell}",
        f"max_plies:     {args.max_plies}",
        f"n_cells:       {len(cells)}",
        f"defaults:      {DEFAULT}",
        f"env-flags:     GOMOKU_DISABLE_NATIVE_MCTS={os.environ.get('GOMOKU_DISABLE_NATIVE_MCTS','')} "
        f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','')}",
        "",
    ]
    with meta_path.open("a") as f:
        f.write("\n".join(block) + "\n")


def update_latest_symlink(out_dir: Path, lane: str | None = None) -> None:
    """Refresh sweep_logs/<latest-symlink> -> out_dir.

    Without --lane: refresh `canonical-sweep-latest`. With --lane:
    refresh `lab-<lane>-latest` only — never clobber
    canonical-sweep-latest, which is the durable pointer to the
    actual canonical sweep."""
    name = f"lab-{lane}-latest" if lane else "canonical-sweep-latest"
    latest = out_dir.parent / name
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
    except OSError:
        return
    try:
        latest.symlink_to(out_dir.name)
    except OSError:
        pass


def resolve_out_dir(arg_value: str) -> Path:
    """Translate `--out-dir latest` (or 'latest', or 'lab-<lane>-latest')
    into the symlink target. Reject empty strings to avoid silently
    writing artifacts into cwd."""
    if not arg_value or not arg_value.strip():
        raise SystemExit("--out-dir must be a non-empty path (or 'latest').")
    sweep_logs = REPO_ROOT / "sweep_logs"
    candidates = []
    if arg_value in ("latest", "LATEST"):
        candidates.append(sweep_logs / "canonical-sweep-latest")
    elif arg_value.startswith("lab-") and arg_value.endswith("-latest"):
        candidates.append(sweep_logs / arg_value)
    if candidates:
        latest = candidates[0]
        if not latest.exists():
            raise SystemExit(
                f"--out-dir {arg_value} requested but {latest} does not exist; "
                "start a new sweep with an explicit --out-dir first."
            )
        return latest.resolve()
    return Path(arg_value).resolve()


# --- Status / planning --------------------------------------------------------

def plan_session(
    cells: list[dict],
    summary_path: Path,
    retry_failed: bool,
) -> tuple[list[dict], dict[str, str], float]:
    """Return (cells_to_run, status_by_id, eta_secs_per_cell_median)."""
    status_by_id = existing_status_by_id(load_summary(summary_path))
    rows = load_summary(summary_path)
    ok_walls = [float(r["wall_secs"]) for r in rows if r.get("cell_status") == "ok" and r.get("wall_secs")]
    median_wall = statistics.median(ok_walls) if ok_walls else 0.0

    to_run: list[dict] = []
    for c in cells:
        status = status_by_id.get(c["cell_id"])
        if status == "ok":
            continue
        if status == "failed" and not retry_failed:
            continue
        to_run.append(c)
    return to_run, status_by_id, median_wall


def print_status(cells: list[dict], summary_path: Path) -> None:
    status_by_id = existing_status_by_id(load_summary(summary_path))
    n_total = len(cells)
    n_ok = sum(1 for c in cells if status_by_id.get(c["cell_id"]) == "ok")
    n_failed = sum(1 for c in cells if status_by_id.get(c["cell_id"]) == "failed")
    n_pending = n_total - n_ok - n_failed
    rows = load_summary(summary_path)
    ok_walls = [float(r["wall_secs"]) for r in rows if r.get("cell_status") == "ok" and r.get("wall_secs")]
    median_wall = statistics.median(ok_walls) if ok_walls else 0.0
    eta_min = (n_pending * (median_wall + 5)) / 60.0  # +5s per-cell setup margin

    print(f"sweep dir:   {summary_path.parent}")
    print(f"cells total: {n_total}")
    print(f"  ok:        {n_ok}")
    print(f"  failed:    {n_failed}  (re-run with --retry-failed)")
    print(f"  pending:   {n_pending}")
    if median_wall:
        print(f"median wall: {median_wall:.1f}s per ok cell")
        print(f"ETA:         ~{eta_min:.1f} min for the {n_pending} pending cell(s)")
    print()
    print("Pending:")
    for c in cells:
        if status_by_id.get(c["cell_id"]) in ("ok",):
            continue
        marker = "FAILED" if status_by_id.get(c["cell_id"]) == "failed" else "      "
        print(f"  {marker}  {c['cell_id']}")


# --- Main ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=str,
                   help="Sweep directory. Pass `latest` to follow the "
                        "sweep_logs/canonical-sweep-latest symlink.")
    p.add_argument("--secs-per-cell", type=int, default=60,
                   help="Wall-clock cap per cell (default 60s smoke-first per "
                        "charter v3). Escalate to 300s only when the smoke "
                        "result is within ~2x of the experimental noise floor.")
    p.add_argument("--max-plies", type=int, default=16)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated cell_id substrings; run only matching cells.")
    p.add_argument("--max-cells", type=int, default=0,
                   help="Hard cap on cells run THIS invocation (0 = no cap).")
    p.add_argument("--max-wall-secs", type=int, default=0,
                   help="Stop after this much wall time and exit cleanly. "
                        "0 = no cap. Use for short top-up sessions.")
    p.add_argument("--retry-failed", action="store_true",
                   help="Re-run cells marked failed (wipes their cell_dir "
                        "and removes their row from summary.tsv first).")
    p.add_argument("--status", action="store_true",
                   help="Print sweep progress + ETA and exit without running.")
    p.add_argument("--cells-from", type=Path, default=None,
                   help="Read cell list from a csv (columns: model, workers, "
                        "games_per_batch, n_simulations, wave_size). Replaces "
                        "the built-in canonical 23-cell list. Use for ad-hoc "
                        "lab lanes per wiki/topics/perf-lab-charter.md.")
    p.add_argument("--lane", type=str, default=None,
                   help="Lane label for autonomous-lab dispatch. Stamps the "
                        "latest-symlink and the cell metadata.")
    p.add_argument("--compile", action="store_true", dest="compile_model",
                   help="Pass --compile through to every selfplay_worker in "
                        "this sweep (torch.compile the eval-only model). "
                        "Worker falls back to uncompiled if compile raises. "
                        "Applies uniformly to all cells in this invocation; "
                        "L05-torch-compile-mps. Do NOT use with the trainer.")
    p.add_argument("--fp16-eval", action="store_true",
                   help="L06 perf-lab lane. Pass --fp16-eval through to every "
                        "spawned selfplay_worker so the eval model runs in "
                        "torch.float16 (inputs cast inside make_torch_evaluator, "
                        "outputs cast back to fp32 before host transfer). "
                        "Default off. See wiki/ops/perf-queue.md L06-fp16-eval.")
    args = p.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.tsv"
    cells_path = out_dir / "cells.csv"
    meta_path = out_dir / "metadata.txt"

    # Cell list resolution (in order of precedence):
    #  1. --cells-from explicit: load from the given path, rewrite cells.csv.
    #  2. cells.csv already in out_dir (a resumed lane): trust the on-disk
    #     list as the source of truth so --status doesn't clobber it.
    #  3. fresh canonical sweep: use build_cells(), write the snapshot.
    if args.cells_from:
        cells = load_cells_csv(args.cells_from.resolve())
        write_cells_csv(cells_path, cells)
    elif cells_path.exists():
        cells = load_cells_csv(cells_path)
    else:
        cells = build_cells()
        write_cells_csv(cells_path, cells)

    if args.status:
        print_status(cells, summary_path)
        return

    lock_path = acquire_lock(out_dir)
    update_latest_symlink(out_dir, lane=args.lane)
    append_metadata(meta_path, args, cells)
    _install_signal_handlers()

    try:
        # Handle --retry-failed: drop failed rows + wipe their cell dirs.
        if args.retry_failed:
            failed_ids = {
                cid for cid, st in existing_status_by_id(load_summary(summary_path)).items()
                if st == "failed"
            }
            n_dropped = drop_rows(summary_path, failed_ids)
            for cid in failed_ids:
                wipe_cell_dir(out_dir, cid)
            print(f"[retry] cleared {n_dropped} failed row(s) and their cell_dir(s)")

        # Stage checkpoints once per model.
        needed_models = sorted({c["model"] for c in cells})
        for m in needed_models:
            ckpt = out_dir / "checkpoints" / f"{m}.pt"
            stage_checkpoint(m, ckpt)
            print(f"[stage] {m} -> {ckpt}")

        to_run, status_by_id, median_wall = plan_session(cells, summary_path, args.retry_failed)
        if args.only:
            needles = [s.strip() for s in args.only.split(",") if s.strip()]
            to_run = [c for c in to_run if any(n in c["cell_id"] for n in needles)]

        n_total = len(cells)
        n_ok = sum(1 for c in cells if status_by_id.get(c["cell_id"]) == "ok")
        n_failed = sum(1 for c in cells if status_by_id.get(c["cell_id"]) == "failed" and not args.retry_failed)
        eta_min = (len(to_run) * (median_wall + 5)) / 60.0 if median_wall else 0.0
        print(
            f"[plan] {n_total} cells total | {n_ok} ok | {n_failed} failed-skipped | "
            f"{len(to_run)} to run this session"
            + (f" | ETA ~{eta_min:.1f} min" if median_wall else "")
        )
        if args.max_wall_secs:
            print(f"[plan] wall-budget {args.max_wall_secs}s; will exit cleanly when hit")

        ran = 0
        session_started = time.perf_counter()
        for cell in to_run:
            if _INTERRUPTED:
                print("[stop] interrupted; exiting before next cell")
                break
            if args.max_cells and ran >= args.max_cells:
                print(f"[stop] hit --max-cells={args.max_cells}")
                break
            if args.max_wall_secs and (time.perf_counter() - session_started) >= args.max_wall_secs:
                print(f"[stop] hit --max-wall-secs={args.max_wall_secs}")
                break

            print(f"[run ] {cell['cell_id']} model={cell['model']} W={cell['workers']} "
                  f"G={cell['games_per_batch']} S={cell['n_simulations']} V={cell['wave_size']}")
            row = run_cell(
                cell, out_dir,
                secs_per_cell=args.secs_per_cell,
                max_plies=args.max_plies,
                device=args.device,
                compile_model=args.compile_model,
                fp16_eval=args.fp16_eval,
            )
            if _INTERRUPTED:
                print(f"[drop] {cell['cell_id']} interrupted mid-cell; not recording row")
                break
            append_summary(summary_path, row)
            elapsed_min = (time.perf_counter() - session_started) / 60.0
            tag = "DONE" if row["cell_status"] == "ok" else "FAIL"
            print(f"[{tag}] {row['cell_id']} aug_pos/s={row['aug_pos_per_sec']:.0f} "
                  f"games/s={row['games_per_sec']:.2f} plies_mean={row['plies_mean']:.1f} "
                  f"games={row['total_games']}  (session {elapsed_min:.1f}m)")
            ran += 1

        elapsed_min = (time.perf_counter() - session_started) / 60.0
        print(f"[end ] ran {ran} cells this session in {elapsed_min:.1f} min")
        print(f"       summary at {summary_path}")
        print(f"       resume with: python scripts/canonical_sweep.py --out-dir {out_dir}")
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    main()
