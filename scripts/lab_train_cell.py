"""Live-training cell driver for the R-TRAIN-* perf reference family.

Spawns one `gomoku.train` trainer + N `gomoku.selfplay_worker` children
against a fresh fused checkpoint, lets them warm up, opens a measurement
window, then SIGTERMs the whole process group and harvests the trainer
log for `^epoch (\\d+)/` lines to compute `epochs_per_sec`. Counts the
worker game records under the records dir for `games_per_sec` /
`aug_pos_per_sec` and parses `(... train=Xs)` from the trainer prints
for `trainer_step_s_p50`.

Schema is a superset of `scripts/canonical_sweep.py`'s `summary.tsv`
(canonical columns first, then training-specific columns) so the
existing plotter and downstream tooling can read both. See
`wiki/topics/perf-lab-charter.md` ("R-TRAIN-*") and the L12 spec in
`wiki/ops/perf-queue.md`.

Resumability contract per `wiki/topics/perf-lab-session-runbook.md`:
  - cell_id derived from params (model / workers / G / S / V / ema_tau /
    grad_accum / wave_mode), never index-based.
  - cells with status=ok skipped on resume; status=failed skipped unless
    --retry-failed (which removes the row + wipes the cell_dir).
  - --status prints done/failed/pending/ETA without running.
  - .sweep.lock file holds PID; stale locks reclaimed automatically.
  - SIGINT/SIGTERM kill live children via start_new_session + killpg,
    don't record the interrupted cell.

Per-cell artifacts under <out-dir>/cell_<cell-id>/:
  trainer.log, worker-NN.log, metadata.txt, checkpoints/, records/
"""

from __future__ import annotations

import argparse
import csv
import errno
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Defer the torch / model imports until we actually need them so --help
# and --dry-run stay fast (and importable on a host without torch in this
# venv, which the L12 functional smoke wants).


# --- Cell shape ----------------------------------------------------------------

CANONICAL_PARAM_FIELDS = [
    "model", "workers", "games_per_batch", "n_simulations", "wave_size",
]
TRAIN_PARAM_FIELDS = [
    "ema_tau", "grad_accum_steps", "wave_mode", "sgd_per_position",
    "batch_size",
]
CELL_PARAM_FIELDS = CANONICAL_PARAM_FIELDS + TRAIN_PARAM_FIELDS


def cell_id_of(cell: dict) -> str:
    """Deterministic cell_id from params. Mirrors canonical_sweep's stem
    so a train cell's id is greppably related to its generator cousin,
    then adds the train-specific tail."""
    tail = (
        f"_EMA{int(round(cell['ema_tau']*100)):02d}"
        f"_GA{cell['grad_accum_steps']:02d}"
        f"_WM{1 if cell['wave_mode'] else 0}"
        f"_B{cell['batch_size']:03d}"
    )
    return (
        f"train_{cell['model']}_W{cell['workers']:02d}"
        f"_G{cell['games_per_batch']:02d}_S{cell['n_simulations']:03d}"
        f"_V{cell['wave_size']:03d}" + tail
    )


# --- Lock + signals ------------------------------------------------------------

LOCK_NAME = ".sweep.lock"
_ACTIVE_PGIDS: list[int] = []
_INTERRUPTED = False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno != errno.ESRCH
    return True


def acquire_lock(out_dir: Path) -> Path:
    lock_path = out_dir / LOCK_NAME
    if lock_path.exists():
        try:
            other = int(lock_path.read_text().strip())
        except (ValueError, OSError):
            other = -1
        if _pid_alive(other) and other != os.getpid():
            raise SystemExit(
                f"another lab_train_cell is running on this dir (PID {other}). "
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


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        global _INTERRUPTED
        _INTERRUPTED = True
        for pgid in list(_ACTIVE_PGIDS):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        sys.stderr.write(
            f"\n[signal] received {signum}; SIGTERM'd workers, current cell unrecorded\n"
        )
        sys.stderr.flush()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# --- Pre-flight ----------------------------------------------------------------

def preflight_idle(out_dir: Path) -> None:
    """Abort if another tenant is using MPS (selfplay_worker / gomoku.train /
    run_sweep / eval_worker) outside of our out_dir. Mirrors session-runbook
    section 1."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-fl", r"selfplay_worker|gomoku\.train|run_sweep|eval_worker"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return  # nothing matched
    foreign = []
    for line in out.splitlines():
        if str(out_dir) in line or "lab_train_cell" in line:
            continue
        if str(os.getpid()) in line.split()[:1]:
            continue
        foreign.append(line)
    if foreign:
        raise SystemExit(
            "preflight: another tenant is on the box; refusing to start.\n"
            + "\n".join(f"  {ln}" for ln in foreign)
            + "\n(use `pkill -f ...` or wait, then retry)"
        )


# --- Trainer-log parsing -------------------------------------------------------

EPOCH_RE = re.compile(r"^epoch (\d+)/")
# Trainer-print tail: "(epoch_s: gen=X.Ys train=Y.Ys)". Pull train= for per-step.
TRAIN_TAIL_RE = re.compile(r"train=([\d.]+)s\)")
STEPS_RE = re.compile(r"\bsteps=(\d+)")
# Cumulative counters on each epoch line. The trainer is the source of truth
# for R-TRAIN-* throughput because it ingests + deletes records as it goes,
# so end-of-window file counts undercount drastically.
GAMES_RE = re.compile(r"\bgames=(\d+)")
BUF_RE = re.compile(r"\bbuf=(\d+)")
PLIES_RE = re.compile(r"\bplies=([\d.]+)")
EPOCH_WALL_RE = re.compile(r"\(([\d.]+)s:")


def parse_trainer_log(log_path: Path, warmup_secs: float, measurement_secs: float
                      ) -> dict:
    """Read trainer.log; return parsed metrics for the measurement window.

    The window is (warmup_secs, warmup_secs+measurement_secs) seconds after
    the FIRST epoch line. epochs_per_sec = (last_epoch - first_epoch_in_window)
    / measurement_secs. trainer_step_s_p50 = median(train=Xs / steps=N) for
    epochs entirely inside the window.
    """
    metrics = dict(
        first_epoch=None, last_epoch=None, epochs_in_window=0,
        epochs_per_sec=0.0, trainer_step_s_p50=0.0,
        first_epoch_t=None, last_epoch_t=None,
        # Derived from the cumulative games= and buf= counters on each epoch
        # line plus the per-epoch wall (Xs: in the tail). These are the
        # authoritative R-TRAIN-* throughput numbers because the trainer
        # ingests and deletes worker game*.pt files as it goes, so the
        # end-of-window record-file count is always near zero.
        games_per_sec_trainer=0.0,
        aug_pos_per_sec_trainer=0.0,
        plies_mean_trainer=0.0,
        total_games_trainer=0,
        total_aug_trainer=0,
    )
    if not log_path.exists():
        return metrics
    # mtime of the first epoch line is our t0; for sub-second resolution
    # we read line-by-line and grab wall-clock at line-arrival time. But
    # the log was written in the past so mtimes don't help — we trust the
    # epoch sequence and the wall-time we observed running it. Since the
    # caller drove the wall clock (warmup_secs + measurement_secs), we
    # filter epochs by integer index assuming uniform spacing isn't safe.
    # Instead: take all epochs; epochs_per_sec uses the COUNT minus 1
    # divided by measurement_secs only if we have >= 2 epochs in window.
    # The driver passes us the actual epoch-time line pairs via stat().st_mtime
    # — but that's also noisy. So we use the simpler scheme the L12 spec
    # asks for: (last - first) / measurement_secs across the whole post-
    # warmup window. To approximate "post-warmup", we drop the first
    # ceil(warmup_secs / median_epoch_s) epochs; if we have fewer epochs
    # than that, return zeros (cell didn't run long enough).
    epoch_ids: list[int] = []
    train_s_per_step: list[float] = []
    # Per-epoch (cumulative_games, cumulative_buf, plies_mean, epoch_wall_s).
    # Populated when the line carries all four; unparseable lines are skipped
    # so an older trainer with a different log shape just leaves these empty
    # and we fall through to the count_records-based numbers.
    epoch_rows: list[tuple[int, int, float, float]] = []
    with log_path.open() as f:
        for line in f:
            m = EPOCH_RE.match(line)
            if not m:
                continue
            epoch_ids.append(int(m.group(1)))
            tm = TRAIN_TAIL_RE.search(line)
            sm = STEPS_RE.search(line)
            if tm and sm:
                steps = int(sm.group(1))
                if steps > 0:
                    train_s_per_step.append(float(tm.group(1)) / steps)
            gm = GAMES_RE.search(line)
            bm = BUF_RE.search(line)
            pm = PLIES_RE.search(line)
            wm = EPOCH_WALL_RE.search(line)
            if gm and bm and pm and wm:
                epoch_rows.append((
                    int(gm.group(1)),
                    int(bm.group(1)),
                    float(pm.group(1)),
                    float(wm.group(1)),
                ))
    metrics["epochs_in_window"] = len(epoch_ids)
    if len(epoch_ids) >= 2:
        # Drop epochs that fell inside the warmup. We approximate by
        # using the trainer's own per-epoch wall (epoch_s) inferred from
        # the train= numbers + an implicit gen= which we don't capture
        # cleanly — instead, just use the simple ratio:
        # epochs_per_sec = (n_epochs - 1) / measurement_secs
        # IF n_epochs > epochs_warmup_estimate, else 0.
        # Heuristic: assume warmup-secs / (epoch_s) epochs lost to warmup.
        # epoch_s ~ train_s_per_step median * steps + gen_s; we don't have
        # gen_s, so take a conservative estimate: warmup_epochs = max(1,
        # floor(n_epochs * warmup_secs / (warmup_secs + measurement_secs))).
        total_t = max(1.0, warmup_secs + measurement_secs)
        warmup_epochs = max(1, int(len(epoch_ids) * warmup_secs / total_t))
        post_warmup = epoch_ids[warmup_epochs:]
        if len(post_warmup) >= 2:
            metrics["first_epoch"] = post_warmup[0]
            metrics["last_epoch"] = post_warmup[-1]
            metrics["epochs_per_sec"] = (
                (post_warmup[-1] - post_warmup[0]) / measurement_secs
            )
        else:
            metrics["first_epoch"] = epoch_ids[0]
            metrics["last_epoch"] = epoch_ids[-1]
            metrics["epochs_per_sec"] = (
                (epoch_ids[-1] - epoch_ids[0]) / (total_t)
            )
    if train_s_per_step:
        metrics["trainer_step_s_p50"] = float(statistics.median(train_s_per_step))
    # Compute the trainer-log-derived throughput rates. Walk epoch_rows;
    # cumulative time is the sum of per-epoch walls. The "measurement window"
    # is whatever fraction of the run remains after `warmup_secs`. Use the
    # first epoch whose end-time exceeds warmup as the window start.
    if len(epoch_rows) >= 2:
        cum_t = 0.0
        per_epoch_endtimes: list[float] = []
        for _g, _b, _p, w in epoch_rows:
            cum_t += w
            per_epoch_endtimes.append(cum_t)
        start_idx = None
        for i, t in enumerate(per_epoch_endtimes):
            if t > warmup_secs:
                start_idx = i
                break
        if start_idx is not None and start_idx < len(epoch_rows) - 1:
            g0, b0, _, _ = epoch_rows[start_idx]
            gN, bN, _, _ = epoch_rows[-1]
            t0 = per_epoch_endtimes[start_idx]
            tN = per_epoch_endtimes[-1]
            span = tN - t0
            if span > 0:
                games_delta = max(0, gN - g0)
                buf_delta = max(0, bN - b0)
                metrics["games_per_sec_trainer"] = games_delta / span
                metrics["aug_pos_per_sec_trainer"] = buf_delta / span
                metrics["total_games_trainer"] = games_delta
                metrics["total_aug_trainer"] = buf_delta
                # plies_mean across post-warmup epochs.
                post = epoch_rows[start_idx:]
                plies_vals = [r[2] for r in post]
                if plies_vals:
                    metrics["plies_mean_trainer"] = (
                        sum(plies_vals) / len(plies_vals)
                    )
    return metrics


# --- Records harvest -----------------------------------------------------------

def count_records(records_dir: Path) -> tuple[int, int, int]:
    """Return (total_games, total_aug_examples, total_raw_plies). Loads each
    game*.pt only for its scalar metadata; payloads can be large but a
    measurement window will rarely produce more than ~100 files."""
    import torch  # lazy
    total_games = 0
    total_aug = 0
    total_plies = 0
    for game_path in records_dir.rglob("game*.pt"):
        if game_path.suffix != ".pt":
            continue
        try:
            payload = torch.load(game_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        total_games += 1
        total_aug += int(payload.get("n_examples", 0))
        for r in payload.get("records", []):
            total_plies += int(getattr(r, "plies", 0))
    return total_games, total_aug, total_plies


# --- Cell runner ---------------------------------------------------------------

def stage_checkpoint(model_size: str, out_path: Path, stem_padding: int = 1) -> None:
    if out_path.exists():
        return
    import torch  # noqa: F401  (ensure available before import_models)
    from gomoku.model import build_model, save_checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = build_model(model_size, stem_padding=stem_padding)
    m.eval()
    save_checkpoint(str(out_path), m, epoch=0)


def build_trainer_cmd(cell: dict, dirs: dict) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "gomoku.train",
        "--size", cell["model"],
        "--stem-padding", "1",
        "--epochs", "1000000",  # effectively unbounded; we SIGTERM
        "--games-per-epoch", str(cell["workers"] * cell["games_per_batch"]),
        "--n-simulations", str(cell["n_simulations"]),
        "--wave-size", str(cell["wave_size"]),
        "--batch-size", str(cell["batch_size"]),
        "--worker-input-dir", str(dirs["records"]),
        "--worker-weights-path", str(dirs["worker_weights"]),
        "--worker-min-games", str(cell["workers"] * cell["games_per_batch"]),
        "--checkpoint-dir", str(dirs["checkpoints"]),
        # save-every must be 1: gomoku/train.py:1220 publishes worker_weights.pt
        # inside the save-every block. With a high value, workers stay on v0,
        # the trainer waits for v1+ games that never come, and only one epoch
        # completes regardless of the measurement window. The buffer write
        # (the expensive part) stays gated by save-buffer-every which we leave
        # high so the 1.4 GB latest.pt isn't rewritten every cell.
        "--save-every", "1",
        "--save-buffer-every", "1000000",
        "--keep-last-n", "1",
        "--no-eval",
        "--no-wandb",
        "--min-training-steps", "16",
    ]
    if cell["sgd_per_position"] > 0:
        cmd += ["--sgd-per-position", str(cell["sgd_per_position"])]
    if cell["wave_mode"]:
        cmd += [
            "--wave-mode",
            "--wave-workers", str(cell["workers"]),
            "--wave-games-per-worker", str(cell["games_per_batch"]),
        ]
    if cell["ema_tau"] > 0:
        cmd += ["--ema-tau", str(cell["ema_tau"])]
    if cell["grad_accum_steps"] > 1:
        cmd += ["--grad-accum-steps", str(cell["grad_accum_steps"])]
    return cmd


def build_worker_cmd(cell: dict, dirs: dict, worker_id: str, seed: int) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "gomoku.selfplay_worker",
        "--weights-path", str(dirs["worker_weights"]),
        "--output-dir", str(dirs["records"]),
        "--worker-id", worker_id,
        "--device", "mps",
        "--games-per-batch", str(cell["games_per_batch"]),
        "--n-simulations", str(cell["n_simulations"]),
        "--wave-size", str(cell["wave_size"]),
        "--seed", str(seed),
    ]
    if cell["wave_mode"]:
        cmd += ["--wave-mode"]
    if cell.get("evaluator", "torch") != "torch":
        cmd += ["--evaluator", cell["evaluator"]]
        if cell["evaluator"] == "coreml":
            cmd += ["--coreml-compute-units", cell.get("coreml_compute_units", "CPU_AND_NE")]
    return cmd


def run_cell(cell: dict, sweep_dir: Path, warmup_secs: int,
             measurement_secs: int, device: str, dry_run: bool = False
             ) -> dict:
    cell_id = cell["cell_id"]
    cell_dir = sweep_dir / f"cell_{cell_id}"
    dirs = {
        "cell": cell_dir,
        "records": cell_dir / "records",
        "checkpoints": cell_dir / "checkpoints",
        "logs": cell_dir / "logs",
        "worker_weights": cell_dir / "checkpoints" / "worker_weights.pt",
    }
    for d in (dirs["records"], dirs["checkpoints"], dirs["logs"]):
        d.mkdir(parents=True, exist_ok=True)

    trainer_cmd = build_trainer_cmd(cell, dirs)
    worker_cmds = [
        build_worker_cmd(cell, dirs, f"w{i}", seed=1000 + i)
        for i in range(cell["workers"])
    ]

    if dry_run:
        print(f"[dry-run] cell {cell_id}")
        print("[dry-run] trainer:", " ".join(trainer_cmd))
        for wc in worker_cmds:
            print("[dry-run] worker :", " ".join(wc))
        return dict(cell_id=cell_id, cell_status="dry", **{k: cell[k] for k in CELL_PARAM_FIELDS})

    # Stage the initial worker_weights file so workers don't spin in poll-load.
    stage_checkpoint(cell["model"], dirs["worker_weights"], stem_padding=1)

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Spawn trainer.
    trainer_log = open(dirs["logs"] / "trainer.log", "w")
    trainer = subprocess.Popen(
        trainer_cmd, stdout=trainer_log, stderr=subprocess.STDOUT,
        env=env, cwd=str(REPO_ROOT), start_new_session=True,
    )
    _ACTIVE_PGIDS.append(os.getpgid(trainer.pid))

    # Give trainer ~2s to publish initial weights file.
    time.sleep(2.0)

    workers: list[tuple[str, subprocess.Popen, object]] = []
    for i, wc in enumerate(worker_cmds):
        wid = f"w{i}"
        wlog = open(dirs["logs"] / f"worker-{i:02d}.log", "w")
        p = subprocess.Popen(
            wc, stdout=wlog, stderr=subprocess.STDOUT,
            env=env, cwd=str(REPO_ROOT), start_new_session=True,
        )
        _ACTIVE_PGIDS.append(os.getpgid(p.pid))
        workers.append((wid, p, wlog))

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    deadline = t0 + warmup_secs + measurement_secs
    try:
        while time.perf_counter() < deadline and not _INTERRUPTED:
            if trainer.poll() is not None:
                break
            time.sleep(1.0)
    finally:
        wall_secs = time.perf_counter() - t0
        # SIGTERM each pgroup, then SIGKILL after grace.
        for pgid in list(_ACTIVE_PGIDS):
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        grace_deadline = time.time() + 15
        for proc in [trainer] + [p for _, p, _ in workers]:
            remaining = max(0.1, grace_deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        trainer_log.close()
        for _, _, wf in workers:
            try:
                wf.close()
            except OSError:
                pass
        _ACTIVE_PGIDS.clear()

    if _INTERRUPTED:
        return dict(cell_id=cell_id, cell_status="interrupted")

    # Harvest metrics.
    parsed = parse_trainer_log(
        dirs["logs"] / "trainer.log",
        warmup_secs=warmup_secs,
        measurement_secs=measurement_secs,
    )
    # The trainer ingests + deletes worker game*.pt files as it goes, so the
    # on-disk record count at SIGTERM is near-zero — useless for R-TRAIN-*
    # throughput. Prefer the trainer-log-derived numbers (cumulative games=N
    # and buf=N counters on each epoch line) when available; fall back to
    # count_records only if the trainer log is empty (e.g. trainer crashed
    # before its first epoch, or we ran without a trainer).
    if parsed["games_per_sec_trainer"] > 0:
        total_games = parsed["total_games_trainer"]
        total_aug = parsed["total_aug_trainer"]
        total_plies = int(parsed["plies_mean_trainer"] * total_games) if total_games else 0
        games_per_sec = parsed["games_per_sec_trainer"]
        aug_pos_per_sec = parsed["aug_pos_per_sec_trainer"]
        plies_mean = parsed["plies_mean_trainer"]
    else:
        total_games, total_aug, total_plies = count_records(dirs["records"])
        games_per_sec = total_games / wall_secs if wall_secs else 0.0
        aug_pos_per_sec = total_aug / wall_secs if wall_secs else 0.0
        plies_mean = (total_plies / total_games) if total_games else 0.0

    # A cell "failed" iff trainer never emitted a single epoch line.
    cell_status = "ok" if parsed["epochs_in_window"] >= 1 else "failed"

    return dict(
        cell_id=cell_id,
        model=cell["model"],
        workers=cell["workers"],
        games_per_batch=cell["games_per_batch"],
        n_simulations=cell["n_simulations"],
        wave_size=cell["wave_size"],
        wall_secs=round(wall_secs, 3),
        total_games=total_games,
        total_aug_examples=total_aug,
        total_raw_plies=total_plies,
        aug_pos_per_sec=round(aug_pos_per_sec, 1),
        games_per_sec=round(games_per_sec, 3),
        plies_mean=round(plies_mean, 2),
        cell_status=cell_status,
        started_at=started_at,
        # train-specific extras:
        epochs_per_sec=round(parsed["epochs_per_sec"], 4),
        trainer_step_s_p50=round(parsed["trainer_step_s_p50"], 4),
        epochs_in_window=parsed["epochs_in_window"],
        ema_tau=cell["ema_tau"],
        grad_accum_steps=cell["grad_accum_steps"],
        wave_mode=int(cell["wave_mode"]),
        warmup_secs=warmup_secs,
        measurement_secs=measurement_secs,
    )


# --- Summary I/O ---------------------------------------------------------------

SUMMARY_COLS = [
    "cell_id", "model", "workers", "games_per_batch", "n_simulations",
    "wave_size", "wall_secs", "total_games", "total_aug_examples",
    "total_raw_plies", "aug_pos_per_sec", "games_per_sec", "plies_mean",
    "cell_status", "started_at",
    # train-specific extras (after canonical block):
    "epochs_per_sec", "trainer_step_s_p50", "epochs_in_window",
    "ema_tau", "grad_accum_steps", "wave_mode",
    "warmup_secs", "measurement_secs",
]


def load_summary(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def existing_status_by_id(rows: list[dict]) -> dict[str, str]:
    return {r["cell_id"]: r.get("cell_status", "ok") for r in rows}


def rewrite_summary(p: Path, rows: list[dict]) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in SUMMARY_COLS})
    os.replace(tmp, p)


def append_summary(p: Path, row: dict) -> None:
    is_new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, delimiter="\t")
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SUMMARY_COLS})
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def drop_rows(p: Path, ids: set[str]) -> int:
    rows = load_summary(p)
    keep = [r for r in rows if r["cell_id"] not in ids]
    n = len(rows) - len(keep)
    if n:
        rewrite_summary(p, keep)
    return n


def wipe_cell_dir(sweep_dir: Path, cell_id: str) -> None:
    d = sweep_dir / f"cell_{cell_id}"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


# --- Metadata ------------------------------------------------------------------

def append_metadata(meta_path: Path, args: argparse.Namespace, cell: dict) -> None:
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
    try:
        import torch  # type: ignore
        torch_ver = torch.__version__
    except ImportError:
        torch_ver = "unavailable"
    block = [
        f"--- {datetime.now(timezone.utc).isoformat(timespec='seconds')} cell={cell['cell_id']} ---",
        f"git commit:        {commit}",
        f"git status:        {'clean' if not porcelain else porcelain}",
        f"hostname:          {platform.node()}",
        f"hardware:          {platform.platform()}; arch={platform.machine()}",
        f"python:            {sys.version.split()[0]}",
        f"torch:             {torch_ver}",
        f"device:            {args.device}",
        f"warmup_secs:       {args.warmup_secs}",
        f"measurement_secs:  {args.measurement_secs}",
        f"lane:              {args.lane}",
        f"env-flags:         GOMOKU_DISABLE_NATIVE_MCTS={os.environ.get('GOMOKU_DISABLE_NATIVE_MCTS','')} "
        f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','')}",
        "",
    ]
    with meta_path.open("a") as f:
        f.write("\n".join(block) + "\n")


# --- CLI -----------------------------------------------------------------------

def make_cell_from_args(args: argparse.Namespace) -> dict:
    cell = dict(
        model=args.model,
        workers=args.workers,
        games_per_batch=args.games_per_batch,
        n_simulations=args.n_simulations,
        wave_size=args.wave_size,
        ema_tau=args.ema_tau,
        grad_accum_steps=args.grad_accum_steps,
        wave_mode=args.wave_mode,
        sgd_per_position=args.sgd_per_position,
        batch_size=args.batch_size,
        evaluator=args.evaluator,
        coreml_compute_units=args.coreml_compute_units,
    )
    if args.cell_id:
        cell["cell_id"] = args.cell_id
    else:
        cell["cell_id"] = cell_id_of(cell)
    return cell


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Sweep directory (artifacts root). Created if missing.")
    p.add_argument("--lane", type=str, default=None,
                   help="Lane label for autonomous dispatch (e.g. L10, L11, L09).")
    p.add_argument("--cell-id", type=str, default=None,
                   help="Override the derived cell_id (rare; useful for tagging "
                        "warmup/measure split cells).")
    # Cell shape — canonical
    p.add_argument("--model", type=str, default="small")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--games-per-batch", type=int, default=8)
    p.add_argument("--n-simulations", type=int, default=400)
    p.add_argument("--wave-size", type=int, default=64)
    # Cell shape — train-specific (WL5 production defaults)
    p.add_argument("--ema-tau", type=float, default=0.99)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--wave-mode", action="store_true", default=True,
                   help="Distributed wave-lockstep mode (WL5 default ON).")
    p.add_argument("--no-wave-mode", dest="wave_mode", action="store_false")
    p.add_argument("--sgd-per-position", type=float, default=0.0025)
    p.add_argument("--batch-size", type=int, default=512)
    # Evaluator backend for the self-play workers. The trainer always uses
    # torch on --device; this controls what backend the workers' eval model
    # runs on. coreml routes inference through CPU_AND_NE (the ANE) so the
    # MPS GPU is left free for the trainer — the L09 architectural lever.
    p.add_argument("--evaluator", type=str, default="torch",
                   choices=["torch", "coreml"],
                   help="Worker eval backend (L09 R-TRAIN-ANE). Default torch.")
    p.add_argument("--coreml-compute-units", type=str, default="CPU_AND_NE",
                   choices=["CPU_AND_NE", "CPU_AND_GPU", "ALL", "CPU_ONLY"],
                   help="Core ML compute-units routing when --evaluator=coreml. "
                        "Default CPU_AND_NE (ANE-first; matches L09 spec).")
    # Window
    p.add_argument("--warmup-secs", type=int, default=30)
    p.add_argument("--measurement-secs", type=int, default=60)
    p.add_argument("--device", type=str, default="mps")
    # Resumability / utility
    p.add_argument("--retry-failed", action="store_true",
                   help="Re-run cells marked failed (wipe cell_dir + drop row).")
    p.add_argument("--status", action="store_true",
                   help="Print summary status and ETA; do not run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the Popen invocations and exit; spawn nothing.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the pgrep idle check. Use only when you know "
                        "you're the only tenant.")
    args = p.parse_args()

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.tsv"
    meta_path = out_dir / "metadata.txt"

    cell = make_cell_from_args(args)

    if args.status:
        rows = load_summary(summary_path)
        status_by_id = existing_status_by_id(rows)
        st = status_by_id.get(cell["cell_id"], "pending")
        print(f"sweep dir:   {out_dir}")
        print(f"cell_id:     {cell['cell_id']}")
        print(f"status:      {st}")
        ok = sum(1 for s in status_by_id.values() if s == "ok")
        failed = sum(1 for s in status_by_id.values() if s == "failed")
        print(f"summary:     {len(rows)} rows ({ok} ok, {failed} failed)")
        return

    if args.dry_run:
        # No lock / preflight needed for dry-run.
        row = run_cell(cell, out_dir,
                       warmup_secs=args.warmup_secs,
                       measurement_secs=args.measurement_secs,
                       device=args.device, dry_run=True)
        print(f"[dry-run] cell_id={row['cell_id']} status={row.get('cell_status')}")
        return

    if not args.skip_preflight:
        preflight_idle(out_dir)

    if args.retry_failed:
        rows = load_summary(summary_path)
        failed_ids = {r["cell_id"] for r in rows if r.get("cell_status") == "failed"}
        if cell["cell_id"] in failed_ids:
            drop_rows(summary_path, {cell["cell_id"]})
            wipe_cell_dir(out_dir, cell["cell_id"])
            print(f"[retry] cleared failed row + cell_dir for {cell['cell_id']}")

    rows = load_summary(summary_path)
    status_by_id = existing_status_by_id(rows)
    existing = status_by_id.get(cell["cell_id"])
    if existing == "ok":
        print(f"[skip] {cell['cell_id']} already ok in {summary_path}; "
              "delete the row or use a different cell-id to re-run")
        return
    if existing == "failed":
        print(f"[skip] {cell['cell_id']} previously failed; pass --retry-failed to re-run")
        return

    lock_path = acquire_lock(out_dir)
    _install_signal_handlers()
    append_metadata(meta_path, args, cell)
    try:
        print(f"[run ] {cell['cell_id']} model={cell['model']} W={cell['workers']} "
              f"G={cell['games_per_batch']} S={cell['n_simulations']} V={cell['wave_size']} "
              f"ema={cell['ema_tau']} ga={cell['grad_accum_steps']} wave={int(cell['wave_mode'])} "
              f"warmup={args.warmup_secs}s measure={args.measurement_secs}s")
        row = run_cell(cell, out_dir,
                       warmup_secs=args.warmup_secs,
                       measurement_secs=args.measurement_secs,
                       device=args.device)
        if _INTERRUPTED or row.get("cell_status") == "interrupted":
            print("[drop] interrupted mid-cell; not recording row")
            return
        append_summary(summary_path, row)
        tag = "DONE" if row["cell_status"] == "ok" else "FAIL"
        print(f"[{tag}] {row['cell_id']} epochs/s={row['epochs_per_sec']} "
              f"games/s={row['games_per_sec']} aug/s={row['aug_pos_per_sec']:.0f} "
              f"trainer_step_s_p50={row['trainer_step_s_p50']} "
              f"epochs_in_window={row['epochs_in_window']}")
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    main()
