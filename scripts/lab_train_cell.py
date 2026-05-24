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
# Full-trajectory extras (epoch-capped mode). The wall/gen/train split lives
# in the tail "(Xs: gen=Ys train=Zs)"; gen= is only meaningful there, so we
# anchor it inside the parens. "new=" is the per-cycle new-games counter and
# "tile=" is the wave's ingested-game count (wave[v.. tile=N ..]); both are the
# load-bearing runaway signals from perf-bench-vs-real-training-cost.md.
GEN_TAIL_RE = re.compile(r":\s*gen=([\d.]+)s")
NEW_RE = re.compile(r"\bnew=(\d+)")
TILE_RE = re.compile(r"\btile=(\d+)")
# elo= trails the line when eval_worker forwarded a model_elo this cycle.
ELO_RE = re.compile(r"\belo=(-?[\d.]+)")
# pl=/vl= are the policy/value losses; we sniff them only to count NaNs.
PL_RE = re.compile(r"\bpl=(\S+)")
VL_RE = re.compile(r"\bvl=(\S+)")


def _slope(xs: list[float]) -> float:
    """Least-squares slope of xs against index (0..n-1). Returns 0.0 for <2
    points or a degenerate fit. Used for the steps/epoch and wall/epoch growth
    trend that perf-bench-vs-real-training-cost.md lane 1 asks for (is the tile
    bounded or diverging?)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(xs) / n
    num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(xs))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def _is_nan_token(tok: str) -> bool:
    return tok.strip().lower() in ("nan", "-nan", "inf", "-inf")


def postfill_window(rows: list[dict], *, fill_frac: float, capacity: int,
                    prefill_epochs: int) -> tuple[int, str]:
    """Pick the index of the FIRST epoch that counts as "post buffer-fill" for
    warm-buffer measurement, and the reason we picked it.

    The post-fill boundary is the EARLIER-satisfied of two conditions, so a cell
    is honest whether or not it has a known capacity:
      - buffer-fill: first epoch whose `buf` >= fill_frac * capacity (only when
        capacity > 0 and fill_frac > 0). This is the real "steady state" signal
        from perf-bench-vs-real-training-cost.md — the regime that emerges once
        the 1.5M replay buffer fills.
      - prefill-epochs: a fixed epoch count to skip as cold warmup (fallback /
        belt-and-suspenders when capacity is unknown or the buffer never fills).

    Returns (start_idx, reason). start_idx is clamped to len(rows). reason is
    one of "fillfrac", "prefill_epochs", "none".
    """
    n = len(rows)
    fill_idx = None
    if capacity > 0 and fill_frac > 0:
        target = fill_frac * capacity
        for i, r in enumerate(rows):
            if r.get("buf", 0) >= target:
                fill_idx = i
                break
    pre_idx = prefill_epochs if prefill_epochs > 0 else None
    candidates = [(i, why) for i, why in
                  ((fill_idx, "fillfrac"), (pre_idx, "prefill_epochs"))
                  if i is not None]
    if not candidates:
        return 0, "none"
    # Earliest-satisfied wins: whichever boundary we cross first is when we're
    # warm enough to start measuring.
    start_idx, reason = min(candidates, key=lambda t: t[0])
    return min(start_idx, n), reason


def parse_trainer_trajectory(log_path: Path, *, fill_frac: float = 0.0,
                             capacity: int = 0, prefill_epochs: int = 0) -> dict:
    """Read trainer.log and harvest the FULL per-epoch trajectory (epoch-capped
    mode). Unlike parse_trainer_log (which windows by warmup/measurement secs),
    this keeps every ^epoch line so a fixed-epoch sweep can see the per-epoch
    cost runaway documented in wiki/topics/perf-bench-vs-real-training-cost.md.

    Returns dict with:
      rows: list[dict] one per epoch with keys
        epoch, steps, wall_s, gen_s, train_s, plies, tile, new, buf, elo
        (elo is None when absent).
      summary fields over the full run (see keys assigned below).

    Warm-buffer mode (fill_frac>0 or prefill_epochs>0): also computes a SECOND
    set of slope/trend fields over only the post-buffer-fill epochs (the steady
    state regime), plus a tile_verdict ("BOUNDED"/"DIVERGING"/"INCONCLUSIVE").
    These are the load-bearing outputs of perf-bench-vs-real-training-cost.md
    lane 1: the cold-window slope is non-predictive; the post-fill slope is the
    real training cost. When neither knob is set the post-fill fields equal the
    full-run fields (start_idx 0), preserving current behavior.
    """
    rows: list[dict] = []
    nan_count = 0
    if log_path.exists():
        with log_path.open() as f:
            for line in f:
                m = EPOCH_RE.match(line)
                if not m:
                    continue
                wm = EPOCH_WALL_RE.search(line)
                gm = GEN_TAIL_RE.search(line)
                tm = TRAIN_TAIL_RE.search(line)
                sm = STEPS_RE.search(line)
                pm = PLIES_RE.search(line)
                bm = BUF_RE.search(line)
                tile_m = TILE_RE.search(line)
                new_m = NEW_RE.search(line)
                elo_m = ELO_RE.search(line)
                plm = PL_RE.search(line)
                vlm = VL_RE.search(line)
                if (plm and _is_nan_token(plm.group(1))) or (
                    vlm and _is_nan_token(vlm.group(1))
                ):
                    nan_count += 1
                rows.append(dict(
                    epoch=int(m.group(1)),
                    steps=int(sm.group(1)) if sm else 0,
                    wall_s=float(wm.group(1)) if wm else 0.0,
                    gen_s=float(gm.group(1)) if gm else 0.0,
                    train_s=float(tm.group(1)) if tm else 0.0,
                    plies=float(pm.group(1)) if pm else 0.0,
                    tile=int(tile_m.group(1)) if tile_m else 0,
                    new=int(new_m.group(1)) if new_m else 0,
                    buf=int(bm.group(1)) if bm else 0,
                    elo=float(elo_m.group(1)) if elo_m else None,
                ))
    n = len(rows)
    steps_seq = [r["steps"] for r in rows]
    wall_seq = [r["wall_s"] for r in rows]
    elos = [r["elo"] for r in rows if r["elo"] is not None]
    first_steps = steps_seq[0] if steps_seq else 0
    last_steps = steps_seq[-1] if steps_seq else 0
    first_wall = wall_seq[0] if wall_seq else 0.0
    last_wall = wall_seq[-1] if wall_seq else 0.0

    # --- warm-buffer (post-fill) window --------------------------------------
    # The cold-buffer slope is non-predictive (perf-bench-vs-real-training-cost
    # lane 1). Measure the SLOPE of tile / steps / wall over only the post-fill
    # epochs and emit a bounded-vs-diverging verdict from the tile slope.
    pf_start, pf_reason = postfill_window(
        rows, fill_frac=fill_frac, capacity=capacity,
        prefill_epochs=prefill_epochs,
    )
    pf_rows = rows[pf_start:]
    pf_n = len(pf_rows)
    pf_tile = [float(r["tile"]) for r in pf_rows]
    pf_steps = [float(r["steps"]) for r in pf_rows]
    pf_wall = [r["wall_s"] for r in pf_rows]
    tile_slope = _slope(pf_tile)
    steps_slope_pf = _slope(pf_steps)
    wall_slope_pf = _slope(pf_wall)
    # Verdict: the runaway has no fixed point — tile grows without bound. We call
    # it DIVERGING when the tile slope is meaningfully positive relative to the
    # window's own scale (a per-epoch growth >= ~2% of the mean tile sustained
    # over the window). BOUNDED when the slope is flat/negative. INCONCLUSIVE
    # when we don't have enough post-fill epochs to trust the slope (the whole
    # POINT of this lane is that a too-short window can't adjudicate).
    mean_tile = (sum(pf_tile) / pf_n) if pf_n else 0.0
    rel_tile_slope = (tile_slope / mean_tile) if mean_tile > 0 else 0.0
    MIN_TREND_EPOCHS = 20  # lane 1 asks for >= 20 post-fill epochs
    if pf_n < max(2, MIN_TREND_EPOCHS):
        tile_verdict = "INCONCLUSIVE"
    elif rel_tile_slope >= 0.02:
        tile_verdict = "DIVERGING"
    else:
        tile_verdict = "BOUNDED"

    return dict(
        rows=rows,
        epochs_completed=n,
        total_wall_s=sum(wall_seq),
        mean_steps_per_epoch=(sum(steps_seq) / n) if n else 0.0,
        last_steps_per_epoch=last_steps,
        first_steps_per_epoch=first_steps,
        # last/first ratio + LS slope: two views of the runaway. >1 / >0 = growing.
        steps_growth_ratio=(last_steps / first_steps) if first_steps else 0.0,
        steps_slope=_slope([float(s) for s in steps_seq]),
        mean_wall_per_epoch=(sum(wall_seq) / n) if n else 0.0,
        last_wall_per_epoch=last_wall,
        wall_growth_ratio=(last_wall / first_wall) if first_wall else 0.0,
        wall_slope=_slope(wall_seq),
        final_plies=rows[-1]["plies"] if rows else 0.0,
        final_elo=(elos[-1] if elos else None),
        nan_count=nan_count,
        # warm-buffer (post-fill) trend fields. Equal to full-run when no
        # prefill knob is set (pf_start == 0).
        postfill_start_epoch=(pf_rows[0]["epoch"] if pf_rows else 0),
        postfill_reason=pf_reason,
        postfill_epochs=pf_n,
        postfill_mean_steps_per_epoch=(sum(pf_steps) / pf_n) if pf_n else 0.0,
        postfill_mean_train_s=(
            sum(r["train_s"] for r in pf_rows) / pf_n) if pf_n else 0.0,
        postfill_mean_wall_per_epoch=(sum(pf_wall) / pf_n) if pf_n else 0.0,
        postfill_mean_tile=mean_tile,
        postfill_tile_slope=tile_slope,
        postfill_tile_slope_rel=rel_tile_slope,
        postfill_steps_slope=steps_slope_pf,
        postfill_wall_slope=wall_slope_pf,
        tile_verdict=tile_verdict,
    )


TRAJECTORY_COLS = [
    "epoch", "steps", "wall_s", "gen_s", "train_s",
    "plies", "tile", "new", "buf", "elo",
]


def write_trajectory_tsv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAJECTORY_COLS, delimiter="\t")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in TRAJECTORY_COLS}
            if out["elo"] is None:
                out["elo"] = ""
            w.writerow(out)
    os.replace(tmp, path)


def format_trajectory_table(rows: list[dict]) -> str:
    """Compact fixed-width table for the driver log."""
    hdr = f"{'ep':>4} {'steps':>6} {'wall_s':>7} {'gen_s':>6} {'train_s':>7} {'plies':>6} {'tile':>5} {'new':>6} {'buf':>8} {'elo':>6}"
    lines = [hdr]
    for r in rows:
        elo = f"{r['elo']:.0f}" if r.get("elo") is not None else "-"
        lines.append(
            f"{r['epoch']:>4} {r['steps']:>6} {r['wall_s']:>7.1f} "
            f"{r['gen_s']:>6.1f} {r['train_s']:>7.1f} {r['plies']:>6.1f} "
            f"{r['tile']:>5} {r['new']:>6} {r['buf']:>8} {elo:>6}"
        )
    return "\n".join(lines)


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


def build_trainer_cmd(cell: dict, dirs: dict, max_epochs: int = 0) -> list[str]:
    # max_epochs == 0 → time-windowed mode: effectively unbounded epochs and the
    # driver SIGTERMs the group after warmup+measurement. max_epochs > 0 →
    # epoch-capped mode: the trainer stops itself after N epochs and the driver
    # waits for it to exit (see run_cell). Only the STOP condition changes here;
    # the WL5 recipe below is identical in both modes.
    epochs_arg = str(max_epochs) if max_epochs > 0 else "1000000"
    cmd = [
        sys.executable, "-u", "-m", "gomoku.train",
        "--size", cell["model"],
        "--stem-padding", "1",
        "--epochs", epochs_arg,  # 1000000 = effectively unbounded; we SIGTERM
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
    # Warm-buffer mode: shrink the replay buffer so it FILLS within a perf-cell
    # time budget (the production 1.5M buffer takes ~27 epochs to fill, far past
    # a smoke cell). The runaway is a property of the loop gain, not the absolute
    # capacity, so a smaller buffer reaches the same post-fill steady state
    # sooner while preserving the dynamics we want to measure. See
    # perf-bench-vs-real-training-cost.md lane 1. 0 = leave trainer default.
    if cell.get("replay_buffer_size", 0) > 0:
        cmd += ["--replay-buffer-size", str(cell["replay_buffer_size"])]
    # Optional Path (a): start from a saved buffer artifact (latest.pt with an
    # embedded replay_buffer). Fastest warm start when the artifact exists and
    # matches this cell's model. NOTE: --resume sets start_epoch from the
    # artifact and the wave barrier waits on v{start_epoch}, so this only works
    # cleanly with a matching artifact — prefer the prefill phase otherwise.
    if cell.get("prefill_buffer_from"):
        cmd += ["--resume", str(cell["prefill_buffer_from"])]
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
    if cell.get("fp16_eval", False):
        cmd += ["--fp16-eval"]
    return cmd


def run_cell(cell: dict, sweep_dir: Path, warmup_secs: int,
             measurement_secs: int, device: str, dry_run: bool = False,
             max_epochs: int = 0
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

    trainer_cmd = build_trainer_cmd(cell, dirs, max_epochs=max_epochs)
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
    timed_out = False
    if max_epochs > 0:
        # Epoch-capped mode: the trainer stops itself after max_epochs. We wait
        # for the TRAINER process to exit (the workers run open-loop and never
        # exit on their own — they keep generating against worker_weights.pt —
        # so we cannot wait on the group; we poll the trainer specifically and
        # tear down the workers afterward). A generous safety timeout prevents a
        # runaway cell (the unbounded per-epoch cost from
        # perf-bench-vs-real-training-cost.md) from hanging forever.
        safety_deadline = t0 + max(600, max_epochs * 120)
        try:
            while not _INTERRUPTED:
                if trainer.poll() is not None:
                    break  # trainer finished its N epochs (or crashed)
                if time.perf_counter() > safety_deadline:
                    timed_out = True
                    break
                time.sleep(1.0)
        finally:
            wall_secs = time.perf_counter() - t0
            # SIGTERM each pgroup (trainer + all workers), then SIGKILL after
            # grace — identical teardown to the time-windowed path.
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

        return _harvest_epoch_capped(
            cell, dirs, wall_secs, started_at, max_epochs,
            timed_out=timed_out,
            warmup_secs=warmup_secs, measurement_secs=measurement_secs,
        )

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


def _harvest_epoch_capped(cell: dict, dirs: dict, wall_secs: float,
                          started_at: str, max_epochs: int, *, timed_out: bool,
                          warmup_secs: int, measurement_secs: int) -> dict:
    """Harvest the FULL per-epoch trajectory for an epoch-capped cell, write
    trajectory.tsv, print a compact table, and build the summary row. Status:
      - 'timeout' if the safety deadline fired (possible runaway).
      - 'ok' if the trainer completed >= max_epochs epoch lines.
      - 'partial' if it exited early (crash) with >=1 epoch.
      - 'failed' if it emitted no epoch lines at all.
    """
    cell_id = cell["cell_id"]
    traj = parse_trainer_trajectory(
        dirs["logs"] / "trainer.log",
        fill_frac=cell.get("prefill_to_fill_frac", 0.0),
        capacity=cell.get("replay_buffer_size", 0),
        prefill_epochs=cell.get("prefill_epochs", 0),
    )
    rows = traj["rows"]
    n = traj["epochs_completed"]
    warm = (cell.get("prefill_to_fill_frac", 0.0) > 0
            or cell.get("prefill_epochs", 0) > 0)

    if rows:
        traj_path = dirs["cell"] / "trajectory.tsv"
        write_trajectory_tsv(traj_path, rows)
        print(f"[traj] {cell_id} {n} epochs -> {traj_path}")
        print(format_trajectory_table(rows))
        if warm:
            # The load-bearing warm-buffer verdict (lane 1). The single
            # throughput number is explicitly NOT the deliverable; the SLOPE is.
            print(
                f"[warmbuf] {cell_id} post-fill window: "
                f"start_epoch={traj['postfill_start_epoch']} "
                f"({traj['postfill_reason']}) "
                f"n={traj['postfill_epochs']} epochs | "
                f"steps/epoch mean={traj['postfill_mean_steps_per_epoch']:.0f} "
                f"slope={traj['postfill_steps_slope']:.2f}/ep | "
                f"train_s/epoch mean={traj['postfill_mean_train_s']:.1f}s | "
                f"wall/epoch mean={traj['postfill_mean_wall_per_epoch']:.1f}s "
                f"slope={traj['postfill_wall_slope']:.2f}s/ep | "
                f"tile mean={traj['postfill_mean_tile']:.0f} "
                f"slope={traj['postfill_tile_slope']:.2f}/ep "
                f"(rel {traj['postfill_tile_slope_rel']*100:.1f}%/ep) "
                f"=> tile {traj['tile_verdict']}"
            )
            if traj["tile_verdict"] == "INCONCLUSIVE":
                print(
                    f"[warmbuf] {cell_id} WARNING: only "
                    f"{traj['postfill_epochs']} post-fill epochs; need >= 20 "
                    f"to adjudicate bounded-vs-diverging. This cell is "
                    f"non-predictive of steady-state cost — extend --max-epochs."
                )
    else:
        print(f"[traj] {cell_id} no epoch lines parsed from trainer.log")

    if timed_out:
        cell_status = "timeout"
    elif n >= max_epochs:
        cell_status = "ok"
    elif n >= 1:
        cell_status = "partial"
    else:
        cell_status = "failed"

    # Throughput proxies over the full run (cumulative buf delta / total wall).
    # buf= is the replay-buffer size, monotone until eviction; new= is per-cycle
    # inflow. Sum(new) is the truer "positions ingested" for an epoch-capped run.
    total_new = sum(r["new"] for r in rows)
    final_buf = rows[-1]["buf"] if rows else 0
    total_wall = traj["total_wall_s"] or wall_secs
    final_elo = traj["final_elo"]

    return dict(
        cell_id=cell_id,
        model=cell["model"],
        workers=cell["workers"],
        games_per_batch=cell["games_per_batch"],
        n_simulations=cell["n_simulations"],
        wave_size=cell["wave_size"],
        wall_secs=round(wall_secs, 3),
        total_games=0,  # n/a in epoch-capped mode (no per-game record walk)
        total_aug_examples=total_new,
        total_raw_plies=0,
        aug_pos_per_sec=round(total_new / total_wall, 1) if total_wall else 0.0,
        games_per_sec=0.0,
        plies_mean=round(traj["final_plies"], 2),
        cell_status=cell_status,
        started_at=started_at,
        # train-specific extras (kept for SUMMARY_COLS compatibility):
        epochs_per_sec=round(n / total_wall, 4) if total_wall else 0.0,
        trainer_step_s_p50=0.0,
        epochs_in_window=n,
        ema_tau=cell["ema_tau"],
        grad_accum_steps=cell["grad_accum_steps"],
        wave_mode=int(cell["wave_mode"]),
        warmup_secs=warmup_secs,
        measurement_secs=measurement_secs,
        # epoch-capped extras (after the time-windowed block):
        max_epochs=max_epochs,
        epochs_completed=n,
        total_wall_s=round(total_wall, 1),
        mean_steps_per_epoch=round(traj["mean_steps_per_epoch"], 1),
        last_steps_per_epoch=traj["last_steps_per_epoch"],
        steps_growth_ratio=round(traj["steps_growth_ratio"], 3),
        steps_slope=round(traj["steps_slope"], 3),
        mean_wall_per_epoch=round(traj["mean_wall_per_epoch"], 2),
        last_wall_per_epoch=round(traj["last_wall_per_epoch"], 2),
        wall_growth_ratio=round(traj["wall_growth_ratio"], 3),
        wall_slope=round(traj["wall_slope"], 3),
        final_plies=round(traj["final_plies"], 2),
        final_elo=("" if final_elo is None else round(final_elo, 1)),
        final_buf=final_buf,
        nan_count=traj["nan_count"],
        # warm-buffer (post-fill) extras (blank/equal-to-full when not warm):
        replay_buffer_size=cell.get("replay_buffer_size", 0),
        prefill_to_fill_frac=cell.get("prefill_to_fill_frac", 0.0),
        prefill_epochs=cell.get("prefill_epochs", 0),
        postfill_start_epoch=traj["postfill_start_epoch"],
        postfill_reason=traj["postfill_reason"],
        postfill_epochs=traj["postfill_epochs"],
        postfill_mean_steps_per_epoch=round(
            traj["postfill_mean_steps_per_epoch"], 1),
        postfill_mean_train_s=round(traj["postfill_mean_train_s"], 2),
        postfill_mean_wall_per_epoch=round(
            traj["postfill_mean_wall_per_epoch"], 2),
        postfill_mean_tile=round(traj["postfill_mean_tile"], 1),
        postfill_tile_slope=round(traj["postfill_tile_slope"], 3),
        postfill_tile_slope_rel=round(traj["postfill_tile_slope_rel"], 4),
        postfill_steps_slope=round(traj["postfill_steps_slope"], 3),
        postfill_wall_slope=round(traj["postfill_wall_slope"], 3),
        tile_verdict=traj["tile_verdict"],
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
    # epoch-capped extras (blank in time-windowed rows):
    "max_epochs", "epochs_completed", "total_wall_s",
    "mean_steps_per_epoch", "last_steps_per_epoch",
    "steps_growth_ratio", "steps_slope",
    "mean_wall_per_epoch", "last_wall_per_epoch",
    "wall_growth_ratio", "wall_slope",
    "final_plies", "final_elo", "final_buf", "nan_count",
    # warm-buffer (post-fill) extras (blank in cold / time-windowed rows):
    "replay_buffer_size", "prefill_to_fill_frac", "prefill_epochs",
    "postfill_start_epoch", "postfill_reason", "postfill_epochs",
    "postfill_mean_steps_per_epoch", "postfill_mean_train_s",
    "postfill_mean_wall_per_epoch", "postfill_mean_tile",
    "postfill_tile_slope", "postfill_tile_slope_rel",
    "postfill_steps_slope", "postfill_wall_slope", "tile_verdict",
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
        fp16_eval=args.fp16_eval,
        # warm-buffer mode (opt-in; 0/empty = unchanged cold behavior)
        replay_buffer_size=args.replay_buffer_size,
        prefill_to_fill_frac=args.prefill_to_fill_frac,
        prefill_epochs=args.prefill_epochs,
        prefill_buffer_from=args.prefill_buffer_from,
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
    # fp16-eval on the worker side. Outputs cast back to fp32 before MCTS
    # reads them (per gomoku/mcts.py make_torch_evaluator), so the perf-lab
    # treats this as a no-behavior-change knob — verified at L06-followup
    # (commit 1490c2d / Reviewer audit). Compounding with --wave-size 512
    # and --sgd-per-position is the L11b' compound finding.
    p.add_argument("--fp16-eval", action="store_true",
                   help="Pass --fp16-eval to workers (model cast to "
                        "torch.float16; outputs cast back to fp32 at MCTS "
                        "boundary). Eval-only — DO NOT use on trainer "
                        "(trainer always runs fp32 SGD).")
    # Window (time-windowed mode; ignored when --max-epochs > 0)
    p.add_argument("--warmup-secs", type=int, default=30)
    p.add_argument("--measurement-secs", type=int, default=60)
    # Epoch-capped mode. 0 (default) = unchanged time-windowed behavior. >0 =
    # pass --epochs N to the trainer, wait for it to exit on its own (with a
    # max(600, N*120)s safety timeout), and harvest the FULL per-epoch
    # trajectory to trajectory.tsv. This is research lane 1 of
    # wiki/topics/perf-bench-vs-real-training-cost.md: run past buffer-fill and
    # measure the per-epoch cost SLOPE, not a cold-window throughput number.
    p.add_argument("--max-epochs", type=int, default=0,
                   help="If >0, run a fixed-epoch cell: trainer stops after N "
                        "epochs and the full per-epoch trajectory is harvested "
                        "to trajectory.tsv. 0 = time-windowed (default).")
    # Warm-buffer mode (lane 1 of perf-bench-vs-real-training-cost.md). ALL
    # opt-in and additive: with none of these set the tool behaves exactly as
    # before. A cold-window R-TRAIN number is non-predictive of steady-state
    # training cost; this mode measures the per-epoch SLOPE at/after buffer-fill
    # and emits a tile BOUNDED-vs-DIVERGING verdict. Use WITH --max-epochs N so
    # there are >= 20 post-fill epochs to adjudicate the slope.
    p.add_argument("--replay-buffer-size", type=int, default=0,
                   help="Warm-buffer: pass --replay-buffer-size to the trainer "
                        "(0 = leave trainer default 1.5M). Shrink it (e.g. "
                        "20000) so the buffer FILLS within the cell's epoch "
                        "budget — the runaway is a loop-gain property, not a "
                        "capacity one, so a small buffer reaches the same "
                        "post-fill steady state much sooner. Also used as the "
                        "denominator for --prefill-to-fill-frac.")
    p.add_argument("--prefill-to-fill-frac", type=float, default=0.0,
                   help="Warm-buffer: treat epochs where buf >= F*capacity as "
                        "post-fill (the measured steady-state window). 0 = off. "
                        "Requires --replay-buffer-size > 0 for the capacity. "
                        "e.g. 0.95 starts the slope/verdict once the buffer is "
                        "95%% full.")
    p.add_argument("--prefill-epochs", type=int, default=0,
                   help="Warm-buffer: skip the first N epochs as cold warmup "
                        "before measuring the post-fill slope. Used as a "
                        "fallback when capacity is unknown, or combined with "
                        "--prefill-to-fill-frac (earliest boundary wins). 0=off.")
    p.add_argument("--prefill-buffer-from", type=str, default=None,
                   help="Warm-buffer Path (a): --resume the trainer from a "
                        "saved checkpoint (latest.pt) whose embedded "
                        "replay_buffer pre-fills the buffer at startup. Fastest "
                        "warm start, but only works cleanly when the artifact's "
                        "model + epoch tag match this cell (wave barrier waits "
                        "on v{start_epoch}); prefer the prefill phase otherwise.")
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

    # Warm-buffer mode is an extension of epoch-capped mode: the post-fill slope
    # needs the full per-epoch trajectory, which only the --max-epochs path
    # harvests. Fail loudly rather than silently producing a cold-window number
    # (the exact non-predictive trap this lane fixes).
    warm = (args.prefill_to_fill_frac > 0 or args.prefill_epochs > 0
            or args.prefill_buffer_from)
    if warm and args.max_epochs <= 0:
        raise SystemExit(
            "warm-buffer mode (--prefill-*) requires --max-epochs N (epoch-"
            "capped mode) so the post-fill slope can be measured over >= 20 "
            "post-fill epochs. A time-windowed cold cell is non-predictive of "
            "steady-state training cost (perf-bench-vs-real-training-cost.md "
            "lane 1)."
        )
    if args.prefill_to_fill_frac > 0 and args.replay_buffer_size <= 0:
        raise SystemExit(
            "--prefill-to-fill-frac needs --replay-buffer-size > 0 for the "
            "capacity denominator (shrink it so the buffer actually fills "
            "inside --max-epochs)."
        )

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
                       device=args.device, dry_run=True,
                       max_epochs=args.max_epochs)
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
        stop_desc = (f"max_epochs={args.max_epochs}" if args.max_epochs > 0
                     else f"warmup={args.warmup_secs}s measure={args.measurement_secs}s")
        print(f"[run ] {cell['cell_id']} model={cell['model']} W={cell['workers']} "
              f"G={cell['games_per_batch']} S={cell['n_simulations']} V={cell['wave_size']} "
              f"ema={cell['ema_tau']} ga={cell['grad_accum_steps']} wave={int(cell['wave_mode'])} "
              f"{stop_desc}")
        row = run_cell(cell, out_dir,
                       warmup_secs=args.warmup_secs,
                       measurement_secs=args.measurement_secs,
                       device=args.device,
                       max_epochs=args.max_epochs)
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
