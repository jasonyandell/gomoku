#!/usr/bin/env python3
"""Token-efficient one-glance status for an autonomous 15x15 training session.

WHY: an overnight loop needs to check "is training alive, progressing, and how
do the Rapfi evals look?" in a few lines, WITHOUT a wandb API round-trip or
dumping logs into the orchestrator's context. This reads only LOCAL files
(sweep_logs/<cell>/trainer.log, sweep_runs/<cell>/checkpoints/*.pt, and any
rapfi/ladder/searchscale *.jsonl) and prints a compact block.

Usage:
    python scripts/lab_status.py                 # autodetect the active run
    python scripts/lab_status.py G15-96x8-deepgen-board15   # a specific cell dir
    python scripts/lab_status.py --evals-only    # just the Rapfi eval rows

Read ≠ report: this only prints. It never touches the GPU or training.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SWEEP_RUNS = REPO / "sweep_runs"
SWEEP_LOGS = REPO / "sweep_logs"

EPOCH_RE = re.compile(
    r"epoch (\d+)/\S+.*?buf=(\d+) new=(\d+).*?pl=([\d.]+) vl=([\d.]+) plies=([\d.]+)"
    r".*?\(([\d.]+)s: gen=([\d.]+)s train=([\d.]+)s\).*?reuse=([\d.]+)"
)
EPOCH_NUM_RE = re.compile(r"(?:epoch|eval|_e)(\d+)")


def _ps_count(needle: str) -> int:
    try:
        out = subprocess.run(["ps", "ax"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return -1
    return sum(1 for ln in out.splitlines() if needle in ln and "lab_status" not in ln)


def _active_run(explicit: str | None) -> Path | None:
    if explicit:
        p = SWEEP_RUNS / explicit
        return p if p.exists() else None
    # most recently modified trainer.log under sweep_logs/
    logs = sorted(SWEEP_LOGS.glob("*/trainer.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return None
    return SWEEP_RUNS / logs[0].parent.name


def _last_epoch_line(cell_name: str) -> str | None:
    log = SWEEP_LOGS / cell_name / "trainer.log"
    if not log.exists():
        return None
    last = None
    err = None
    try:
        with log.open(errors="ignore") as f:
            for ln in f:
                if ln.startswith("epoch "):
                    last = ln.rstrip()
                if "Traceback" in ln or "Error" in ln or "[stop]" in ln:
                    err = ln.rstrip()
    except Exception:
        return None
    return last, err  # type: ignore


def _epoch_rate(ckpt_dir: Path) -> str:
    pts = sorted(ckpt_dir.glob("epoch*.pt"), key=lambda p: p.stat().st_mtime)
    if len(pts) < 2:
        return ""
    a, b = pts[-2], pts[-1]
    ea = int(re.search(r"(\d+)", a.name).group(1))
    eb = int(re.search(r"(\d+)", b.name).group(1))
    dt = b.stat().st_mtime - a.stat().st_mtime
    de = max(1, eb - ea)
    spe = dt / de
    age = time.time() - b.stat().st_mtime
    return f"~{spe:.0f}s/ep (~{60/spe:.1f} ep/min); newest ckpt {age/60:.0f}m old"


def _ep_tag(ckpt: str) -> str:
    m = EPOCH_NUM_RE.search(Path(ckpt).name)
    return f"e{m.group(1)}" if m else Path(ckpt).stem[:18]


def _eval_rows(paths: list[Path], limit: int = 4) -> list[str]:
    out = []
    for jp in paths:
        rows = []
        try:
            with jp.open(errors="ignore") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        rows.append(json.loads(ln))
        except Exception:
            continue
        if not rows:
            continue
        out.append(f"  [{jp.name}]")
        for r in rows[-limit:]:
            tag = _ep_tag(r.get("checkpoint", ""))
            sims = r.get("model_sims", "?")
            tmo = r.get("timeout_ms", "?")
            wr = r.get("win_rate")
            n = r.get("n_games", "?")
            w, l, d = r.get("wins", "?"), r.get("losses", "?"), r.get("draws", "?")
            wrs = f"{100*wr:.0f}%" if isinstance(wr, (int, float)) else "?"
            out.append(f"    {tag} sims={sims} @{tmo}ms  {wrs}  (n{n}: {w}W-{l}L-{d}D)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None, help="cell dir name under sweep_runs/")
    ap.add_argument("--evals-only", action="store_true")
    args = ap.parse_args()

    run = _active_run(args.run)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"== lab status {now} ==")

    if not args.evals_only:
        if run is None:
            print("train: (no run dir found)")
        else:
            cell = run.name
            trainers = _ps_count("gomoku.train")
            workers = _ps_count("gomoku.selfplay_worker")
            alive = "ALIVE" if trainers > 0 else "DOWN"
            print(f"train: {alive} (trainer={trainers} workers={workers})  cell={cell}")
            le = _last_epoch_line(cell)
            if le:
                line, err = le
                m = EPOCH_RE.search(line or "")
                if m:
                    ep, buf, new, pl, vl, plies, wall, gen, trn, reuse = m.groups()
                    flood = "  ⚠FLOOD" if float(gen) > float(trn) else ""
                    print(f"epoch {ep}  vl={vl} pl={pl} plies={plies} reuse={reuse}  "
                          f"buf={buf} new={new}g/ep  {wall}s/ep (gen{gen}/trn{trn}){flood}")
                else:
                    print(f"epoch-line: {line}")
                if err:
                    print(f"  note: {err}")
            print("rate:", _epoch_rate(run / "checkpoints") or "(n/a)")

    # eval rows: this run's rapfi/ladder jsonl + global searchscale/deepTC files
    epaths: list[Path] = []
    if run is not None:
        epaths += sorted((run).glob("*.jsonl")) + sorted((run / "checkpoints").glob("rapfi*.jsonl"))
    epaths += [Path(p) for p in glob.glob(str(SWEEP_RUNS / "*searchscale*.jsonl"))]
    epaths += [Path(p) for p in glob.glob(str(SWEEP_RUNS / "*deepTC*.jsonl"))]
    epaths += [Path(p) for p in glob.glob(str(SWEEP_RUNS / "g96x8_*.jsonl"))]
    # de-dup, keep rapfi/ladder/searchscale-ish
    seen = set()
    keep = []
    for p in epaths:
        if p in seen or not p.exists():
            continue
        seen.add(p)
        if any(k in p.name for k in ("rapfi", "ladder", "searchscale", "deepTC")):
            keep.append(p)
    print("rapfi evals:")
    rows = _eval_rows(keep)
    print("\n".join(rows) if rows else "  (none yet)")


if __name__ == "__main__":
    main()
