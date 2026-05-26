"""lab_status — the cockpit. ONE glance over the whole lab.

The autopilot (gpu daemon + derby broker + candidate pool) keeps state in three
places; reading it should not cost ten shell commands (the "human IS the
dashboard" anti-pattern — wiki/topics/cockpit-vs-autopilot.md). This composes all
three into one regenerated view and, most importantly, surfaces the single
"NEEDS YOU" line when the broker has a decision it can't make on the evidence.

    python scripts/lab_status.py --derby-dir sweep_runs/derby_v9 [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gpu_daemon as gd  # noqa: E402
import derby_pool  # noqa: E402


def snapshot(derby_dir: Path, queue_dir: Optional[Path], pool_dir: Optional[Path]) -> dict:
    """Compose the three state sources into one dict. Degrades gracefully when a
    source is absent (no daemon yet, no broker state, no pool)."""
    q = gd.Queue(queue_dir) if queue_dir else gd.Queue(gd.DEFAULT_QUEUE_DIR)
    counts = {st: len(list(q.state_dir(st).glob("*.json"))) for st in gd.STATES}
    daemon = {
        "alive": q.daemon_alive(),
        "pid": q.daemon_pid(),
        "disk_free_gb": round(gd.disk_free_bytes() / 1e9, 1),
        "counts": counts,
        "running": gd._running_with_progress(q),
    }

    broker_state_path = Path(derby_dir) / "broker_state.json"
    broker: dict[str, Any] = {}
    if broker_state_path.exists():
        try:
            broker = json.loads(broker_state_path.read_text())
        except (OSError, json.JSONDecodeError):
            broker = {}

    pool_dir = pool_dir or derby_pool.DEFAULT_POOL_DIR
    pool = {st: [c["name"] for c in derby_pool.list_candidates(status=st, pool_dir=pool_dir)]
            for st in ("available", "running", "retired")}

    needs_you = broker.get("needs_you")
    return {"daemon": daemon, "broker": broker, "pool": pool, "needs_you": needs_you}


def render(snap: dict) -> str:
    d = snap["daemon"]
    b = snap["broker"]
    pool = snap["pool"]
    out: list[str] = []
    out.append(f"═══ LAB STATUS ═══  {gd.now_iso()}")
    ny = snap.get("needs_you")
    out.append(f"NEEDS YOU: {ny}" if ny else "NEEDS YOU: —")
    out.append("")

    c = d["counts"]
    state = "RUNNING" if d["alive"] else "DOWN"
    out.append(f"DAEMON: {state}  pid={d['pid']}  disk_free={d['disk_free_gb']}GB")
    out.append(f"  queue: pending={c['pending']} running={c['running']} "
               f"done={c['done']} failed={c['failed']} cancelled={c['cancelled']}")
    for job in d["running"]:
        el = job.get("elapsed_secs")
        eta = job.get("eta_secs")
        prog = f"{el:.0f}s" + (f" / ~{eta:.0f}s left" if eta is not None else "") if el is not None else ""
        out.append(f"  in flight: {job['id']}  {prog}")

    if b:
        out.append("")
        out.append(f"DERBY: tick={b.get('tick')}  champion={b.get('champion')}")
        cur = b.get("current_job")
        if cur:
            out.append(f"  chunk in flight: {cur['lane']}  ({cur['job_id']})")
        hdr = f"  {'LANE':16s} {'STATUS':9s} {'CHUNK':>5s} {'WALL':>7s} {'PEAK':>7s} {'RATE/h':>7s} {'NOPEAK':>6s}"
        out.append(hdr)
        for n, l in (b.get("lanes") or {}).items():
            peak = "" if l.get("peak_elo") is None else f"{l['peak_elo']:.0f}"
            rate = "" if l.get("climb_rate") is None else f"{l['climb_rate']:+.0f}"
            out.append(f"  {n:16s} {l['status']:9s} {l['chunks_done']:5d} "
                       f"{l['wall_secs_total']/60:6.1f}m {peak:>7s} {rate:>7s} "
                       f"{l['chunks_since_peak']:6d}")
        lv = b.get("last_verdict")
        if lv:
            verd = lv.get("crowned") or ("inconclusive: " + str(lv.get("reason", "")))
            out.append(f"  last verdict ({lv.get('ts')}): {verd}")

    out.append("")
    out.append(f"POOL: available={len(pool['available'])} running={len(pool['running'])} "
               f"retired={len(pool['retired'])}")
    if pool["available"]:
        out.append(f"  available: {', '.join(pool['available'])}")
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derby-dir", type=Path, required=True)
    p.add_argument("--queue-dir", type=Path, default=None)
    p.add_argument("--pool-dir", type=Path, default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    snap = snapshot(args.derby_dir, args.queue_dir, args.pool_dir)
    print(json.dumps(snap, indent=2, sort_keys=True, default=str) if args.json else render(snap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
