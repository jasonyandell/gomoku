"""derby_broker — the nonstop, open-entry, age-out derby's POLICY layer.

The [gpu daemon](gpu_daemon.py) is the *mechanism*: the one serial GPU queue that
owns box state and decides nothing except run-next-by-(tier,priority). This broker
is the *policy*: it decides what to QUEUE and what to RETIRE, and never touches the
GPU itself. Jason's model (2026-05-26): "all the contention goes to one thing that
has a queue; Claude is free to do Claude things."

One tick (see wiki/topics/gpu-broker-architecture.md):

  1. poll the in-flight chunk; when it finishes, fold its anchored elo into the
     lane's CLIMB history (cheap, in-race signal — NOT a swap signal).
  2. age-out: a lane past its TTL is retired ONLY if it has also plateaued
     (no new peak in `peak_window` chunks). "Never cut a climber", as a rule.
  3. refill: claim candidates from the open-entry pool into freed slots.
  4. verdict (~hourly, OFF-GPU, in a thread so GPU keeps flowing): pairwise H2H
     round-robin over lane peaks with a CI<delta gate — the only trustworthy
     swap/crown signal. No verdict -> hold incumbent + raise the "needs you" line.
  5. pick one lane by climb-rate priority (entry-fee -> starvation -> steepest
     climb, with patience) and submit ONE 300s chunk to the daemon.

The daemon is serial, so the broker keeps exactly one chunk in flight at a time.

CLI::
    python scripts/derby_broker.py run   --derby-dir sweep_runs/derby_v9 [--once]
    python scripts/derby_broker.py status --derby-dir sweep_runs/derby_v9
    python scripts/derby_broker.py seed  --derby-dir ... --name control --cell derby-v7-control
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gpu_daemon as gd  # noqa: E402  (the queue + submit/poll API)
import derby_pool  # noqa: E402     (the open-entry candidate pool)

REPO_ROOT = gd.REPO_ROOT

# ---- defaults (mirrors derby_v8 board + the 1–2h age-out Jason asked for) ----
DEFAULTS = {
    "slice_secs": 300.0,        # the 5-min chunk
    "pool_size": 4,             # running lanes
    "ttl_secs": 5400.0,         # 1.5h of accrued GPU before a lane is eligible to retire
    "peak_window": 6,           # chunks; also the plateau window for age-out + patience
    "starvation_factor": 2,     # a lane unfed for factor*N picks jumps the queue
    "verdict_period_secs": 3600.0,  # accrued GPU between trustworthy re-ranks
    "verdict_games": 200,       # games/pair for the H2H verdict
    "swap_losers": True,        # retire a clearly-dominated lane early on a verdict
}


def _latest_pt(cell: str) -> Path:
    """The cell's resumable checkpoint (mirrors gpu_daemon.resolve_resume 'auto')."""
    try:
        run_sweep = gd._import_run_sweep()
        return run_sweep.cell_dirs(run_sweep.CELLS[cell])["checkpoint_dir"] / "latest.pt"
    except Exception:
        return REPO_ROOT / "sweep_runs" / cell / "checkpoints" / "latest.pt"


class Broker:
    """Owns broker_state.json and runs the policy tick. All GPU/eval dependencies
    are injectable so the tick logic is unit-testable with no MPS and no real evals.
    """

    def __init__(
        self,
        derby_dir: Path,
        queue: Optional["gd.Queue"] = None,
        pool_dir: Optional[Path] = None,
        config: Optional[dict] = None,
        submit_fn: Callable = gd.submit,
        poll_fn: Callable = gd.poll,
        verdict_fn: Optional[Callable[[dict], dict]] = None,
        sync_verdict: bool = False,
    ):
        self.derby_dir = Path(derby_dir)
        self.derby_dir.mkdir(parents=True, exist_ok=True)
        self.peaks_dir = self.derby_dir / "peaks"
        self.state_path = self.derby_dir / "broker_state.json"
        self.queue = queue if queue is not None else gd.Queue(gd.DEFAULT_QUEUE_DIR)
        self.pool_dir = Path(pool_dir) if pool_dir else None
        self.submit_fn = submit_fn
        self.poll_fn = poll_fn
        self.verdict_fn = verdict_fn or self._default_verdict_fn
        self.sync_verdict = sync_verdict
        self._verdict_thread: Optional[threading.Thread] = None
        self._verdict_result: Optional[dict] = None
        self.state = self._load(config or {})

    # -- persistence ----------------------------------------------------------
    def _load(self, config_overrides: dict) -> dict:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            # let new config keys flow in without clobbering live values
            for k, v in DEFAULTS.items():
                state.setdefault("config", {}).setdefault(k, v)
            state["config"].update(config_overrides)
            return state
        cfg = dict(DEFAULTS)
        cfg.update(config_overrides)
        return {
            "config": cfg, "tick": 0, "lanes": {}, "current_job": None,
            "last_verdict": None, "verdict_wall_at_last": 0.0,
            "champion": None, "needs_you": None,
        }

    def save(self) -> None:
        gd.atomic_write_json(self.state_path, self.state)

    @property
    def cfg(self) -> dict:
        return self.state["config"]

    # -- lane helpers ---------------------------------------------------------
    @staticmethod
    def _fresh_lane(name: str, cell: str) -> dict:
        return {
            "name": name, "cell": cell, "wall_secs_total": 0.0, "chunks_done": 0,
            "climb_history": [], "climb_rate": None, "peak_elo": None,
            "peak_path": None, "chunks_since_peak": 0, "last_picked": 0,
            "status": "running",
        }

    def seed(self, name: str, cell: str) -> None:
        """Place a lane directly in the running pool (the initial field / a manual add)."""
        self.state["lanes"][name] = self._fresh_lane(name, cell)
        self.save()

    def _running(self) -> list[str]:
        return [n for n, l in self.state["lanes"].items() if l["status"] == "running"]

    # -- (1) fold a finished chunk's climb signal -----------------------------
    def _fold_result(self, lane_name: str, info: Optional[dict]) -> None:
        lane = self.state["lanes"].get(lane_name)
        if lane is None:
            return
        job = (info or {}).get("job") or {}
        result = (info or {}).get("result") or {}
        wall = job.get("wall_secs") or self.cfg["slice_secs"]
        lane["wall_secs_total"] = round(lane["wall_secs_total"] + float(wall), 1)
        lane["chunks_done"] += 1
        elo = result.get("model_elo")
        if elo is None:
            lane["chunks_since_peak"] += 1
            return
        lane["climb_history"].append([lane["wall_secs_total"], float(elo)])
        lane["climb_rate"] = self._climb_rate(lane)
        if lane["peak_elo"] is None or elo > lane["peak_elo"]:
            lane["peak_elo"] = float(elo)
            lane["peak_path"] = self._snapshot_peak(lane)
            lane["chunks_since_peak"] = 0
        else:
            lane["chunks_since_peak"] += 1

    def _climb_rate(self, lane: dict) -> Optional[float]:
        """Δelo per HOUR over the recent window (the in-race scheduling gradient)."""
        h = lane["climb_history"]
        if len(h) < 2:
            return None
        seg = h[-int(self.cfg["peak_window"]):]
        dwall = seg[-1][0] - seg[0][0]
        delo = seg[-1][1] - seg[0][1]
        return round(delo / dwall * 3600.0, 2) if dwall > 0 else None

    def _snapshot_peak(self, lane: dict) -> Optional[str]:
        """Copy the cell's latest.pt aside so the verdict ranks a STABLE peak."""
        latest = _latest_pt(lane["cell"])
        if not latest.exists():
            return lane.get("peak_path")  # keep prior peak if no checkpoint yet
        self.peaks_dir.mkdir(parents=True, exist_ok=True)
        dst = self.peaks_dir / f"{lane['name']}.pt"
        try:
            shutil.copy2(latest, dst)
        except OSError:
            return lane.get("peak_path")
        return str(dst)

    # -- (2) conditional peak-stay age-out ------------------------------------
    def _age_out_pass(self) -> list[str]:
        aged = []
        ttl = self.cfg["ttl_secs"]
        window = self.cfg["peak_window"]
        for name in self._running():
            lane = self.state["lanes"][name]
            if lane["wall_secs_total"] >= ttl and lane["chunks_since_peak"] >= window:
                lane["status"] = "aged_out"
                self._pool_retire(name, f"aged out: {lane['wall_secs_total']:.0f}s "
                                        f"accrued, plateaued {lane['chunks_since_peak']} chunks")
                aged.append(name)
        return aged

    def _pool_retire(self, name: str, reason: str) -> None:
        if self.pool_dir is None:
            return
        try:
            derby_pool.retire(name, reason, pool_dir=self.pool_dir)
        except (ValueError, OSError):
            pass  # lane wasn't pool-sourced (e.g. a seeded starter) — fine

    # -- (3) refill freed slots from the open-entry pool ----------------------
    def _refill_pass(self) -> list[str]:
        added = []
        if self.pool_dir is None:
            return added
        while len(self._running()) < int(self.cfg["pool_size"]):
            cand = derby_pool.claim(pool_dir=self.pool_dir)  # oldest available
            if cand is None:
                break
            self.state["lanes"][cand["name"]] = self._fresh_lane(cand["name"], cand["cell"])
            added.append(cand["name"])
        return added

    # -- (4) the trustworthy verdict (off-GPU, threaded) ----------------------
    def _default_verdict_fn(self, peaks: dict) -> dict:
        import derby_gate
        return derby_gate.verdict(peaks, games_per_pair=int(self.cfg["verdict_games"]))

    def _maybe_verdict(self) -> None:
        s = self.state
        # an async verdict landed -> apply it
        if self._verdict_result is not None:
            self._apply_verdict(self._verdict_result)
            self._verdict_result = None
            s["verdict_wall_at_last"] = self._total_wall()
            return
        if self._verdict_thread is not None and self._verdict_thread.is_alive():
            return  # one verdict at a time; GPU keeps flowing meanwhile
        peaks = {n: self.state["lanes"][n]["peak_path"]
                 for n in self._running() if self.state["lanes"][n].get("peak_path")}
        accrued = self._total_wall() - s.get("verdict_wall_at_last", 0.0)
        if accrued < self.cfg["verdict_period_secs"] or len(peaks) < 2:
            return
        if self.sync_verdict:
            self._apply_verdict(self.verdict_fn(peaks))
            s["verdict_wall_at_last"] = self._total_wall()
        else:
            def _run():
                try:
                    self._verdict_result = self.verdict_fn(peaks)
                except Exception as e:  # never let a verdict crash the broker loop
                    self._verdict_result = {"crowned": None, "escalate": True,
                                            "reason": f"verdict error: {e}", "ranking": []}
            self._verdict_thread = threading.Thread(target=_run, daemon=True)
            self._verdict_thread.start()

    def _apply_verdict(self, v: dict) -> None:
        self.state["last_verdict"] = {"ts": gd.now_iso(), **v}
        if v.get("crowned"):
            self.state["champion"] = v["crowned"]
            self.state["needs_you"] = None
            if self.cfg.get("swap_losers"):
                self._swap_loser(v)
        elif v.get("escalate"):
            # no trustworthy verdict — hold the incumbent, raise the one line
            self.state["needs_you"] = f"derby verdict inconclusive: {v.get('reason','')}"

    def _swap_loser(self, v: dict) -> None:
        """On a trustworthy verdict, retire the clearly-dominated lane (margin below
        the leader exceeds the combined CI) so its slot frees for a fresh candidate."""
        ranking = v.get("ranking") or []
        if len(ranking) < 2:
            return
        leader = ranking[0]
        loser = ranking[-1]
        l_name, l_delo, l_ci = loser[0], loser[1], loser[2]
        lead_delo, lead_ci = leader[1], leader[2]
        if l_name not in self._running():
            return
        margin = lead_delo - l_delo
        if margin > (lead_ci ** 2 + l_ci ** 2) ** 0.5:
            self.state["lanes"][l_name]["status"] = "aged_out"
            self._pool_retire(l_name, f"swapped out: lost H2H to {leader[0]} "
                                      f"by {margin:.0f} elo (beyond CI)")

    def _total_wall(self) -> float:
        return sum(l["wall_secs_total"] for l in self.state["lanes"].values())

    # -- (5) climb-rate priority pick -----------------------------------------
    def _pick_lane(self) -> Optional[str]:
        cands = self._running()
        if not cands:
            return None
        lanes = self.state["lanes"]
        # entry-fee: never-measured lanes first (no climb rate yet)
        entry = [n for n in cands if lanes[n].get("climb_rate") is None]
        if entry:
            return sorted(entry, key=lambda n: (lanes[n]["chunks_done"], n))[0]
        # starvation floor: anyone unfed too long jumps the queue
        tick = self.state["tick"]
        thresh = max(1, int(self.cfg["starvation_factor"]) * len(cands))
        starved = [n for n in cands if tick - lanes[n].get("last_picked", 0) >= thresh]
        if starved:
            return sorted(starved, key=lambda n: (lanes[n].get("last_picked", 0), n))[0]
        # hill-climb by Δelo/hr, plateaued lanes deprioritized (patience)
        window = int(self.cfg["peak_window"])

        def key(n: str):
            l = lanes[n]
            plateaued = 1 if l.get("chunks_since_peak", 0) >= window else 0
            return (plateaued, -(l.get("climb_rate") or 0.0), -(l.get("peak_elo") or 0.0), n)

        return sorted(cands, key=key)[0]

    # -- the tick -------------------------------------------------------------
    def tick(self) -> dict:
        s = self.state
        s["tick"] += 1
        # (1) poll the one in-flight chunk
        if s["current_job"]:
            info = self.poll_fn(self.queue, s["current_job"]["job_id"])
            state = (info or {}).get("state")
            if info is None or state in ("done", "failed", "cancelled"):
                self._fold_result(s["current_job"]["lane"], info)
                s["current_job"] = None
            else:
                self.save()
                return {"action": "chunk_running", "job": s["current_job"]["job_id"]}
        # (2) age-out  (3) refill  (4) verdict
        aged = self._age_out_pass()
        added = self._refill_pass()
        self._maybe_verdict()
        # (5) pick + submit one chunk
        lane = self._pick_lane()
        action = "idle"
        if lane is not None:
            spec = {
                "kind": "train", "cell": s["lanes"][lane]["cell"],
                "max_wall_secs": float(self.cfg["slice_secs"]), "final_eval": True,
                "resume_from": "auto", "tier": 2,
                "note": f"broker:{lane}:chunk{s['lanes'][lane]['chunks_done']}",
            }
            job_id = self.submit_fn(self.queue, spec)
            s["current_job"] = {"job_id": job_id, "lane": lane, "submitted_at": gd.now_iso()}
            s["lanes"][lane]["last_picked"] = s["tick"]
            action = "submitted"
        self.save()
        return {"action": action, "lane": lane, "aged_out": aged, "added": added,
                "needs_you": s["needs_you"]}

    def run_forever(self, poll_secs: float = 5.0, once: bool = False) -> None:
        while True:
            out = self.tick()
            print(f"[{gd.now_iso()}] tick {self.state['tick']}: {out}", flush=True)
            if out.get("needs_you"):
                print(f"[{gd.now_iso()}] NEEDS YOU: {out['needs_you']}", flush=True)
            if once:
                return
            time.sleep(poll_secs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _broker_from_args(args) -> Broker:
    pool_dir = args.pool_dir or derby_pool.DEFAULT_POOL_DIR
    queue = gd.Queue(args.queue_dir) if args.queue_dir else gd.Queue(gd.DEFAULT_QUEUE_DIR)
    return Broker(derby_dir=args.derby_dir, queue=queue, pool_dir=pool_dir)


def cmd_run(args) -> int:
    b = _broker_from_args(args)
    b.run_forever(poll_secs=args.poll_secs, once=args.once)
    return 0


def cmd_seed(args) -> int:
    b = _broker_from_args(args)
    b.seed(args.name, args.cell)
    print(f"seeded lane {args.name} (cell {args.cell}) into {b.state_path}")
    return 0


def cmd_status(args) -> int:
    b = _broker_from_args(args)
    s = b.state
    print(f"derby: {b.derby_dir}  tick={s['tick']}  champion={s.get('champion')}")
    if s.get("needs_you"):
        print(f"  ** NEEDS YOU: {s['needs_you']}")
    cur = s.get("current_job")
    print(f"  in flight: {cur['job_id'] + ' (' + cur['lane'] + ')' if cur else 'none'}")
    for n, l in s["lanes"].items():
        print(f"  {l['status']:9s} {n:16s} chunks={l['chunks_done']:3d} "
              f"wall={l['wall_secs_total']/60:6.1f}m peak={l['peak_elo']} "
              f"rate={l['climb_rate']} since_peak={l['chunks_since_peak']}")
    if args.json:
        print(json.dumps(s, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derby-dir", type=Path, required=True, help="broker state dir")
    p.add_argument("--pool-dir", type=Path, default=None, help="candidate pool dir")
    p.add_argument("--queue-dir", type=Path, default=None, help="gpu_daemon queue dir")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the broker tick loop")
    r.add_argument("--poll-secs", type=float, default=5.0)
    r.add_argument("--once", action="store_true", help="one tick then exit (testing)")
    r.set_defaults(func=cmd_run)

    sd = sub.add_parser("seed", help="place a lane in the running pool")
    sd.add_argument("--name", required=True)
    sd.add_argument("--cell", required=True)
    sd.set_defaults(func=cmd_seed)

    stt = sub.add_parser("status", help="show broker state")
    stt.add_argument("--json", action="store_true")
    stt.set_defaults(func=cmd_status)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
