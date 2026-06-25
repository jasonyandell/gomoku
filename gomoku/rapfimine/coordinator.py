"""The mine coordinator: BFS frontier + global D4-canonical dedup + K workers.

Owns the single source of truth (the ``seen`` canonical-key set and the BFS
frontier) and keeps K worker processes saturated. The hot path is a single
non-blocking loop: refill each worker's in-flight slot from the frontier, drain
finished results, dedup their children, enqueue novel ones. Workers do the CPU
(analyze + planes + canonicalize) and write their own shards; the coordinator
only does set-membership + queue plumbing, so it never becomes the bottleneck.

Durability/resume: nothing in RAM is load-bearing. On (re)start ``seen`` is
rebuilt from the stored shard ``keys`` and BFS re-walks from the root — any board
enqueued-but-unanalyzed at a crash is simply re-found from its stored parent.
``--total`` counts EXAMPLES already on disk + produced this run.

Monitoring (the goal's metric is moves/sec): prints count, instantaneous and
mean examples/sec, frontier/in-flight depth, and total Rapfi-engine CPU% every
``monitor_every_s``. If CPU is far from maxed the operator can raise ``workers``.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import subprocess
import time

from gomoku.game import GameState
from gomoku.rapfimine.canonical import canonical_state, canonical_key, drop_history
from gomoku.rapfimine.store import (
    count_examples, load_frontier, load_seen_keys, save_frontier,
)
from gomoku.rapfimine.worker import worker_main


def _rapfi_cpu_percent() -> float:
    """Total %CPU across all pbrain-rapfi processes (one cheap ps call)."""
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "%cpu,comm"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return -1.0
    tot = 0.0
    for line in out.splitlines():
        if "pbrain-rapfi" in line:
            try:
                tot += float(line.split(None, 1)[0])
            except (ValueError, IndexError):
                pass
    return tot


def run_mine(*, start_state: GameState, out_dir: str, total: int, workers: int,
             cmd: str, board_size: int, max_node: int = 20_000, max_pv: int = 24,
             expand_k: int = 8, timeout_ms: int = 1000, shard_size: int = 2_000,
             queue_high: int | None = None, monitor_every_s: float = 5.0,
             checkpoint_every_s: float = 60.0, frontier_cap: int = 1_200_000,
             ncpu: int | None = None) -> int:
    """Mine canonical idx-2 positions until ``total`` examples exist on disk.

    BFS (breadth) for diverse coverage of the idx-2 neighbourhood — a deep
    narrow DFS line would be a poor 'master this position' dataset. ``frontier_cap``
    bounds RAM + checkpoint cost: past it, analyzed boards are still stored but
    not expanded (we already have ample pending breadth). Returns the final
    on-disk example count.
    """
    import collections

    os.makedirs(out_dir, exist_ok=True)
    ncpu = ncpu or (os.cpu_count() or 1)
    queue_high = queue_high or max(2 * workers, 64)  # frontier-in-flight ceiling

    # ---- resume: rebuild dedup-set + count from existing shards --------------
    seen = load_seen_keys(out_dir)
    already = count_examples(out_dir)
    print(f"[mine] resume: {already} examples on disk, {len(seen)} canonical keys "
          f"seen", flush=True)

    # ---- restore the frontier (pending work) from its checkpoint, else seed ---
    # The frontier checkpoint is the pending-work half of the durable log; filter
    # it against the seen-set so anything analyzed since the snapshot is dropped.
    # deque + popleft = BFS order; appendleft on restore preserves it.
    frontier = collections.deque(
        drop_history(b) for b in load_frontier(out_dir) if canonical_key(b) not in seen
    )
    if frontier:
        print(f"[mine] resume: restored {len(frontier)} pending frontier boards",
              flush=True)
    else:
        canon_root, _ = canonical_state(drop_history(start_state))
        root_key = canonical_key(canon_root)
        if root_key not in seen:
            seen.add(root_key)
        frontier.append(canon_root)  # fresh start (or empty checkpoint) → BFS root
    if already >= total:
        print(f"[mine] already at target ({already} >= {total}); nothing to do.",
              flush=True)
        return already

    ctx = mp.get_context("spawn")
    work_q = ctx.Queue(maxsize=queue_high)
    result_q = ctx.Queue()

    procs = []
    for wid in range(workers):
        p = ctx.Process(target=worker_main, kwargs=dict(
            worker_id=wid, work_q=work_q, result_q=result_q, cmd=cmd,
            out_dir=out_dir, board_size=board_size, max_node=max_node,
            max_pv=max_pv, expand_k=expand_k, timeout_ms=timeout_ms,
            shard_size=shard_size), daemon=True)
        p.start()
        procs.append(p)

    produced = 0                       # examples produced THIS run
    target_run = total - already       # examples still needed
    inflight: dict[int, GameState] = {}  # board_id -> board, for frontier checkpoint
    next_id = 0
    t0 = time.time()
    state = {"last_mon": t0, "last_produced": 0, "last_ckpt": t0}

    def _dispatch() -> None:
        nonlocal next_id
        while frontier and len(inflight) < queue_high:
            board = frontier.popleft()  # FIFO = BFS breadth (diverse coverage)
            bid = next_id
            try:
                work_q.put_nowait((bid, board))
                inflight[bid] = board
                next_id += 1
            except queue.Full:
                frontier.appendleft(board)
                break

    def _checkpoint_frontier() -> None:
        # Pending work = not-yet-dispatched frontier + in-flight boards (re-queued
        # on resume). Capped at frontier_cap, so the pickle stays bounded.
        save_frontier(out_dir, list(frontier) + list(inflight.values()))
        state["last_ckpt"] = time.time()

    def _maybe_monitor() -> None:
        now = time.time()
        if now - state["last_ckpt"] >= checkpoint_every_s:
            _checkpoint_frontier()
        if now - state["last_mon"] < monitor_every_s:
            return
        inst = (produced - state["last_produced"]) / (now - state["last_mon"])
        mean = produced / max(now - t0, 1e-9)
        cpu = _rapfi_cpu_percent()
        cpu_frac = cpu / (ncpu * 100.0) if cpu >= 0 else -1
        print(f"[mine] {already + produced:>8d}/{total}  "
              f"inst={inst:6.0f}/s  mean={mean:6.0f}/s  "
              f"frontier={len(frontier):>6d}  in_flight={len(inflight):>3d}  "
              f"rapfi_cpu={cpu:6.0f}%  (~{cpu_frac*100:4.0f}% of {ncpu} cores)",
              flush=True)
        state["last_mon"] = now
        state["last_produced"] = produced

    try:
        _dispatch()
        while produced < target_run and (inflight or frontier):
            # Drain finished results (block briefly so we don't busy-spin).
            try:
                res = result_q.get(timeout=0.5)
            except queue.Empty:
                _dispatch()
                _maybe_monitor()
                continue
            inflight.pop(res.board_id, None)
            produced += res.n_examples
            # Past frontier_cap we keep STORING analyzed boards but stop expanding
            # (ample breadth already pending) — bounds RAM + checkpoint cost.
            if len(frontier) < frontier_cap:
                for ckey, cstate in res.children:
                    if ckey in seen:
                        continue
                    seen.add(ckey)
                    frontier.append(cstate)
            _dispatch()
            _maybe_monitor()
    finally:
        _checkpoint_frontier()  # final snapshot so a clean stop resumes exactly
        # Stop workers: drain queue, send sentinels, join.
        for _ in procs:
            try:
                work_q.put_nowait(None)
            except queue.Full:
                pass
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    final = count_examples(out_dir)
    dt = time.time() - t0
    print(f"[mine] done: {final} examples on disk (+{produced} this run) in "
          f"{dt:.0f}s = {produced/max(dt,1):.0f}/s mean", flush=True)
    return final
