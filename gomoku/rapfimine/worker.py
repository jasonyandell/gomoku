"""One mining worker = one OS process + one Rapfi engine in a tight analyze loop.

This is the whole point of the harness: the in-process ``RapfiPool`` fed 60
engines from ONE python process and the GIL-serial stdio feed starved them
(measured ~170% CPU / 9% machine). Here each worker owns a SINGLE engine and its
own GIL, so K worker processes keep K engines genuinely busy → the machine
saturates. Workers write their OWN shards (planes never cross the IPC boundary);
only small canonical child descriptors go back to the coordinator for dedup.

Protocol (multiprocessing Queues):
  * work_q yields canonical ``GameState`` boards to analyze, or ``None`` = stop.
  * result_q receives ``WorkerResult`` (n_examples, child (key, state) pairs, or
    a fatal flag). The coordinator dedups child keys and enqueues novel ones.
Idle-exit: if work_q is empty for ``idle_exit_s`` (coordinator died / drained),
flush the shard and exit cleanly rather than hang forever.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass, field

import numpy as np

from gomoku.external_engine import (
    ExternalEngineConfig, ExternalEnginePlayer, ExternalEngineError,
)
from gomoku.game import GameState
from gomoku.rapfimine.canonical import canonical_state, canonical_key
from gomoku.rapfimine.store import ShardWriter


@dataclass
class WorkerResult:
    board_id: int = -1
    n_examples: int = 0
    children: list = field(default_factory=list)  # list[(key_bytes, GameState)]


def _make_engine(cmd: str, board_size: int, timeout_ms: int) -> ExternalEnginePlayer:
    return ExternalEnginePlayer(ExternalEngineConfig(
        cmd=cmd, board_size=board_size, incremental=False, timeout_ms=timeout_ms))


def worker_main(*, worker_id: int, work_q, result_q, cmd: str, out_dir: str,
                board_size: int, max_node: int, max_pv: int, expand_k: int,
                timeout_ms: int, shard_size: int, idle_exit_s: float = 30.0) -> None:
    writer = ShardWriter(out_dir, worker_id, shard_size=shard_size)
    eng = _make_engine(cmd, board_size, timeout_ms)
    try:
        while True:
            try:
                item = work_q.get(timeout=idle_exit_s)
            except queue.Empty:
                break  # coordinator gone / fully drained — exit cleanly
            if item is None:
                break  # explicit stop sentinel
            board_id, state = item  # (int id, GameState)

            # Analyze (respawn the engine once on a death, then skip on a second).
            wr = None
            for attempt in (0, 1):
                try:
                    wr = eng.analyze(state, max_node=max_node, max_pv=max_pv)
                    break
                except ExternalEngineError:
                    try:
                        eng.close()
                    except Exception:
                        pass
                    eng = _make_engine(cmd, board_size, timeout_ms)
                    wr = None
            if not wr:
                result_q.put(WorkerResult(board_id=board_id))  # done (empty); no expansion
                continue

            best = max(wr.items(), key=lambda kv: kv[1])[0]
            writer.add(
                planes=np.asarray(state.to_planes(), dtype=np.float16),
                winrates={int(a): float(w) for a, w in wr.items()},
                key=canonical_key(state),
                side=int(state.move_count % 2),
                ply=int(state.move_count),
            )

            # Expansion: top-k legal children, each canonicalized (coordinator dedups).
            children = []
            done, _ = state.is_terminal()
            if not done:
                legal = {int(a) for a in state.legal_actions()}
                kept = 0
                for a, _w in sorted(wr.items(), key=lambda kv: kv[1], reverse=True):
                    if kept >= expand_k:
                        break
                    if int(a) not in legal:
                        continue
                    child = state.apply(int(a))
                    canon, _sym = canonical_state(child)
                    children.append((canonical_key(canon), canon))
                    kept += 1
            result_q.put(WorkerResult(board_id=board_id, n_examples=1, children=children))
    finally:
        try:
            writer.close()
        except Exception:
            pass
        try:
            eng.close()
        except Exception:
            pass
