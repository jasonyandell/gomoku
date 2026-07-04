"""Append-only sharded store for the idx-2 mine — crash-robust by construction.

Each worker owns its own shard stream (``shard_w{wid}_{seq}.npz``) so there is no
cross-process write contention. A shard is written whole via temp-file + atomic
``os.replace`` and only THEN counted, so a crash mid-write never leaves a torn
shard a reader can see. Resume rebuilds the global canonical dedup-set from the
``keys`` array stored in every complete shard and re-walks the BFS — any board
that was enqueued-but-not-yet-analyzed at crash time is simply re-discovered from
its (already-stored) parent, so no special in-flight journal is needed.

Shard payload mirrors ``gomoku.teacher.save_teacher_npz`` (so the trainer's
loader is unchanged) plus a ``keys`` (M, 16) uint8 array = the per-row
canonical dedup key, used only for resume.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from gomoku.board_config import BOARD_SIZE, N_ACTIONS

# float16 winrate floor mirrors teacher.py (a genuinely-scored move stays > 0).
_SOFT_FLOOR = np.float16(1e-4)


class ShardWriter:
    """Buffers examples for ONE worker; flushes whole shards atomically."""

    def __init__(self, out_dir: str, worker_id: int, shard_size: int = 50_000):
        self.out_dir = out_dir
        self.worker_id = worker_id
        self.shard_size = shard_size
        os.makedirs(out_dir, exist_ok=True)
        self._seq = self._next_seq()
        self._buf: list = []  # list of (planes_f16, winrates_dict, key_bytes, side, ply)
        self.n_written = 0

    def _next_seq(self) -> int:
        existing = glob.glob(os.path.join(self.out_dir, f"shard_w{self.worker_id}_*.npz"))
        seqs = []
        for p in existing:
            try:
                seqs.append(int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
            except (ValueError, IndexError):
                pass
        return (max(seqs) + 1) if seqs else 0

    def add(self, *, planes: np.ndarray, winrates: dict, key: bytes,
            side: int, ply: int) -> None:
        self._buf.append((planes.astype(np.float16), winrates, key, side, ply))
        if len(self._buf) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        M = len(self._buf)
        planes = np.stack([b[0] for b in self._buf])                       # (M,17,N,N) f16
        soft = np.zeros((M, N_ACTIONS), dtype=np.float16)
        moves = np.zeros(M, dtype=np.int32)
        for i, (_pl, wr, _k, _s, _p) in enumerate(self._buf):
            best_a, best_w = -1, -1.0
            for a, w in wr.items():
                wv = max(float(_SOFT_FLOOR), float(w))
                soft[i, int(a)] = np.float16(wv)
                if w > best_w:
                    best_w, best_a = w, int(a)
            moves[i] = best_a
        keys = np.stack([np.frombuffer(b[2], dtype=np.uint8) for b in self._buf])  # (M,16)
        side = np.asarray([b[3] for b in self._buf], dtype=np.int8)
        ply = np.asarray([b[4] for b in self._buf], dtype=np.int16)

        path = os.path.join(self.out_dir, f"shard_w{self.worker_id}_{self._seq}.npz")
        tmp = path + ".tmp"
        # Pass a file HANDLE: np.savez(<str>) appends '.npz' to a name that lacks
        # it (so 'x.npz.tmp' -> 'x.npz.tmp.npz'); a handle writes the exact path.
        with open(tmp, "wb") as fh:
            np.savez(  # uncompressed: speed > size on the hot path; merge can recompress
                fh,
                planes=planes, soft_policy=soft, moves=moves, keys=keys,
                side=side, ply=ply,
                board_size=np.int32(BOARD_SIZE), n_actions=np.int32(N_ACTIONS),
                teacher_version=np.int32(2),
            )
        os.replace(tmp, path)  # atomic: a reader never sees a partial shard
        self.n_written += M
        self._seq += 1
        self._buf = []

    def close(self) -> None:
        self.flush()


def iter_shard_paths(out_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(out_dir, "shard_w*_*.npz")))


def load_seen_keys(out_dir: str) -> set[bytes]:
    """Rebuild the global canonical dedup-set from every complete shard (resume)."""
    seen: set[bytes] = set()
    for p in iter_shard_paths(out_dir):
        try:
            with np.load(p) as z:
                for row in z["keys"]:
                    seen.add(row.tobytes())
        except (OSError, KeyError, ValueError):
            # a torn/partial shard (shouldn't happen with atomic replace) — skip
            continue
    return seen


def count_examples(out_dir: str) -> int:
    n = 0
    for p in iter_shard_paths(out_dir):
        try:
            with np.load(p) as z:
                n += int(z["moves"].shape[0])
        except (OSError, KeyError, ValueError):
            continue
    return n


# --------------------------------------------------------------------------
# Frontier checkpoint — the pending-work half of the durable log.
#
# Shards are the append-only log of COMPLETED work; the frontier checkpoint is a
# periodic immutable snapshot of PENDING work (enqueued + in-flight canonical
# boards). Written via temp+atomic-replace, newest wins. On resume the frontier
# is reloaded and filtered against the seen-set (anything analyzed since the
# snapshot is dropped). A SIGKILL between snapshots loses at most the branches
# discovered in that window — never a completed example.
# --------------------------------------------------------------------------
_FRONTIER = "frontier.pkl"


def save_frontier(out_dir: str, boards: list) -> None:
    path = os.path.join(out_dir, _FRONTIER)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(boards, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_frontier(out_dir: str) -> list:
    path = os.path.join(out_dir, _FRONTIER)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return []
