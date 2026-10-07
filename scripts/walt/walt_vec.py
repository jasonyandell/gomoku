"""Vectorized lockstep rollouts for walt-gomoku, sized to the GPU oracle's native batch.

All live (leaf, tape) rollouts are held as numpy arrays and advanced one ply at a
time: one VCT call + one net call per ply for the whole lot. Tape draws are keyed
on a zobrist hash of the canonical board (the tickertape), shared with the tree.
"""
from __future__ import annotations

import numpy as np

from gomoku.game import BOARD_SIZE, HISTORY_PLY, N_INPUT_PLANES

N = BOARD_SIZE
A = N * N
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_ZOB = np.random.default_rng(0xC0FFEE).integers(0, 2**63, size=(2, A), dtype=np.uint64) * np.uint64(2) + np.uint64(1)


def zobrist(boards: np.ndarray) -> np.ndarray:
    """(R,2,N,N) bool canonical boards -> (R,) uint64."""
    b = boards.reshape(len(boards), 2, A)
    return np.bitwise_xor.reduce(np.where(b, _ZOB[None], np.uint64(0)), axis=(1, 2))


def splitmix_u(x: np.ndarray) -> np.ndarray:
    """uint64 -> uniform float64 in [0,1)."""
    with np.errstate(over="ignore"):
        z = (x + np.uint64(0x9E3779B97F4A7C15)) & _M64
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def tape_u(boards: np.ndarray, tapes: np.ndarray) -> np.ndarray:
    return splitmix_u(zobrist(boards) ^ tapes.astype(np.uint64))


def draw(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    c = np.cumsum(probs, axis=1)
    idx = (c < (u * c[:, -1])[:, None]).sum(axis=1)
    return np.minimum(idx, A - 1)


def five(plane: np.ndarray) -> np.ndarray:
    """(R,N,N) bool -> (R,) has a 5-in-a-row (overlines count: free-style)."""
    p = plane
    h = p[:, :, 0:N - 4] & p[:, :, 1:N - 3] & p[:, :, 2:N - 2] & p[:, :, 3:N - 1] & p[:, :, 4:N]
    v = p[:, 0:N - 4] & p[:, 1:N - 3] & p[:, 2:N - 2] & p[:, 3:N - 1] & p[:, 4:N]
    d = p[:, 0:N - 4, 0:N - 4] & p[:, 1:N - 3, 1:N - 3] & p[:, 2:N - 2, 2:N - 2] & p[:, 3:N - 1, 3:N - 1] & p[:, 4:N, 4:N]
    e = p[:, 0:N - 4, 4:N] & p[:, 1:N - 3, 3:N - 1] & p[:, 2:N - 2, 2:N - 2] & p[:, 3:N - 1, 1:N - 3] & p[:, 4:N, 0:N - 4]
    return h.any((1, 2)) | v.any((1, 2)) | d.any((1, 2)) | e.any((1, 2))


def near_mask(boards: np.ndarray, radius: int = 2) -> np.ndarray:
    occ = boards[:, 0] | boards[:, 1]
    near = occ.copy()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            s = np.zeros_like(occ)
            rs, re_ = max(dr, 0), N + min(dr, 0)
            cs, ce = max(dc, 0), N + min(dc, 0)
            s[:, rs:re_, cs:ce] = occ[:, rs - dr:re_ - dr, cs - dc:ce - dc]
            near |= s
    m = (near & ~occ).reshape(len(boards), A)
    empty = ~occ.reshape(len(boards), A)
    none = ~m.any(1)
    m[none] = empty[none]
    centre = ~occ.any((1, 2))
    if centre.any():
        m[centre] = False
        m[centre, (N // 2) * N + N // 2] = True
    return m


def planes(boards: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """Vectorized GameState.to_planes. hist: (R,H-1,2,N,N), most-recent first, zeros if absent."""
    H = HISTORY_PLY
    R = len(boards)
    out = np.zeros((R, N_INPUT_PLANES, N, N), dtype=np.float32)
    out[:, 0] = boards[:, 0]
    out[:, H] = boards[:, 1]
    for k in range(1, H):
        pb = hist[:, k - 1]
        if k % 2 == 0:
            out[:, k], out[:, H + k] = pb[:, 0], pb[:, 1]
        else:
            out[:, k], out[:, H + k] = pb[:, 1], pb[:, 0]
    out[:, 2 * H] = 1.0
    return out


def field_probs(kind: str, evaluator, boards, hist, temperature: float) -> np.ndarray:
    legal = ~(boards[:, 0] | boards[:, 1]).reshape(len(boards), A)
    if kind == "uniform":
        m = near_mask(boards)
        return m / m.sum(1, keepdims=True)
    logits, _ = evaluator.evaluate_planes(planes(boards, hist))
    z = np.where(legal, np.asarray(logits, np.float64) / max(temperature, 1e-6), -np.inf)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def state_arrays(states) -> tuple[np.ndarray, np.ndarray]:
    boards = np.stack([s.board for s in states]).astype(bool)
    hist = np.zeros((len(states), HISTORY_PLY - 1, 2, N, N), bool)
    for i, s in enumerate(states):
        for k, hb in enumerate(s.history[:HISTORY_PLY - 1]):
            hist[i, k] = hb
    return boards, hist


def rollouts(boards, hist, walt_tm, tapes, *, kind, evaluator, temperature, vct, vct_every=1):
    """Lockstep rollouts. Both sides play the field (policy via tape + VCT finisher).

    Returns (payoff (R,) int in {0,1,2} walt POV, length (R,), ended_by_vct (R,) bool, batch sizes list).
    """
    R = len(boards)
    boards, hist = boards.copy(), hist.copy()
    walt_tm = walt_tm.copy()
    pay = np.full(R, -1, np.int64)
    length = np.zeros(R, np.int64)
    by_vct = np.zeros(R, bool)
    live = np.arange(R)
    batch_sizes = []
    ply = 0
    while live.size:
        b, h = boards[live], hist[live]
        batch_sizes.append(live.size)
        if ply % vct_every == 0:
            win, _, _ = vct.solve_boards(b)
        else:
            win = np.zeros(live.size, bool)
        if win.any():  # side to move has a forced win; finisher plays it
            idx = live[win]
            pay[idx] = np.where(walt_tm[idx], 2, 0)
            length[idx] = ply
            by_vct[idx] = True
            keep = ~win
            live, b, h = live[keep], b[keep], h[keep]
            if not live.size:
                break
        p = field_probs(kind, evaluator, b, h, temperature)
        a = draw(p, tape_u(b, tapes[live]))
        r, c = np.divmod(a, N)
        snap = b.copy()
        b = b.copy()
        b[np.arange(live.size), 0, r, c] = True
        won = five(b[:, 0])
        full = (b[:, 0] | b[:, 1]).all((1, 2))
        # flip perspective + push history
        b = b[:, ::-1].copy()
        h = np.concatenate([snap[:, None], h[:, :-1]], axis=1)
        done = won | full
        idx = live[done]
        pay[idx] = np.where(won[done], np.where(walt_tm[idx], 2, 0), 1)
        length[idx] = ply + 1
        cont = ~done
        live = live[cont]
        boards[live], hist[live] = b[cont], h[cont]
        walt_tm[live] = ~walt_tm[live]
        ply += 1
    return pay, length, by_vct, batch_sizes
