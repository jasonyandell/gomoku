"""Bitboard primitives — golden host reference (Python big-ints).

The megakernels' dominant cost is per-node detection done by *cell scanning*
(O(N²) per board, and the per-move soundness check is O(N²) per move → O(N⁴) per
node). The fix is bitboard detection: pack each colour's stones into a bitset and
find fives / four-completions with shift-AND, O(1) in the board area.

This module is the GOLDEN REFERENCE for that bit-twiddling, using arbitrary-width
Python ints so there is no word-splitting to get wrong. The Metal port (`bb.py`)
packs the same bitset into `ulong[4]` (256 bits ≥ N*N=225) and must reproduce
these results bit-for-bit; that is validated in `test_bb.py`.

Bit layout: cell (r,c) → bit index r*N + c (row-major, little-endian).

Validated here against `detect_ref.five_completion_mask` / `four_structure`
(themselves validated against `gomoku.vcf`).
"""
from __future__ import annotations

import numpy as np

from gomoku import state_ops

N = state_ops.BOARD_SIZE
WIN = 5
NN = N * N
BOARDMASK = (1 << NN) - 1

# Four line directions as (step-in-bits, start-validity predicate on (r,c)).
# A length-5 window starting at cell p=(r,c) and walking +step stays on one line
# iff the start cell satisfies the predicate (this subsumes row-wrap correctness).
_DIR_DEFS = [
    ("H", 1, lambda r, c: c <= N - WIN),                    # horizontal  dc=+1
    ("V", N, lambda r, c: r <= N - WIN),                    # vertical    dr=+1
    ("D1", N + 1, lambda r, c: r <= N - WIN and c <= N - WIN),  # diag down-right
    ("D2", N - 1, lambda r, c: r <= N - WIN and c >= WIN - 1),  # diag down-left
]


def _startmask(pred) -> int:
    m = 0
    for r in range(N):
        for c in range(N):
            if pred(r, c):
                m |= 1 << (r * N + c)
    return m


# (name, step, startmask) precomputed once.
DIRS = [(name, step, _startmask(pred)) for name, step, pred in _DIR_DEFS]


def pack(grid: np.ndarray) -> int:
    """(N,N) bool/0-1 array → bitset int (bit r*N+c)."""
    flat = np.ascontiguousarray(grid).reshape(-1).astype(bool)
    out = 0
    for i in np.nonzero(flat)[0]:
        out |= 1 << int(i)
    return out


def unpack(bits: int) -> np.ndarray:
    """bitset int → (N,N) bool array."""
    out = np.zeros(NN, dtype=bool)
    b = bits & BOARDMASK
    while b:
        low = b & (-b)
        out[low.bit_length() - 1] = True
        b ^= low
    return out.reshape(N, N)


def has_five(P: int) -> bool:
    """True iff bitset P contains >= WIN in a row (freestyle: overlines count)."""
    for _name, s, mask in DIRS:
        t = P & (P >> s) & (P >> (2 * s)) & (P >> (3 * s)) & (P >> (4 * s)) & mask
        if t:
            return True
    return False


def completion_mask(P: int, empty: int) -> int:
    """Empty cells e (subset of `empty`) where placing a P-stone makes >= WIN.

    Equivalent to detect_ref.five_completion_mask for the side `P`. For each
    direction and each hole position j in the 5-window, AND the four non-hole
    cells (P) with the hole cell (empty), then shift the start-cell mask to the
    hole cell.
    """
    comp = 0
    for _name, s, mask in DIRS:
        for j in range(WIN):
            m = mask
            for i in range(WIN):
                src = empty if i == j else P
                m &= src >> (i * s)
            comp |= (m << (j * s))
    return comp & BOARDMASK & empty


def four_count_block(own: int, opp: int, m: int):
    """If side `own` plays cell index m, (#five-completions, a-completion-cell).

    Mirrors detect_ref.four_structure per move: n>=1 ⇒ m makes a four (forcing),
    n>=2 ⇒ double/open four (win-next). Returns (n, block) with block=-1 if n==0.
    """
    mb = 1 << m
    own2 = own | mb
    empty2 = (~(own2 | opp)) & BOARDMASK     # empties after playing m (excludes m)
    comp = completion_mask(own2, empty2)
    n = bin(comp).count("1")
    block = (comp & (-comp)).bit_length() - 1 if comp else -1
    return n, block


# --------------------------------------------------------------------------- #
# self-validation vs detect_ref (which is validated vs gomoku.vcf)
# --------------------------------------------------------------------------- #
def _gen_boards(n, seed, density=0.16):
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n):
        occ = rng.random((N, N)) < density
        who = rng.random((N, N)) < 0.5
        own = occ & who
        opp = occ & ~who
        boards.append((own, opp))
    return boards


def _selftest(n=400, seed=0):
    from scripts.vct_metal import detect_ref as dref

    fail = {"completion": 0, "has_five": 0, "four_count": 0}
    for own_g, opp_g in _gen_boards(n, seed):
        own = pack(own_g)
        opp = pack(opp_g)
        empty = (~(own | opp)) & BOARDMASK

        # 1. completion_mask == detect_ref.five_completion_mask
        cm = unpack(completion_mask(own, empty))
        ref = dref.five_completion_mask(own_g[None], opp_g[None])[0]
        if not np.array_equal(cm, ref):
            fail["completion"] += 1

        # 2. has_five == brute-force five-in-a-row for own
        if has_five(own) != _brute_five(own_g):
            fail["has_five"] += 1

        # 3. four_count_block per empty m == four_structure.
        # four_structure counts completions *through m*; four_count_block counts
        # all completions of own∪{m}. They agree exactly when own has no
        # pre-existing completion — which is the only regime the OR node uses it
        # (it tests immediate-five before ever counting fours). Gate on that.
        if completion_mask(own, empty) == 0:
            n_comp, _block = dref.four_structure(own_g[None], opp_g[None])
            n_comp = n_comp[0].reshape(-1)
            emp_idx = np.nonzero(unpack(empty).reshape(-1))[0]
            for m in emp_idx:
                nc, _bk = four_count_block(own, opp, int(m))
                if nc != int(n_comp[m]):
                    fail["four_count"] += 1
                    break
    return fail


def _brute_five(grid: np.ndarray) -> bool:
    g = grid.astype(bool)
    for r in range(N):
        for c in range(N):
            if not g[r, c]:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cnt = 0
                rr, cc = r, c
                while 0 <= rr < N and 0 <= cc < N and g[rr, cc]:
                    cnt += 1
                    rr += dr
                    cc += dc
                if cnt >= WIN:
                    return True
    return False


if __name__ == "__main__":
    import sys

    total = {"completion": 0, "has_five": 0, "four_count": 0}
    for s in range(3):
        f = _selftest(n=300, seed=s)
        for k in total:
            total[k] += f[k]
        print(f"seed={s}: {f}")
    bad = sum(total.values())
    print(f"totals: {total}")
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)
