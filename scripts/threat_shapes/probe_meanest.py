"""Probe: how does solve_vct_mega_bb treat synthetic all-white 'meanest' boards?
Decides whether the stencil minimizer's 'fill DC with white' certificate is usable."""
import os
os.environ.setdefault("GOMOKU_BOARD_SIZE", "15")
import gzip, json
import numpy as np
from gomoku.board_config import BOARD_SIZE as N
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb

rows = [json.loads(l) for l in gzip.open(os.path.expanduser(
    "~/data/vct_first/first_vct.jsonl.gz"), "rt")]


def board_from(atk, dfd):
    b = np.zeros((2, N, N), bool)
    for i in atk:
        b[0, i // N, i % N] = True
    for i in dfd:
        b[1, i // N, i % N] = True
    return b


def whiten(b, keep_empty=()):
    bb = b.copy()
    occ = b[0] | b[1]
    for i in range(N * N):
        if not occ[i // N, i % N] and i not in keep_empty:
            bb[1, i // N, i % N] = True
    return bb


def count_white5(b):
    """How many defender (board[1]) five-in-a-rows exist (sanity on illegality)."""
    P = b[1].astype(int)
    n = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(N):
            for c in range(N):
                if all(0 <= r + k * dr < N and 0 <= c + k * dc < N and P[r + k * dr, c + k * dc]
                       for k in range(5)):
                    n += 1
    return n


tests, labels, meta = [], [], []
for r in rows[:6]:
    b = board_from(r["atk"], r["dfd"])
    p = r["move"]
    tests.append(b); labels.append(f"orig    ply{r['ply']:>2} md{r['md']:>2}"); meta.append(0)
    w = whiten(b)
    tests.append(w); labels.append("  all-white-DC (full)"); meta.append(count_white5(w))
    wk = whiten(b, keep_empty={p})
    tests.append(wk); labels.append("  white-DC keep p empty"); meta.append(count_white5(wk))

batch = np.stack(tests)
win, cap = solve_vct_mega_bb(batch, max_nodes=20000)
print(f"N={N}  {len(batch)} boards")
for l, wv, cv, m in zip(labels, win, cap, meta):
    extra = f"  (defender-5s={m})" if m else ""
    print(f"  {l:28s} win={str(bool(wv)):5s} cap={bool(cv)}{extra}")
