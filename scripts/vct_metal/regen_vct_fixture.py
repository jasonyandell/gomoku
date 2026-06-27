"""One-shot golden-fixture generator for the GPU VCT megakernel fast test.

Run ON-DEMAND only (it deliberately uses the RETIRED CPU oracle as ground
truth, so it sets ``GOMOKU_ALLOW_CPU_SOLVER=1`` for its own process):

    GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_metal.regen_vct_fixture

It draws a fixed, seeded stack of real Rapfi positions, solves each with the CPU
oracle ``gomoku.vcf.solve_vct`` at a high budget, KEEPS ONLY clean (non-cap)
boards (those have definitive truth), and writes the verdict truth plus the GPU
``complete`` winmask (captured at high budget, as a regression baseline) to a
small COMMITTED ``fixtures/vct_golden.npz``. The seed + budget are recorded
inside so the fixture is fully reproducible.

The fast test (``test_mega_vct_bb.py``) then loads this npz and NEVER touches the
CPU solver or the slow cell-scan ``mega_vct`` — see wiki/topics/mega-vct-solver.md
(§ CPU solver retired) for the tiering rationale.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Sanctioned deliberate use of the retired CPU oracle (must precede the import).
os.environ.setdefault("GOMOKU_ALLOW_CPU_SOLVER", "1")

from gomoku import state_ops  # noqa: E402
from gomoku.vcf import solve_vct  # noqa: E402
from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb  # noqa: E402
from scripts.vct_metal.positions import load_position_stack  # noqa: E402

N = state_ops.BOARD_SIZE
FIXTURE = Path(__file__).parent / "fixtures" / "vct_golden.npz"

# Fixed generation parameters baked into the fixture for reproducibility.
SEED = 0
N_BOARDS = 72  # kept small: the CPU oracle is the slow part; clean subset is fewer
MIN_PLY, MAX_PLY = 6, 40
ORACLE_MAX_DEPTH = 7        # gomoku.vcf.DEFAULT_VCT_MAX_DEPTH
# Budget bounds per-board CPU time (capped boards are discarded anyway). A board
# that does NOT cap at this budget is definitively solved -> sound golden truth;
# the DEEP tier (validate_deep.py) covers the high-budget / hard tail with live
# vcf. Kept modest so fixture-gen stays brief and never hits vcf's deep-search
# tail (which is exactly the wall this retirement removes from the gate).
ORACLE_MAX_NODES = 2_000


def main() -> int:
    if N != 15:
        raise SystemExit(
            f"BOARD_SIZE={N}; the Rapfi fixture is 15x15 — run with GOMOKU_BOARD_SIZE=15"
        )
    st = load_position_stack(N_BOARDS, seed=SEED, min_ply=MIN_PLY, max_ply=MAX_PLY)

    win = np.zeros(N_BOARDS, dtype=bool)
    hit = np.zeros(N_BOARDS, dtype=bool)
    move = np.full(N_BOARDS, -1, dtype=np.int32)
    for b in range(N_BOARDS):
        res = solve_vct(st[b], max_depth=ORACLE_MAX_DEPTH, max_nodes=ORACLE_MAX_NODES)
        win[b] = bool(res.has_forced_win)
        hit[b] = bool(res.hit_cap)
        if res.winning_move is not None:
            move[b] = int(res.winning_move)

    # KEEP ONLY clean (non-cap) boards — definitive CPU truth.
    clean = ~hit
    st_c = st[clean]
    win_c = win[clean]
    move_c = move[clean]

    # GPU `complete` winmask at high budget — the regression baseline the fast
    # test diffs against (GPU-vs-GPU, but pins the kernel's winning-move set).
    wc, hc, winmask = solve_vct_mega_bb(
        st_c, max_nodes=ORACLE_MAX_NODES, complete=True)
    if hc.any():
        # Drop any board the GPU complete-solve capped at high budget too.
        keep = ~hc
        st_c, win_c, move_c, winmask = st_c[keep], win_c[keep], move_c[keep], winmask[keep]
        wc = wc[keep]
    # Sanity: at high budget the GPU verdict must already agree with the oracle.
    bad = np.where(win_c != wc)[0]
    if bad.size:
        raise SystemExit(
            f"oracle/GPU verdict disagree on {bad.size} clean boards at high budget "
            f"(indices {list(bad)}) — investigate before banking the fixture"
        )

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIXTURE,
        boards=st_c,
        win=win_c,
        move=move_c,
        winmask=winmask,
        board_size=np.array(N, dtype=np.int32),
        seed=np.array(SEED, dtype=np.int32),
        oracle_max_depth=np.array(ORACLE_MAX_DEPTH, dtype=np.int32),
        oracle_max_nodes=np.array(ORACLE_MAX_NODES, dtype=np.int32),
        min_ply=np.array(MIN_PLY, dtype=np.int32),
        max_ply=np.array(MAX_PLY, dtype=np.int32),
    )
    print(
        f"wrote {FIXTURE} : {st_c.shape[0]}/{N_BOARDS} clean boards "
        f"(wins={int(win_c.sum())}), N={N}, seed={SEED}, "
        f"oracle budget depth={ORACLE_MAX_DEPTH} nodes={ORACLE_MAX_NODES}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
