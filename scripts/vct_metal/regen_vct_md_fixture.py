"""One-shot GPU-SELF golden-fixture generator for the depth_cap / md_min FAST test.

Run on-demand:
    GOMOKU_BOARD_SIZE=15 PYTHONPATH=. uv run python -m scripts.vct_metal.regen_vct_md_fixture

Unlike regen_vct_fixture.py, this touches NO CPU oracle: md_min is GPU-self-validated
(order-independent depth-capped binary search — issue #91), and the verdict half
(``md>=1 <=> win``) is cross-checked against the high-budget GPU solve here and, in
the FAST test, against the already-committed ``vct_golden.npz`` verdict labels. The
adversary review (workflow w51fen2a4) showed a *live* CPU md cross-check is
mis-calibrated (the kernel's ``candidate_own`` is narrower than CPU's any-stone
candidate set ⇒ md_gpu>md_cpu with no bug) AND re-summons the retired solver — so
md is validated GPU-self + verdict-golden, per wiki/topics/mega-vct-solver.md.

Reuses the SAME seeded clean boards as vct_golden.npz (load_position_stack 72/seed0)
so the two fixtures' boards align. A non-capped md is budget-independent, so the
high-budget md banked here is the definitive truth the tight-budget FAST test must
reproduce on every board it does not cap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gomoku import state_ops
from scripts.vct_metal.mega_vct_bb import MAXD, solve_md_min, solve_vct_mega_bb
from scripts.vct_metal.positions import load_position_stack

N = state_ops.BOARD_SIZE
FIXTURE = Path(__file__).parent / "fixtures" / "vct_md_golden.npz"

SEED = 0
N_BOARDS = 72
MIN_PLY, MAX_PLY = 6, 40
ORACLE_MAX_NODES = 8_000    # high budget: a non-capped md is definitive truth


def main() -> int:
    if N != 15:
        raise SystemExit(f"BOARD_SIZE={N}; run with GOMOKU_BOARD_SIZE=15")
    st = load_position_stack(N_BOARDS, seed=SEED, min_ply=MIN_PLY, max_ply=MAX_PLY)

    md, capped = solve_md_min(st, max_nodes=ORACLE_MAX_NODES, hi=MAXD - 2)
    w0, h0, move = solve_vct_mega_bb(st, max_nodes=ORACLE_MAX_NODES, return_move=True)

    # Keep boards where BOTH the md search and the default solve are clean.
    keep = (~capped) & (~h0)
    st, md, w0, move = st[keep], md[keep], w0[keep], move[keep]

    # md>=1 must equal the default verdict (the definitive consistency the FAST
    # test re-checks against the committed verdict golden).
    bad = np.where((md >= 1) != w0)[0]
    if bad.size:
        raise SystemExit(f"md/verdict disagree on clean boards {list(bad)} at high "
                         f"budget — investigate before banking")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIXTURE,
        boards=st,
        md=md.astype(np.int32),
        win=w0,
        move=move.astype(np.int32),
        board_size=np.array(N, dtype=np.int32),
        seed=np.array(SEED, dtype=np.int32),
        oracle_max_nodes=np.array(ORACLE_MAX_NODES, dtype=np.int32),
        min_ply=np.array(MIN_PLY, dtype=np.int32),
        max_ply=np.array(MAX_PLY, dtype=np.int32),
    )
    alive = md >= 1
    print(f"wrote {FIXTURE}: {st.shape[0]}/{N_BOARDS} clean boards (wins={int(w0.sum())}), "
          f"md range [{int(md[alive].min()) if alive.any() else 0},"
          f"{int(md.max())}], N={N}, seed={SEED}, budget={ORACLE_MAX_NODES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
