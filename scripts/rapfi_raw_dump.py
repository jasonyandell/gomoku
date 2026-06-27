"""Measurement, not a fix: drive ONE Rapfi engine through the exact analyze()
preamble, then dump the RAW YXNBEST line stream to disk instead of parsing it.

Reproduces the '2000+ lines without a bestmove terminator' crash on a single
15x15 board and lets us READ what Rapfi actually emits before touching the
parser. Run with GOMOKU_BOARD_SIZE=15.
"""
import os
import sys

os.environ.setdefault("GOMOKU_BOARD_SIZE", "15")

from gomoku.game import GameState
from gomoku.eval_panel import IDX2_OPENING, fixed_opening_state
from gomoku.external_engine import ExternalEnginePlayer, ExternalEngineConfig
from gomoku.rapfi_pool import default_rapfi_cmd


def make_engine():
    cfg = ExternalEngineConfig(cmd=default_rapfi_cmd(), board_size=15,
                               incremental=False, timeout_ms=1000)
    return ExternalEnginePlayer(cfg)


def descend(state, k, levels):
    """Play `levels` plies from `state`, one fresh engine per parsed analyze."""
    s = state
    chain = [s]
    for _ in range(levels):
        eng = make_engine()
        print(f"  descend: type={type(eng)!r} has_analyze={hasattr(eng, 'analyze')} "
              f"module={type(eng).__module__}")
        try:
            wr = eng.analyze(s, max_node=20000, max_pv=k)
        finally:
            eng.close()
        if not wr:
            break
        best = max(wr.items(), key=lambda kv: kv[1])[0]
        s = s.apply(int(best))
        chain.append(s)
    return chain


def raw_dump(eng, state, *, max_pv, out_path, hard_cap=8000):
    """Replicate analyze()'s preamble, then read RAW lines from YXNBEST."""
    n = eng.config.board_size
    own = state.board[0]
    opp = state.board[1]
    own_cells = {(x, y) for y in range(n) for x in range(n) if own[y, x]}
    opp_cells = {(x, y) for y in range(n) for x in range(n) if opp[y, x]}
    moves_left = n * n - len(own_cells) - len(opp_cells)

    eng._send("INFO max_node 20000")
    eng._send("INFO SHOW_DETAIL 2")
    eng._send("INFO CAUTION_FACTOR 5")
    eng._send("RESTART")
    eng._expect_ok()
    eng._send("BOARD")
    for x, y in sorted(own_cells):
        eng._send(f"{x},{y},1")
    for x, y in sorted(opp_cells):
        eng._send(f"{x},{y},2")
    eng._send("DONE")
    eng._read_move()  # drain the default-multiPV bestmove

    pv_count = moves_left if max_pv is None else max(1, min(int(max_pv), moves_left))
    eng._send(f"YXNBEST {pv_count}")

    deadline = eng._read_deadline_s()
    n_lines = 0
    saw_terminator = False
    with open(out_path, "w") as f:
        f.write(f"# moves_left={moves_left} pv_count={pv_count} stones={len(own_cells)+len(opp_cells)}\n")
        for i in range(hard_cap):
            try:
                line = eng._read_line(deadline)
            except Exception as e:
                f.write(f"[{i}] <<READ ERROR: {e}>>\n")
                break
            n_lines += 1
            f.write(f"[{i}] {line}\n")
            # The real parser's terminator: a non-INFO line that parses as a coord.
            from gomoku.external_engine import _parse_coord
            if not line.startswith("INFO") and line and _parse_coord(line) is not None:
                saw_terminator = True
                f.write(f"[{i}] <<TERMINATOR: bare coord bestmove>>\n")
                break
    return n_lines, saw_terminator, moves_left, pv_count


def main():
    print(f"rapfi cmd: {default_rapfi_cmd()}")
    base = fixed_opening_state(IDX2_OPENING)
    print(f"idx-2 base: stones={int(base.board[0].sum()+base.board[1].sum())} ply={base.move_count}")

    # Build a chain of progressively deeper boards (the BFS principal variation).
    chain = descend(base, k=8, levels=7)
    print(f"descended to {len(chain)} boards (ply {chain[0].move_count}..{chain[-1].move_count})")

    # Raw-dump pv8 at EACH depth with a high cap, to find where lines cross 2000
    # AND confirm the terminator still arrives past it. Fresh engine each time.
    print("\ndepth  ply  stones  lines  terminator  out")
    for d, s in enumerate(chain):
        stones = int(s.board[0].sum() + s.board[1].sum())
        out = f"/tmp/rapfi_raw_d{d}_pv8.txt"
        eng = make_engine()
        try:
            n, term, ml, pv = raw_dump(eng, s, max_pv=8, out_path=out, hard_cap=12000)
        finally:
            eng.close()
        flag = "" if term else "  <<<NO TERMINATOR (would crash at 2000)"
        print(f"{d:5d}  {s.move_count:3d}  {stones:6d}  {n:5d}  {str(term):10s}  {out}{flag}")


if __name__ == "__main__":
    sys.exit(main())
