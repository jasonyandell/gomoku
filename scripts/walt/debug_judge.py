"""Trace a few judge playouts move by move."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_b_coverage import replay  # noqa: E402

from gomoku.rapfi_pool import RapfiPool  # noqa: E402

rows = json.load(open("sweep_logs/walt_sense/expB/choices.json"))
with RapfiPool(size=1, timeout_ms=50, board_size=9) as pool:
    for i in range(4):
        r = rows[i]
        m = r["choices"]["mcts+veto"]
        s0 = replay(r["moves"])[-1]
        print(f"pos {i} back {r['back']} ply {len(r['moves'])} defender_black {r['defender_black']} choice {divmod(m, 9)} rapfi_actual {divmod(r['rapfi_move'], 9)}")
        print(s0.render() if hasattr(s0, "render") else "")
        s = s0.apply(m)
        seq = []
        t = time.perf_counter()
        for n in range(81):
            a = pool.pick(s)
            seq.append(divmod(a, 9))
            s = s.apply(a)
            d, v = s.is_terminal()
            if d:
                break
        print(f"   rapfi moves {len(seq)} {seq[:12]} secs {time.perf_counter()-t:.2f} terminal {d} v {v}")
