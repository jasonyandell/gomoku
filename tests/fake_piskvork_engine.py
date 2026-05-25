#!/usr/bin/env python3
"""A tiny fake Gomocup/Piskvork-protocol engine for testing the wrapper.

It speaks just enough of the protocol to exercise gomoku.external_engine:
- `START <size>` -> prints OK (or ERROR if mode=reject_start)
- `INFO ...`     -> ignored
- `BOARD` ... `DONE` -> read the position, then reply a move per the mode
- `TURN x,y` / `BEGIN` -> also reply a move
- `END` / EOF -> exit

Modes (argv[1]):
  lowest        play the lowest-index empty cell (deterministic, legal)
  illegal       always reply "0,0" even if occupied (to test rejection)
  reject_start  reply ERROR to START (to test unsupported-size handling)
  chatter       emit MESSAGE/DEBUG lines before the move (to test skipping)
"""

import sys


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "lowest"
    size = 9
    # occupied[(x,y)] = field (1 own / 2 opp)
    occupied: dict[tuple[int, int], int] = {}

    out = sys.stdout

    def reply_move() -> None:
        if mode == "illegal":
            out.write("0,0\n")
            out.flush()
            return
        if mode == "chatter":
            out.write("MESSAGE thinking...\n")
            out.write("DEBUG depth 1\n")
        for y in range(size):
            for x in range(size):
                if (x, y) not in occupied:
                    out.write(f"{x},{y}\n")
                    out.flush()
                    return
        # board full -> shouldn't happen in tests
        out.write("0,0\n")
        out.flush()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("START"):
            occupied.clear()
            try:
                size = int(line.split()[1])
            except (IndexError, ValueError):
                size = 9
            if mode == "reject_start":
                out.write("ERROR unsupported size (fake)\n")
            else:
                out.write("OK\n")
            out.flush()
        elif upper.startswith("INFO"):
            continue
        elif upper == "BOARD":
            occupied.clear()
            for bline in sys.stdin:
                bl = bline.strip()
                if bl.upper() == "DONE" or not bl:
                    if bl.upper() == "DONE":
                        break
                    continue
                parts = bl.split(",")
                if len(parts) >= 3:
                    x, y, field = int(parts[0]), int(parts[1]), int(parts[2])
                    occupied[(x, y)] = field
            reply_move()
        elif upper.startswith("TURN"):
            # opponent just moved at the given coord
            try:
                coord = line.split()[1]
                x, y = (int(v) for v in coord.split(","))
                occupied[(x, y)] = 2
            except (IndexError, ValueError):
                pass
            reply_move()
        elif upper == "BEGIN":
            reply_move()
        elif upper == "END":
            break
        else:
            # Unknown command: ignore.
            continue


if __name__ == "__main__":
    main()
