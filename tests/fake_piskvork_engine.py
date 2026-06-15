#!/usr/bin/env python3
"""A tiny fake Gomocup/Piskvork-protocol engine for testing the wrapper.

It speaks just enough of the protocol to exercise gomoku.external_engine:
- `START <size>` -> prints OK (or ERROR if mode=reject_start)
- `RESTART`      -> prints OK, clears the board (per the real Piskvork protocol)
- `INFO ...`     -> ignored
- `BOARD` ... `DONE` -> read the position, then reply a move per the mode
- `TURN x,y` / `BEGIN` -> also reply a move
- `END` / EOF -> exit

Modes (argv[1]):
  lowest        play the lowest-index empty cell (deterministic, legal)
  illegal       always reply "0,0" even if occupied (to test rejection)
  reject_start  reply ERROR to START (to test unsupported-size handling)
  chatter       emit MESSAGE/DEBUG lines before the move (to test skipping)
  err_chatter   emit non-fatal "ERROR my move [..]" diagnostic lines before the
                move (mimics Yixin/Pela/Eulring: ERROR is chatter, not failure)
  boardonce     accept BOARD only ONCE per game; a second BOARD without an
                intervening RESTART errors + treats stone lines as top-level
                commands (mimics Zetor2017's one-shot BOARD desync)
  resign_empty  resign (no move + EOF) on an empty BOARD/DONE, but answer BEGIN
                (mimics Zetor2017 refusing to open via an empty BOARD)
"""

import sys


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "lowest"
    size = 9
    # occupied[(x,y)] = field (1 own / 2 opp)
    occupied: dict[tuple[int, int], int] = {}
    # For the 'boardonce' mode: has BOARD been consumed since the last RESTART?
    board_used = [False]

    out = sys.stdout

    def reply_move(*, empty_board: bool = False) -> None:
        if mode == "illegal":
            out.write("0,0\n")
            out.flush()
            return
        if mode == "resign_empty" and empty_board:
            out.write("MESSAGE No select move, resign...\n")
            out.flush()
            return  # close stdin loop -> EOF, mimicking Zetor's resign-on-empty
        if mode == "chatter":
            out.write("MESSAGE thinking...\n")
            out.write("DEBUG depth 1\n")
        if mode == "err_chatter":
            # Non-fatal diagnostics some engines emit BEFORE the real move.
            out.write("ERROR my move [7,7]\n")
            out.write("ERROR opponents's move [7,7]\n")
            out.write("DATABASE hit\n")
            out.write("?\n")
        for y in range(size):
            for x in range(size):
                if (x, y) not in occupied:
                    out.write(f"{x},{y}\n")
                    out.flush()
                    return
        # board full -> shouldn't happen in tests
        out.write("0,0\n")
        out.flush()

    def read_board() -> None:
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

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("START"):
            occupied.clear()
            board_used[0] = False
            try:
                size = int(line.split()[1])
            except (IndexError, ValueError):
                size = 9
            if mode == "reject_start":
                out.write("ERROR unsupported size (fake)\n")
            else:
                out.write("OK\n")
            out.flush()
        elif upper == "RESTART":
            occupied.clear()
            board_used[0] = False
            out.write("OK\n")
            out.flush()
        elif upper.startswith("INFO"):
            continue
        elif upper == "BOARD":
            if mode == "boardonce" and board_used[0]:
                # Zetor2017 desync: a second BOARD without RESTART errors, and
                # the following stone lines fall through as top-level commands.
                out.write("ERROR Board isn't initialized. Use: 'START size'\n")
                out.flush()
                continue
            board_used[0] = True
            read_board()
            reply_move(empty_board=not occupied)
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
            # Unknown command: ignore (mimics an engine that mis-parses stray
            # stone lines as commands after a desync).
            continue


if __name__ == "__main__":
    main()
