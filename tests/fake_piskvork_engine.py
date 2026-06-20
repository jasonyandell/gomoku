#!/usr/bin/env python3
"""A tiny fake Gomocup/Piskvork-protocol engine for testing the wrapper.

It speaks just enough of the protocol to exercise gomoku.external_engine:
- `START <size>` -> prints OK (or ERROR if mode=reject_start)
- `RESTART`      -> prints OK, clears the board (per the real Piskvork protocol)
- `INFO ...`     -> ignored
- `BOARD` ... `DONE` -> read the position, then reply a move per the mode
- `SWAP2BOARD` ... `DONE` -> read the (0/3/5) `x,y,color` stone lines (color in
                {1,2}; a colorless `x,y` line is rejected with an ERROR like real
                Rapfi), reply a swap2 reply per the mode (see "swap2 modes" below)
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

swap2 modes (drive the SWAP2BOARD path; reply depends on stones-in-block):
  swap2_open    0 stones -> reply three coords; 3/5 -> a single coord
  swap2_swap    reply the literal `SWAP` (legal at 3 or 5 stones)
  swap2_one     reply a single coord (responder keeps color / picker plays)
  swap2_two     reply two coords (responder places 4th+5th)
  swap2_chatter emit MESSAGE/DEBUG + a stray `?` before a single-coord reply
  swap2_badarity reply a single coord even when 0 stones are in (arity error:
                the opener must return THREE coords)
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

    def read_swap2board() -> list[tuple[int, int]]:
        """Read the SWAP2BOARD stone block up to DONE; return the stones in order.

        Stone lines are `x,y,color` (the spec form the wrapper sends), with color
        in {1, 2} — 1 = black, 2 = white. A bare `x,y` line (no color) or an
        out-of-range color is a protocol violation here, surfaced as an ERROR line
        (mirroring real Rapfi, which rejects a colorless stone with
        ``ERROR Color is not a valid value, must be one of [1, 2, 3]``). Returns
        the (x, y) list so the reply can pick its arity off the count (0/3/5);
        color is validated but not needed to choose the reply."""
        stones: list[tuple[int, int]] = []
        for bline in sys.stdin:
            bl = bline.strip()
            if not bl:
                continue
            if bl.upper() == "DONE":
                break
            parts = bl.split(",")
            if len(parts) < 3:
                out.write("ERROR Color is not a valid value, must be one of [1, 2, 3].\n")
                out.flush()
                continue
            x, y, color = int(parts[0]), int(parts[1]), int(parts[2])
            if color not in (1, 2):
                out.write("ERROR Color is not a valid value, must be one of [1, 2, 3].\n")
                out.flush()
                continue
            stones.append((x, y))
        return stones

    def reply_swap2(stones: list[tuple[int, int]]) -> None:
        occ = set(stones)

        def lowest_empty(skip: set[tuple[int, int]] = frozenset()) -> tuple[int, int]:
            for yy in range(size):
                for xx in range(size):
                    if (xx, yy) not in occ and (xx, yy) not in skip:
                        return (xx, yy)
            return (0, 0)

        if mode == "swap2_chatter":
            out.write("MESSAGE thinking about the opening...\n")
            out.write("DEBUG swap2 eval\n")
            out.write("?\n")
        if mode == "swap2_swap":
            out.write("SWAP\n")
            out.flush()
            return
        if mode == "swap2_two":
            a = lowest_empty()
            b = lowest_empty({a})
            out.write(f"{a[0]},{a[1]} {b[0]},{b[1]}\n")
            out.flush()
            return
        if mode == "swap2_badarity":
            # Always a single coord, even when the engine was asked to OPEN (0
            # stones), where THREE coords are required -> arity mismatch.
            a = lowest_empty()
            out.write(f"{a[0]},{a[1]}\n")
            out.flush()
            return
        # swap2_open / swap2_one / swap2_chatter default: arity by stones-in.
        if len(stones) == 0:
            a = lowest_empty()
            b = lowest_empty({a})
            c = lowest_empty({a, b})
            out.write(f"{a[0]},{a[1]} {b[0]},{b[1]} {c[0]},{c[1]}\n")
        else:
            a = lowest_empty()
            out.write(f"{a[0]},{a[1]}\n")
        out.flush()

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
        elif upper == "SWAP2BOARD":
            stones = read_swap2board()
            reply_swap2(stones)
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
