"""Process-level board-size configuration.

THE CONTRACT
============
Board size is a *process-level* constant, resolved exactly once, at the time
this module is first imported (which happens transitively on any
``import gomoku.game`` / ``gomoku.state_ops`` / ``gomoku.model``):

1. ``--board-size N`` / ``--board-size=N`` anywhere on the command line
   (highest precedence — this is how the CLI entry points set the size
   *before* heavy imports run, without threading a parameter through every
   module-level constant);
2. else the ``GOMOKU_BOARD_SIZE`` environment variable;
3. else the default, 9.

The resolved value is written back to ``os.environ["GOMOKU_BOARD_SIZE"]`` so
any subprocess this process spawns (self-play workers, eval workers, sweep
cells) inherits the same board size automatically.

Everything downstream (``game.BOARD_SIZE``, ``N_ACTIONS``, model head sizes,
replay-buffer shapes, the native-extension choice) derives from
``BOARD_SIZE``/``N_ACTIONS`` here. Changing the size after import is NOT
supported — start a new process instead.

Native extensions are compiled per size (9 is ``_state_ops_native`` /
``_mcts_native``; 15 is ``_state_ops_native15`` / ``_mcts_native15``). Other
sizes transparently fall back to the pure-Python/NumPy paths (equivalent to
``GOMOKU_DISABLE_NATIVE_*=1``).
"""

from __future__ import annotations

import os
import sys

ENV_VAR = "GOMOKU_BOARD_SIZE"
DEFAULT_BOARD_SIZE = 9
MIN_BOARD_SIZE = 5    # WIN_LEN; smaller boards cannot hold five-in-a-row
MAX_BOARD_SIZE = 25   # sanity bound (gomocup tops out at 20; 19 = go board)

# Board sizes that have a dedicated compiled native extension. Anything else
# uses the pure-Python/NumPy fallback paths.
NATIVE_BOARD_SIZES = (9, 15)


def _parse(raw: str, source: str) -> int:
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"invalid board size {raw!r} from {source}") from None
    if not (MIN_BOARD_SIZE <= n <= MAX_BOARD_SIZE):
        raise ValueError(
            f"board size {n} from {source} out of range "
            f"[{MIN_BOARD_SIZE}, {MAX_BOARD_SIZE}]"
        )
    return n


def _resolve_board_size() -> int:
    # 1. Explicit --board-size flag on the command line. Resolved here (not in
    #    each entry point's argparse) because module-level constants across the
    #    package are derived at import time, before any main() runs.
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--board-size" and i + 1 < len(argv):
            return _parse(argv[i + 1], "--board-size")
        if arg.startswith("--board-size="):
            return _parse(arg.split("=", 1)[1], "--board-size")
    # 2. Environment variable.
    raw = os.environ.get(ENV_VAR)
    if raw is not None and raw.strip():
        return _parse(raw, ENV_VAR)
    # 3. Default.
    return DEFAULT_BOARD_SIZE


BOARD_SIZE: int = _resolve_board_size()
N_ACTIONS: int = BOARD_SIZE * BOARD_SIZE

# Propagate to children: spawned workers resolve the same size from the env.
os.environ[ENV_VAR] = str(BOARD_SIZE)


def add_board_size_arg(parser) -> None:
    """Register the documented --board-size flag on an entry point's parser.

    The flag's real work happened at import time (argv sniff above); this
    argparse entry exists so --help documents it and argparse accepts it.
    Pair with ``require_board_size(args.board_size)`` in main().
    """
    parser.add_argument(
        "--board-size",
        type=int,
        default=BOARD_SIZE,
        help=f"Board side length (active: {BOARD_SIZE}). Resolved once at "
             "process start: --board-size flag > GOMOKU_BOARD_SIZE env var > 9. "
             "Native extensions exist for 9 and 15; other sizes use the "
             "pure-Python fallback.",
    )


def require_board_size(n: int) -> None:
    """Fail fast if a parsed --board-size disagrees with the active config.

    Can only trigger when main() is driven with a synthetic argv that
    gomoku.board_config never saw (e.g. programmatic parse_args calls after
    the package was already imported at a different size).
    """
    if int(n) != BOARD_SIZE:
        raise SystemExit(
            f"--board-size {n} conflicts with the active board size "
            f"{BOARD_SIZE}, which was fixed at import time. Set "
            f"GOMOKU_BOARD_SIZE={n} (or pass --board-size on the actual "
            "command line) and start a new process."
        )
