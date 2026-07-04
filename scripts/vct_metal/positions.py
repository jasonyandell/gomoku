"""Real-position loader for the VCT spike — random boards from Rapfi self-play.

This feeds *every* test of the VCT (Victory-by-Continuous-Threats) solver, so it
is deliberately conservative and validated against the replayed ground truth.

Source: ``~/data/games_raphi/*.jsonl.gz`` — 450 gzipped-JSONL shards of Rapfi
self-play (~45k games). Each line is one full game::

    {"moves": [97, 124, 107, ...], "winner": 1, "bs": 15}

  * ``moves``  flat *row-major* board indices (``idx = row * bs + col``), the
               exact encoding ``state_ops`` uses (``divmod(action, BOARD_SIZE)``).
               Black moves first; the list strictly alternates black, white, ...
  * ``bs``     board size (15 for every shard).
  * ``winner`` 0 = black (first player) wins, 1 = white wins, -1 = no result
               (game hit the 120-move cap; no five on the board). Empirically the
               winner is always exactly the last mover and holds the only five
               (verified across 4000 games, 0 mismatches), so we re-derive
               terminality from the replayed board rather than trusting this field.

Conventions match ``detect_ref`` / ``gomoku.vcf``:
  * ``own`` is plane 0 (the side to move = the VCT attacker), ``opp`` is plane 1.
  * a *five* is ``>= WIN_LEN`` in a row (freestyle: overlines win) — exactly what
    ``state_ops.has_five_in_a_row`` reports.

At ply ``p`` (after ``p`` moves) the side to move is black iff ``p`` is even, so
the side-to-move always holds ``p // 2`` stones and the opponent ``ceil(p / 2)``.

Nothing here is imported by the rest of the repo — it is part of the hermetic
VCT spike.
"""

from __future__ import annotations

import functools
import gzip
import json
import os
from pathlib import Path

import numpy as np

from gomoku import state_ops

N = state_ops.BOARD_SIZE

# Override with GOMOKU_RAPFI_GAMES_DIR; defaults to where the shards live.
DEFAULT_GAMES_DIR = Path(
    os.environ.get("GOMOKU_RAPFI_GAMES_DIR", "~/data/games_raphi")
).expanduser()


# --------------------------------------------------------------------------- #
# Loading (cached): parse every shard once into flat-index move arrays.
# --------------------------------------------------------------------------- #
def _resolve_dir(data_dir: str | os.PathLike | None) -> Path:
    return DEFAULT_GAMES_DIR if data_dir is None else Path(data_dir).expanduser()


@functools.lru_cache(maxsize=8)
def _load_games(data_dir_str: str) -> tuple[np.ndarray, ...]:
    """Return all games (whose board size matches ``N``) as int16 move arrays.

    Cached by directory: the first call parses ~45k games (~0.1 s); later calls
    are free. Games recorded at a different board size than the active
    ``state_ops.BOARD_SIZE`` are skipped (their flat indices would be off-board).
    """
    data_dir = Path(data_dir_str)
    shards = sorted(data_dir.glob("*.jsonl.gz"))
    if not shards:
        raise FileNotFoundError(
            f"no '*.jsonl.gz' game shards under {data_dir} "
            "(set GOMOKU_RAPFI_GAMES_DIR or pass data_dir=...)"
        )

    games: list[np.ndarray] = []
    skipped_size = 0
    for shard in shards:
        with gzip.open(shard, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if int(rec["bs"]) != N:
                    skipped_size += 1
                    continue
                mv = np.asarray(rec["moves"], dtype=np.int16)
                if mv.size and (mv.min() < 0 or mv.max() >= N * N):
                    raise ValueError(
                        f"{shard.name}: move index out of range for {N}x{N} board"
                    )
                games.append(mv)

    if not games:
        raise ValueError(
            f"no {N}x{N} games found under {data_dir} "
            f"(skipped {skipped_size} games of a different board size). "
            "These Rapfi shards are 15x15 — run with GOMOKU_BOARD_SIZE=15."
        )
    return tuple(games)


# --------------------------------------------------------------------------- #
# Replay: reconstruct the (own, opp) planes after the first ``p`` moves.
# --------------------------------------------------------------------------- #
def _planes_at_ply(game: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Replay ``game`` to ply ``p`` -> (own, opp) as (N, N) bool planes.

    Black plays the even move-indices, white the odd ones. ``own`` is the side to
    move at ply ``p`` (black iff ``p`` is even), ``opp`` the other side.
    """
    black = game[0:p:2]  # moves 0, 2, 4, ...
    white = game[1:p:2]  # moves 1, 3, 5, ...
    own_idx, opp_idx = (black, white) if p % 2 == 0 else (white, black)

    own = np.zeros(N * N, dtype=bool)
    opp = np.zeros(N * N, dtype=bool)
    own[own_idx] = True
    opp[opp_idx] = True
    return own.reshape(N, N), opp.reshape(N, N)


def _is_terminal(own: np.ndarray, opp: np.ndarray) -> bool:
    return state_ops.has_five_in_a_row(own) or state_ops.has_five_in_a_row(opp)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_random_positions(
    n: int,
    seed: int = 0,
    min_ply: int = 6,
    max_ply: int | None = None,
    data_dir: str | os.PathLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``n`` random *real* board positions from the Rapfi game shards.

    Each sample is an independently-drawn game replayed to a random ply in
    ``[min_ply, min(max_ply, len(game) - 1)]``. Terminal positions (either side
    already has a five) are skipped by re-sampling — capping at ``len - 1``
    already excludes the single winning position, so this is belt-and-braces.

    Returns ``(own, opp)``, each ``(n, N, N)`` bool with ``N = BOARD_SIZE``:
    ``own`` = the side-to-move's stones (the VCT attacker, plane 0), ``opp`` =
    the opponent (plane 1).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if min_ply < 0:
        raise ValueError(f"min_ply must be >= 0, got {min_ply}")
    if max_ply is not None and max_ply < min_ply:
        raise ValueError(f"max_ply ({max_ply}) < min_ply ({min_ply})")

    games = _load_games(str(_resolve_dir(data_dir)))
    lengths = np.fromiter((len(g) for g in games), dtype=np.int64, count=len(games))

    # Per game the highest non-terminal ply we may stop at is len - 1, further
    # clamped by max_ply. A game is eligible iff that cap reaches min_ply.
    hi_per = lengths - 1
    if max_ply is not None:
        hi_per = np.minimum(hi_per, max_ply)
    eligible = np.nonzero(hi_per >= min_ply)[0]
    if eligible.size == 0:
        raise ValueError(
            f"no game is long enough for min_ply={min_ply}"
            + (f", max_ply={max_ply}" if max_ply is not None else "")
            + f" (longest game has {int(lengths.max())} moves)"
        )

    rng = np.random.default_rng(seed)
    own_out = np.zeros((n, N, N), dtype=bool)
    opp_out = np.zeros((n, N, N), dtype=bool)

    filled = 0
    attempts = 0
    max_attempts = 100 * n + 1000  # rejection essentially never fires
    while filled < n:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"only filled {filled}/{n} positions after {attempts} draws; "
                "the game pool may be too small or fully terminal"
            )
        attempts += 1
        gi = int(eligible[rng.integers(eligible.size)])
        game = games[gi]
        hi = min(len(game) - 1, max_ply) if max_ply is not None else len(game) - 1
        p = int(rng.integers(min_ply, hi + 1))  # inclusive of hi
        own, opp = _planes_at_ply(game, p)
        if _is_terminal(own, opp):
            continue
        own_out[filled] = own
        opp_out[filled] = opp
        filled += 1

    return own_out, opp_out


def load_position_stack(n: int, seed: int = 0, **kw) -> np.ndarray:
    """Convenience wrapper: ``(n, 2, N, N)`` bool = ``stack([own, opp], axis=1)``."""
    own, opp = load_random_positions(n, seed=seed, **kw)
    return np.stack([own, opp], axis=1)


# --------------------------------------------------------------------------- #
# Self-test / validation report:  python -m scripts.vct_metal.positions
# --------------------------------------------------------------------------- #
def _grid(own: np.ndarray, opp: np.ndarray) -> str:
    """Tiny text board: X = own (side to move), O = opp, . = empty."""
    rows = []
    for r in range(N):
        cells = ["X" if own[r, c] else "O" if opp[r, c] else "." for c in range(N)]
        rows.append(" ".join(cells))
    return "\n".join(rows)


def _validate() -> int:
    import collections

    n = 300
    min_ply, max_ply = 6, 80
    print(f"=== positions.py validation  (BOARD_SIZE={N}, "
          f"native={state_ops.USING_NATIVE}) ===")
    games = _load_games(str(_resolve_dir(None)))
    print(f"loaded {len(games)} games; "
          f"lengths min/median/max = "
          f"{min(len(g) for g in games)}/"
          f"{int(np.median([len(g) for g in games]))}/"
          f"{max(len(g) for g in games)}")

    own, opp = load_random_positions(n, seed=0, min_ply=min_ply, max_ply=max_ply)
    own_cnt = own.reshape(n, -1).sum(1)
    opp_cnt = opp.reshape(n, -1).sum(1)
    plies = own_cnt + opp_cnt

    checks: list[tuple[str, bool, str]] = []

    overlap = (own & opp).any()
    checks.append(("no own/opp overlap", not overlap,
                   f"overlapping samples = {int((own & opp).reshape(n, -1).any(1).sum())}"))

    # side to move holds floor(p/2); opponent holds ceil(p/2).
    counts_ok = bool(np.all(own_cnt == plies // 2) and np.all(opp_cnt == (plies + 1) // 2))
    checks.append(("stone counts match ply (own=floor(p/2), opp=ceil(p/2))",
                   counts_ok,
                   f"diff opp-own in {sorted(set((opp_cnt - own_cnt).tolist()))}"))

    term = sum(_is_terminal(own[i], opp[i]) for i in range(n))
    checks.append(("no terminal positions", term == 0, f"terminal samples = {term}"))

    spread_ok = bool(plies.min() >= min_ply and plies.max() <= max_ply
                     and len(set(plies.tolist())) >= 10)
    checks.append((f"ply spread within [{min_ply},{max_ply}], well spread",
                   spread_ok,
                   f"ply min/max = {int(plies.min())}/{int(plies.max())}, "
                   f"distinct plies = {len(set(plies.tolist()))}"))

    # determinism: same seed identical, different seed differs.
    own2, opp2 = load_random_positions(n, seed=0, min_ply=min_ply, max_ply=max_ply)
    own3, _ = load_random_positions(n, seed=1, min_ply=min_ply, max_ply=max_ply)
    det_ok = bool(np.array_equal(own, own2) and np.array_equal(opp, opp2)
                  and not np.array_equal(own, own3))
    checks.append(("deterministic by seed (same==, diff!=)", det_ok, ""))

    # stack wrapper agrees.
    stack = load_position_stack(n, seed=0, min_ply=min_ply, max_ply=max_ply)
    stack_ok = bool(stack.shape == (n, 2, N, N)
                    and np.array_equal(stack[:, 0], own)
                    and np.array_equal(stack[:, 1], opp))
    checks.append(("load_position_stack shape/content", stack_ok, str(stack.shape)))

    print("\nply histogram (binned by 5):")
    binned = collections.Counter((int(p) // 5) * 5 for p in plies)
    for lo in sorted(binned):
        bar = "#" * binned[lo]
        print(f"  {lo:3d}-{lo + 4:<3d} | {binned[lo]:3d} {bar}")

    print("\nexample boards (X = side to move / attacker, O = opponent):")
    for i in (0, 1):
        print(f"\n  sample {i}: ply={int(plies[i])}  "
              f"own={int(own_cnt[i])}  opp={int(opp_cnt[i])}")
        print("    " + _grid(own[i], opp[i]).replace("\n", "\n    "))

    print("\n--- results ---")
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))
    print(f"\n{'PASS' if all_ok else 'FAIL'}: "
          f"{sum(ok for _, ok, _ in checks)}/{len(checks)} checks")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_validate())
