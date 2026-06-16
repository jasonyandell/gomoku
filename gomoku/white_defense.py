"""White-defense evaluation suite — core library (GH #45).

The defense arc needs a gate signal that MOVES. On the ~50-elo 128x10 bigbuf
plateau the overall H2H PROMOTE/REVERT verdict is statistically dead, but
``white_loss_rate`` still moves. The old ``white_loss`` (sliding_gate's
``candidate_white_loss_tally``) cannot gate: it tallies only the candidate's
white half of anchor H2H pairs, vs the ~50-elo frozen peak, on random openings,
as a bare ratio with NO CI. This module replaces that noisy footnote with a
FIXED, reusable defend-as-white suite: a checked-in fixture of white-to-move
positions facing a LIVE black threat, scored by having the net DEFEND as white
from each position against a fixed, reproducible attacker, reported as
``white_loss_rate`` with a Wilson CI.

Three load-bearing pieces live here:

1. ``planes_to_state`` / ``state_to_planes`` — the planes<->GameState
   reconstructor. The fixture stores the canonical input planes + side-to-move +
   ply; the per-game H2H primitives need a real ``GameState``. The round-trip
   (state -> planes -> state) is LOSSLESS for everything the H2H loop reads
   (board, side-to-move, legal moves, and the HISTORY_PLY-1 most-recent history
   frames). A silent error here makes the whole fixture garbage, so the
   reconstructor has a dedicated round-trip test (tests/test_white_defense.py).

2. ``black_threat_present`` — the fixture filter: is there a LIVE black threat
   that white must answer? Reuses ``vcf.has_four_threat`` (a four / immediate
   five) plus a window-count open-three detector. Only such positions are
   genuinely defense-critical.

3. ``defend_one_position`` / ``score_white_defense`` — play the net as WHITE
   from a fixture position against a fixed CPU attacker (default
   ``lookahead_player``), tally white losses (a draw is a SUCCESSFUL defense),
   and report ``white_loss_rate`` + ``binomial_ci``.

``white_defense_tally`` exposes the SAME ``(rate, defense_score, n)`` return
shape as ``sliding_gate.candidate_white_loss_tally`` so it drops into
``run_gate``'s ``white_loss_fn`` seam with ZERO changes to ``decide_verdict``.

Everything board-shaped here derives from ``gomoku.game`` constants, so set
``GOMOKU_BOARD_SIZE=15`` (process-level) before import for the 15x15 era.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from gomoku.game import (
    BOARD_SIZE,
    HISTORY_PLY,
    N_ACTIONS,
    N_INPUT_PLANES,
    GameState,
)

# A "picker": (state, rng) -> action. Mirrors gomoku.eval.Picker / baselines.Player.
Picker = Callable[[GameState, np.random.Generator], int]

FIXTURE_SCHEMA = "white_defense_fixture.v1"


# ---------------------------------------------------------------------------
# 1. planes <-> GameState reconstructor (the correctness keystone).
# ---------------------------------------------------------------------------
def state_to_planes(state: GameState) -> np.ndarray:
    """Canonical (N_INPUT_PLANES, N, N) float32 planes — thin alias of
    ``GameState.to_planes`` so the round-trip reads symmetrically here."""
    return state.to_planes()


def planes_to_state(planes: np.ndarray, ply: int) -> GameState:
    """Reconstruct a ``GameState`` from canonical input planes + ply count.

    The plane layout (see ``GameState.to_planes``) is, with H = HISTORY_PLY:
        0           : side-to-move's stones at t
        1..H-1      : side-to-move's stones at t-1 .. t-(H-1)
        H           : opponent's stones at t
        H+1..2H-1   : opponent's stones at t-1 .. t-(H-1)
        2H          : constant 1.0
    For history step k (1..H-1) ``to_planes`` writes
        out[k]   = past_board[0] if k even else past_board[1]
        out[H+k] = past_board[1] if k even else past_board[0]
    where ``past_board`` is the canonical board ``history[k-1]``. We invert that
    map exactly, so the reconstructed ``history`` is byte-identical to the
    original for the H-1 frames the planes carry.

    ``ply`` becomes ``move_count`` (its parity must match the side-to-move the
    planes encode: side-to-move is black iff ply is even). We do NOT fabricate
    history frames older than H-1 (the planes don't carry them); that has no
    effect on the H2H loop, which only reads the current board (legal moves +
    terminal) and the net input planes (board + the H-1 frames stored here).
    """
    planes = np.asarray(planes)
    if planes.shape != (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE):
        raise ValueError(
            f"planes shape {planes.shape} != "
            f"({N_INPUT_PLANES}, {BOARD_SIZE}, {BOARD_SIZE}) — wrong board size? "
            f"set GOMOKU_BOARD_SIZE before import"
        )
    H = HISTORY_PLY
    bool_planes = planes > 0.5
    board = np.stack([bool_planes[0], bool_planes[H]]).copy()  # (2, N, N)

    # Reconstruct the history tuple, most-recent first, inverting the parity map.
    history: list[np.ndarray] = []
    for k in range(1, H):
        me_k = bool_planes[k]
        opp_k = bool_planes[H + k]
        if not me_k.any() and not opp_k.any():
            # No real frame stored at this depth (early game / no-history mine).
            break
        if k % 2 == 0:
            past = np.stack([me_k, opp_k]).copy()
        else:
            past = np.stack([opp_k, me_k]).copy()
        history.append(past)

    return GameState(board=board, move_count=int(ply), history=tuple(history))


def side_to_move(planes_or_state) -> int:
    """0 = black to move, 1 = white to move. From a GameState use its
    move_count parity; from planes the side is the same as the canonical mover
    and must be supplied via ply, so this helper takes a GameState only."""
    if isinstance(planes_or_state, GameState):
        return planes_or_state.move_count % 2
    raise TypeError("side_to_move expects a GameState (use ply%2 for raw planes)")


# ---------------------------------------------------------------------------
# 2. Live-black-threat filter (which white-to-move positions are defensive).
# ---------------------------------------------------------------------------
def _open_three_present(attacker: np.ndarray, defender: np.ndarray) -> bool:
    """True if the attacker has a 'live three': some length-5 window holding
    exactly 3 attacker stones and 0 defender stones. Such a three can become an
    open four next move and forces a defensive response. ``attacker``/``defender``
    are (N, N) bool planes."""
    from gomoku.baselines import _WINDOWS  # precomputed (n_windows, 5) line idx

    a = attacker.reshape(-1)
    d = defender.reshape(-1)
    a_cnt = a[_WINDOWS].sum(axis=1)
    d_cnt = d[_WINDOWS].sum(axis=1)
    return bool(((a_cnt == 3) & (d_cnt == 0)).any())


def black_threat_present(state: GameState, *, require_four: bool = False) -> bool:
    """Is there a LIVE black threat the white side-to-move must answer?

    ``state`` MUST be white-to-move (move_count odd): plane 0 is white's stones,
    plane 1 (= input plane H) is black's. We re-orient to black-as-attacker and
    test for a four threat (``vcf.has_four_threat``: an immediate five or a
    four-making move) and, unless ``require_four``, a live open three.

    Returns False on a black-to-move state (defensive filter is white-only).
    """
    if state.move_count % 2 == 0:
        return False  # black to move — not a white-defense position
    from gomoku.vcf import has_four_threat

    black = state.board[1]
    white = state.board[0]
    attacker_board = np.stack([black, white]).copy()  # plane 0 = black attacker
    if has_four_threat(attacker_board):
        return True
    if require_four:
        return False
    return _open_three_present(black, white)


# ---------------------------------------------------------------------------
# 3. Fixture I/O — a small, versioned, checked-in position set.
#
# Stored as JSON of MOVE SEQUENCES (the action played at each ply, in order),
# not raw planes: replaying a sequence reconstructs the board AND the
# HISTORY_PLY frames AND the ply EXACTLY, so it's a faithful, tiny, diffable,
# human-auditable artifact that side-steps the repo's `*.pt` gitignore. The
# planes a position exposes are produced by `state.to_planes()` after replay,
# identical to mining time.
# ---------------------------------------------------------------------------
def state_from_moves(moves: Sequence[int]) -> GameState:
    """Replay a move sequence from the empty board to a GameState (board +
    history + ply all reconstructed exactly)."""
    state = GameState.initial()
    for a in moves:
        state = state.apply(int(a))
    return state


@dataclass
class WhiteDefenseFixture:
    """A loaded white-defense fixture: white-to-move positions + provenance.

    Each entry in ``positions`` is
    ``{"moves": list[int], "ply": int, "provenance": str}``. ``planes`` and a
    ``GameState`` are produced on demand by replaying ``moves`` (see ``states``).
    """

    board_size: int
    positions: list[dict]
    meta: dict

    def __len__(self) -> int:
        return len(self.positions)

    def states(self) -> list[GameState]:
        return [state_from_moves(p["moves"]) for p in self.positions]


def save_fixture(
    path: str | Path,
    positions: Sequence[dict],
    *,
    meta: Optional[dict] = None,
) -> None:
    """Write a fixture as checked-in JSON. Each position dict needs ``moves``
    (the list of actions to replay), ``ply`` (int, == len(moves)), ``provenance``
    (str). Validated by re-loading + threat-checking in the CLI."""
    if not positions:
        raise ValueError("refusing to write an empty fixture")
    rows = []
    for p in positions:
        moves = [int(m) for m in p["moves"]]
        rows.append(
            {
                "moves": moves,
                "ply": int(p.get("ply", len(moves))),
                "provenance": str(p.get("provenance", "unknown")),
            }
        )
    archive = {
        "schema": FIXTURE_SCHEMA,
        "board_size": int(BOARD_SIZE),
        "n_input_planes": int(N_INPUT_PLANES),
        "n_actions": int(N_ACTIONS),
        "meta": dict(meta or {}),
        "positions": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(archive, indent=1, sort_keys=False) + "\n")


def load_fixture(path: str | Path) -> WhiteDefenseFixture:
    """Load a fixture. Validates board size matches the current process."""
    archive = json.loads(Path(path).read_text())
    if archive.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"{path}: not a {FIXTURE_SCHEMA} fixture")
    bs = int(archive["board_size"])
    if bs != BOARD_SIZE:
        raise ValueError(
            f"{path}: fixture board_size={bs} but process BOARD_SIZE={BOARD_SIZE} "
            f"(set GOMOKU_BOARD_SIZE={bs})"
        )
    positions = [
        {
            "moves": [int(m) for m in row["moves"]],
            "ply": int(row.get("ply", len(row["moves"]))),
            "provenance": str(row.get("provenance", "unknown")),
        }
        for row in archive["positions"]
    ]
    return WhiteDefenseFixture(board_size=bs, positions=positions, meta=dict(archive.get("meta", {})))


# ---------------------------------------------------------------------------
# 4. Defend-from-position game + scoring.
# ---------------------------------------------------------------------------
def defend_one_position(
    defender_picker: Picker,
    attacker_picker: Picker,
    start_state: GameState,
    *,
    rng: np.random.Generator,
) -> str:
    """Play ONE game from ``start_state`` to terminal, with the DEFENDER (the
    net, playing WHITE) and the ATTACKER (a fixed baseline, playing BLACK)
    moving by side-to-move. Returns "win"/"loss"/"draw" from the DEFENDER's
    (white's) perspective. A draw is a SUCCESSFUL defense.

    The injected ``start_state`` is white-to-move (move_count odd). Side-to-move
    is derived from ``move_count`` parity exactly as the H2H primitives do, so no
    color is hard-coded; the white defender plays whenever ply is odd.
    """
    state = start_state
    ply = state.move_count
    defender_side = 1  # white
    winner_side: Optional[int] = None
    # Guard against a degenerate already-terminal injected position.
    done, _ = state.is_terminal()
    if done:
        return "draw"
    while True:
        cur_side = ply % 2
        picker = defender_picker if cur_side == defender_side else attacker_picker
        action = int(picker(state, rng))
        side_just_moved = ply % 2
        state = state.apply(action)
        done, v = state.is_terminal()
        if done:
            if v == -1.0:
                winner_side = side_just_moved
            break
        ply += 1
    if winner_side is None:
        return "draw"
    return "win" if winner_side == defender_side else "loss"


@dataclass
class WhiteDefenseResult:
    n: int                       # total defend games played (fixture x seeds)
    white_losses: int            # games lost as white (failed defenses)
    white_loss_rate: float       # white_losses / n
    ci_lo: float                 # Wilson CI on the LOSS rate (lo)
    ci_hi: float                 # Wilson CI on the LOSS rate (hi)
    defense_score: float         # n - white_losses (successful defenses; draws count)
    per_provenance: dict         # provenance -> (losses, n)

    @property
    def ci_half(self) -> float:
        return 0.5 * (self.ci_hi - self.ci_lo)


def score_white_defense(
    defender_picker: Picker,
    attacker_picker: Picker,
    fixture: WhiteDefenseFixture,
    *,
    n_seeds: int = 1,
    base_seed: int = 0,
    confidence: float = 0.95,
    progress: Optional[Callable[[int, int], None]] = None,
) -> WhiteDefenseResult:
    """Score a defender over (fixture positions x n_seeds) and report
    ``white_loss_rate`` + a Wilson CI (reusing ``delta_e_harness.binomial_ci``).

    Determinism: each (position i, seed s) game uses RNG seed
    ``base_seed + s*100003 + i`` (the picker's tie-breaks + any MCTS noise read
    from it), so the metric is reproducible across runs.
    """
    from scripts.delta_e_harness import binomial_ci  # CI machinery (reused verbatim)

    states = fixture.states()
    prov = [p["provenance"] for p in fixture.positions]
    n = 0
    white_losses = 0
    per_prov: dict[str, list[int]] = {}
    total = len(states) * max(1, n_seeds)
    for s in range(max(1, n_seeds)):
        for i, st in enumerate(states):
            rng = np.random.default_rng(base_seed + s * 100003 + i)
            outcome = defend_one_position(defender_picker, attacker_picker, st, rng=rng)
            n += 1
            lost = 1 if outcome == "loss" else 0
            white_losses += lost
            tag = prov[i]
            bucket = per_prov.setdefault(tag, [0, 0])
            bucket[0] += lost
            bucket[1] += 1
            if progress is not None:
                progress(n, total)
    rate = white_losses / n if n else 0.0
    ci_lo, ci_hi = binomial_ci(float(white_losses), n, confidence=confidence)
    return WhiteDefenseResult(
        n=n,
        white_losses=white_losses,
        white_loss_rate=rate,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        defense_score=float(n - white_losses),
        per_provenance={k: tuple(v) for k, v in per_prov.items()},
    )


# ---------------------------------------------------------------------------
# 5. Gate-primitive seam — drop-in for run_gate's white_loss_fn.
# ---------------------------------------------------------------------------
def white_defense_tally(
    candidate: str,
    peak: str,  # noqa: ARG001 — unused; the suite uses a FIXED attacker, not the peer
    *,
    n_games: int,  # noqa: ARG001 — n is fixture x n_seeds, not a paired-H2H count
    sims: int,
    c_puct: float,
    seed: int,
    device: str,
    opening_plies: int,  # noqa: ARG001 — fixtures replace random openings
    fixture_path: str | Path,
    attacker: str = "lookahead:depth=2",
    n_seeds: int = 1,
) -> tuple[Optional[float], float, int]:
    """Fixture-based white-defense tally with the EXACT
    ``(white_loss_rate, defense_score, white_n)`` shape of
    ``sliding_gate.candidate_white_loss_tally`` — drops into ``run_gate``'s
    ``white_loss_fn`` seam with NO change to ``decide_verdict``.

    Unlike the old tally (candidate-plays-white vs the ~50-elo peer on random
    openings) this scores the candidate DEFENDING from the FIXED fixture against
    a FIXED reliable attacker. ``peak``/``n_games``/``opening_plies`` are accepted
    for signature compatibility and intentionally ignored (the fixture + fixed
    attacker replace them). The candidate is the only checkpoint that gets loaded.

    Bind ``fixture_path`` (and optionally ``attacker``/``n_seeds``) with
    ``functools.partial`` before passing as ``white_loss_fn``.
    """
    defender = build_ckpt_defender(candidate, sims=sims, c_puct=c_puct, device=device)
    attacker_picker = build_attacker(attacker)
    fixture = load_fixture(fixture_path)
    res = score_white_defense(
        defender, attacker_picker, fixture, n_seeds=n_seeds, base_seed=seed,
    )
    if res.n == 0:
        return (None, 0.0, 0)
    return (res.white_loss_rate, res.defense_score, res.n)


# ---------------------------------------------------------------------------
# Picker builders (kept here so the CLI + gate seam share one construction path).
# ---------------------------------------------------------------------------
def build_ckpt_defender(checkpoint: str, *, sims: int, c_puct: float, device: str) -> Picker:
    """The net-as-white defender (the only MPS consumer). Reuses the harness's
    load/fuse/evaluator + mcts_picker idiom so the defender is byte-identical to
    the H2H executor's checkpoint picker."""
    from scripts.delta_e_harness import _build_ckpt_picker

    return _build_ckpt_picker(checkpoint, sims=sims, c_puct=c_puct, device=device)


def build_attacker(spec: str) -> Picker:
    """Build a fixed, reproducible CPU attacker from a spec string. NO wine —
    only the project's own baselines (#35 panel-wine lesson):
        random | heuristic | defensive | lookahead | lookahead:depth=K
    """
    from gomoku.baselines import (
        defensive_player,
        heuristic_player,
        lookahead_player,
        random_player,
    )

    spec = spec.strip()
    if spec == "random":
        return random_player
    if spec == "heuristic":
        return heuristic_player
    if spec == "defensive":
        return defensive_player
    if spec == "lookahead":
        return lookahead_player(depth=4)
    if spec.startswith("lookahead:"):
        kv = spec.split(":", 1)[1]
        if kv.startswith("depth="):
            return lookahead_player(depth=int(kv.split("=", 1)[1]))
    raise ValueError(
        f"unknown attacker spec {spec!r}; use one of "
        f"random|heuristic|defensive|lookahead|lookahead:depth=K"
    )


def fixture_summary_json(fixture: WhiteDefenseFixture) -> str:
    """One-line JSON summary of a fixture (counts by provenance)."""
    from collections import Counter

    counts = Counter(p["provenance"] for p in fixture.positions)
    return json.dumps(
        {
            "board_size": fixture.board_size,
            "n_positions": len(fixture),
            "by_provenance": dict(sorted(counts.items())),
            "meta": fixture.meta,
        },
        sort_keys=True,
    )
