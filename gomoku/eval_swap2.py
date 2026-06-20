"""Swap2 eval-vs-engine harness — strength under the REAL Gomocup protocol.

Why this exists
===============
`eval_vs_rapfi.py` scores a checkpoint from the empty board, where 15x15
freestyle is a proven *first-player win* and "white can't win" is a forced
constraint. That makes the empty-board yardstick a measure of how well we exploit
(or suffer) an unbalanced opening, not of negotiated-game strength. Swap2 removes
the forced lost role: BOTH sides negotiate the opening, so the resulting position
is balanced and the eval measures strength under the protocol the engine actually
ladders on.

This is the gate. Per game, BOTH the opener and the responder of the swap2
negotiation are real players — our net (via :func:`gomoku.swap2_search.agent_act`,
one OpeningState node at a time) and the external engine (via the SWAP2BOARD
protocol methods on :class:`gomoku.external_engine.ExternalEnginePlayer`). After
the negotiation reaches DONE, normal gomoku is played from the handed-off
position; our net moves via the existing eval MCTS picker, the engine via its
normal `__call__` path.

EVAL-ONLY. The engine never touches self-play training (same rule as the Rapfi
yardstick). This module only *calls* swap2 / swap2_search / external_engine /
model; it owns none of them.

The protocol <-> OpeningState mapping (the load-bearing risk)
============================================================
The engine tracks its own board in a separate process and speaks coords as
(x=col, y=row). We drive a single :class:`gomoku.swap2.OpeningState` from
:func:`gomoku.swap2.initial_opening`, mapping OPENER/RESPONDER to {our_net,
engine} by the per-game role assignment. At each node:

* OUR node    -> ``agent_act(oracle, state, greedy=True)`` -> ``state.apply(...)``
  (one node at a time; a placement square or a choice slot).
* ENGINE node -> resolve the WHOLE negotiation boundary with ONE protocol call,
  then apply its result to the OpeningState. The exact per-boundary mappings are
  in :func:`_engine_open`, :func:`_engine_respond`, :func:`_engine_pick`.

The one subtle bug this guards is **move-parity coloring** in the engine's PLACE2
(TWO_COORDS) reply: move 4 is WHITE and move 5 is BLACK (1-indexed, black on odd
plies), regardless of the order the engine lists the two coords. OpeningState's
PLACE2 steps are black-first then white, so we apply the move-5 coord (black) at
the black step and the move-4 coord (white) at the white step. See
:func:`_engine_respond`.

Color/role tracking for attribution
===================================
After the opening is DONE, ``state.opener_color`` is the color the OPENER plays
the whole game. Our net is the OPENER or the RESPONDER (assigned per game), so
our net's color is ``opener_color`` when we opened, else the other color. The
winner of the finished game is a color; we win iff that color == our net's color.
We also break the tally down by our net's final color and by our role, and report
the distribution of ``opener_color`` across the negotiated games.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from gomoku.board_config import BOARD_SIZE
from gomoku.external_engine import (
    ExternalEnginePlayer,
    Swap2Option,
    Swap2Reply,
)
from gomoku.game import GameState
from gomoku.swap2 import (
    Actor,
    Color,
    OpeningState,
    PICK_BLACK,
    PICK_WHITE,
    Phase,
    RESP_PLACE2,
    RESP_STAY,
    RESP_SWAP,
    initial_opening,
)
from gomoku.swap2_search import OracleFn, agent_act

# A safety cap on normal-play plies so a pathological non-terminating game (should
# be impossible on a finite board, but defends against a bad engine reply loop)
# can't hang the harness.
_MAX_NORMAL_PLIES = BOARD_SIZE * BOARD_SIZE + 4


def _sq(x: int, y: int) -> int:
    """A SWAP2 coord (x=col, y=row) -> an OpeningState placement square."""
    return y * BOARD_SIZE + x


# ---------------------------------------------------------------------------
# Engine-boundary resolvers. Each takes the current OpeningState (at the node
# the engine is to act) and the engine, performs the ONE protocol call for that
# boundary, applies the reply to the OpeningState, and returns the advanced
# state. The reveal of an engine first-normal-move (the ONE_COORD pick) is
# returned alongside so the driver can apply it without re-querying the engine.
# ---------------------------------------------------------------------------


def _engine_open(state: OpeningState, engine: ExternalEnginePlayer) -> OpeningState:
    """Engine is OPENER at the start: call ``swap2_open()`` and place its 3 stones.

    The engine returns 3 coords in MOVE ORDER; by Gomocup alternation their colors
    are black, white, black. OpeningState's three opener steps are exactly
    black, white, black, so we apply the coords in returned order.
    """
    reply = engine.swap2_open()
    if reply.option is not Swap2Option.OPEN_THREE or len(reply.coords) != 3:
        raise ValueError(f"engine swap2_open returned a non-opening reply: {reply.option}")
    for (x, y) in reply.coords:
        state = state.apply(_sq(x, y))
    return state


def _colored_stones_xy(state: OpeningState) -> list[tuple[int, int, int]]:
    """All stones on ``state`` as (x, y, color) triples, color in {1, 2}.

    Color is read DIRECTLY off the absolute (black, white) planes — black -> 1,
    white -> 2 — never assumed from placement parity. The SWAP2BOARD block needs
    an explicit color per stone (`x,y,color`); Rapfi rejects a bare `x,y` line
    (see `external_engine` module docstring). Blacks are listed before whites in
    board (row-major) order; with explicit colors the listing order is irrelevant
    to the engine's coloring, it only needs the set of colored squares.
    """
    blacks = [(int(c), int(r), 1) for r, c in np.argwhere(state.black)]  # (x=col, y=row, BLACK)
    whites = [(int(c), int(r), 2) for r, c in np.argwhere(state.white)]  # (..., WHITE)
    return blacks + whites


def _our_three_stones_xy(state: OpeningState) -> list[tuple[int, int, int]]:
    """The 3 opening stones (2 black + 1 white) as (x, y, color) triples.

    Thin wrapper over :func:`_colored_stones_xy`; the responder query just needs
    the three colored opening squares.
    """
    return _colored_stones_xy(state)


def _engine_respond(
    state: OpeningState, engine: ExternalEnginePlayer
) -> OpeningState:
    """Engine is RESPONDER at the 3-stone choice node: call ``swap2_respond``.

    Branches (see module docstring for the move-parity coloring rationale):
      * SWAP        -> ``apply(RESP_SWAP)``; engine took black, opener (us) is
                       white and moves next.
      * ONE_COORD   -> engine STAYED white and played move 4 (white) at (x,y):
                       ``apply(RESP_STAY).apply(sq(x,y))``.
      * TWO_COORDS  -> engine PLACE2 (moves 4 & 5). move 4 = WHITE, move 5 = BLACK
                       by GLOBAL ply parity, NOT reply order. OpeningState PLACE2
                       steps are black-first then white, so apply the move-5
                       (black) coord first, then the move-4 (white) coord.
    """
    stones = _our_three_stones_xy(state)
    reply = engine.swap2_respond(stones)
    if reply.option is Swap2Option.SWAP:
        return state.apply(RESP_SWAP)
    if reply.option is Swap2Option.ONE_COORD:
        (x, y) = reply.coords[0]
        return state.apply(RESP_STAY).apply(_sq(x, y))
    if reply.option is Swap2Option.TWO_COORDS:
        move4_white, move5_black = reply.coords[0], reply.coords[1]
        state = state.apply(RESP_PLACE2)  # -> P2_P2_B (black step) next
        # Black step first (move 5 is black), then white step (move 4 is white).
        state = state.apply(_sq(*move5_black))
        state = state.apply(_sq(*move4_white))
        return state
    raise ValueError(f"engine swap2_respond returned an unexpected reply: {reply.option}")


def _five_stones_xy(state: OpeningState) -> list[tuple[int, int, int]]:
    """The 5 stones on a 5-stone (PLACE2-resolved) OpeningState as (x,y,color).

    3 black + 2 white; the engine picks a color off the board and (for
    ONE_COORD) plays its first normal move. Each stone's TRUE color is read off
    the absolute planes (do NOT assume a fixed black/white alternation), so the
    engine sees the exact position regardless of the listing order.
    """
    return _colored_stones_xy(state)


def _engine_pick(
    state: OpeningState, engine: ExternalEnginePlayer
) -> tuple[OpeningState, int | None]:
    """Engine is OPENER picking color at the 5-stone PICK node: ``swap2_pick``.

    Branches:
      * SWAP        -> engine took BLACK: ``apply(PICK_BLACK)`` (we are white,
                       move next). No engine move revealed.
      * ONE_COORD   -> engine took WHITE and played move 6 at (x,y):
                       ``apply(PICK_WHITE)``; the engine's FIRST normal move is
                       (x,y) — returned so the driver applies it directly instead
                       of re-querying the engine.

    Returns ``(advanced_state, engine_first_normal_action_or_None)``.
    """
    stones = _five_stones_xy(state)
    reply = engine.swap2_pick(stones)
    if reply.option is Swap2Option.SWAP:
        return state.apply(PICK_BLACK), None
    if reply.option is Swap2Option.ONE_COORD:
        (x, y) = reply.coords[0]
        return state.apply(PICK_WHITE), _sq(x, y)
    raise ValueError(f"engine swap2_pick returned an unexpected reply: {reply.option}")


# ---------------------------------------------------------------------------
# Per-game driver.
# ---------------------------------------------------------------------------


@dataclass
class GameOutcome:
    """One finished swap2 game, from our net's perspective."""

    our_role: Actor          # OPENER or RESPONDER (our net's negotiation role)
    opener_color: Color      # color the opener played the whole game
    our_color: Color         # color our net ended up playing
    result: str              # "win" / "loss" / "draw" (our net's perspective)
    normal_plies: int        # plies of normal play after the opening
    opening_stones: int      # stones placed during the negotiation


def play_swap2_game(
    oracle: OracleFn,
    engine: ExternalEnginePlayer,
    our_role: Actor,
    rng: np.random.Generator,
    *,
    model_picker,
) -> GameOutcome:
    """Play ONE full swap2 game: negotiate the opening, then play normally.

    ``our_role`` assigns which negotiation role our net plays; the engine plays
    the other. ``oracle`` is our net's swap2 oracle (a thin wrapper around the
    net); ``model_picker`` is the eval MCTS picker for normal play. Returns a
    :class:`GameOutcome` attributed to our net.
    """
    def actor_is_ours(actor: Actor) -> bool:
        return actor == our_role

    state = initial_opening()
    # An engine first-normal-move revealed by a ONE_COORD pick (apply directly,
    # do not re-query the engine for it).
    engine_first_normal: int | None = None

    # -- negotiate the opening -----------------------------------------------
    while not state.is_done():
        if actor_is_ours(state.to_act):
            action, _ = agent_act(oracle, state, rng, greedy=True)
            state = state.apply(action)
        else:
            # Engine's turn: resolve the whole boundary with ONE protocol call.
            if state.node_is_opener_start():
                state = _engine_open(state, engine)
            elif state.phase is Phase.CHOICE and state.to_act == Actor.RESPONDER:
                state = _engine_respond(state, engine)
            elif state.phase is Phase.CHOICE and state.to_act == Actor.OPENER:
                state, engine_first_normal = _engine_pick(state, engine)
            else:  # pragma: no cover - the engine only acts at these boundaries
                raise ValueError(
                    f"engine reached an unexpected opening node: {state.node!r}"
                )

    # -- attribution bookkeeping (color our net plays the whole game) --------
    opener_color = state.opener_color
    assert opener_color is not None
    our_color = opener_color if our_role == Actor.OPENER else _other(opener_color)
    opening_stones = state.n_black + state.n_white

    # -- play normally from the handed-off position --------------------------
    gs = state.to_normal()
    # Side-to-move when normal play begins maps to an actor -> our_net or engine.
    mover_actor = state.mover_actor()
    our_turn = actor_is_ours(mover_actor)

    winner_color: Color | None = None
    plies = 0
    while plies < _MAX_NORMAL_PLIES:
        done, _ = gs.is_terminal()
        if done:
            break
        if our_turn:
            action = model_picker(gs, rng)
        else:
            if engine_first_normal is not None:
                action = engine_first_normal
                engine_first_normal = None
            else:
                action = engine(gs, rng)
        # The mover's color this ply: side-to-move is encoded by gs.board[0]; we
        # track color explicitly from move parity off the opening's mover_color.
        mover_color = _mover_color(state, plies)
        gs = gs.apply(action)
        plies += 1
        done, v = gs.is_terminal()
        if done:
            if v == -1.0:  # the side that JUST moved won
                winner_color = mover_color
            break
        our_turn = not our_turn

    if winner_color is None:
        result = "draw"
    elif winner_color == our_color:
        result = "win"
    else:
        result = "loss"

    return GameOutcome(
        our_role=our_role,
        opener_color=opener_color,
        our_color=our_color,
        result=result,
        normal_plies=plies,
        opening_stones=opening_stones,
    )


def _other(color: Color) -> Color:
    return Color.WHITE if color == Color.BLACK else Color.BLACK


def _mover_color(done_state: OpeningState, ply_idx: int) -> Color:
    """Color of the side to move at normal-play ply ``ply_idx`` (0-based).

    Normal play strictly alternates colors from the opening's ``mover_color``;
    ply 0 is mover_color, ply 1 the other, and so on.
    """
    base = done_state.mover_color
    assert base is not None
    if ply_idx % 2 == 0:
        return base
    return _other(base)


# A small structural helper used by the driver to recognise the engine-OPENER
# start node without reaching into the private _Node enum. Monkey-add to
# OpeningState would mutate a module we don't own, so we detect it positionally:
# the start is the only node where the board is empty AND it is a PLACE node for
# the OPENER. (PLACE + OPENER + 0 stones uniquely identifies P1_S1.)
def _node_is_opener_start(self: OpeningState) -> bool:
    return (
        self.phase is Phase.PLACE
        and self.to_act == Actor.OPENER
        and self.n_black == 0
        and self.n_white == 0
    )


# Bind as a method-like accessor without editing swap2.py (we only CALL it).
OpeningState.node_is_opener_start = _node_is_opener_start  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Match-level driver: many games, role-alternated, with a result dict that
# mirrors eval_vs_rapfi's shape where it makes sense.
# ---------------------------------------------------------------------------


@dataclass
class Swap2EvalResult:
    """Aggregate of an eval_swap2 run, mirroring the Rapfi-eval result shape."""

    n_games: int
    wins: int
    losses: int
    draws: int
    # Breakdown by our net's FINAL color (the color it actually played).
    black: list[int] = field(default_factory=lambda: [0, 0, 0])  # [W, L, D]
    white: list[int] = field(default_factory=lambda: [0, 0, 0])  # [W, L, D]
    # Breakdown by our net's negotiation ROLE.
    opener: list[int] = field(default_factory=lambda: [0, 0, 0])  # [W, L, D]
    responder: list[int] = field(default_factory=lambda: [0, 0, 0])  # [W, L, D]
    # opener_color distribution across the negotiated games.
    opener_color_black: int = 0
    opener_color_white: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(self.n_games, 1)

    def to_dict(self) -> dict:
        return {
            "n_games": self.n_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "by_color": {"black": list(self.black), "white": list(self.white)},
            "by_role": {
                "opener": list(self.opener),
                "responder": list(self.responder),
            },
            "opener_color_dist": {
                "black": self.opener_color_black,
                "white": self.opener_color_white,
            },
        }


def _wld_index(result: str) -> int:
    return {"win": 0, "loss": 1, "draw": 2}[result]


def aggregate_outcomes(outcomes: list[GameOutcome]) -> Swap2EvalResult:
    """Aggregate per-game :class:`GameOutcome`s into a :class:`Swap2EvalResult`.

    Pure + unit-testable (no engine / GPU). The aggregate W/L/D and the per-color
    / per-role splits are consistent by construction: black[i] + white[i] ==
    opener[i] + responder[i] == the aggregate count for each of W/L/D.
    """
    res = Swap2EvalResult(n_games=len(outcomes), wins=0, losses=0, draws=0)
    for o in outcomes:
        i = _wld_index(o.result)
        if i == 0:
            res.wins += 1
        elif i == 1:
            res.losses += 1
        else:
            res.draws += 1
        (res.black if o.our_color == Color.BLACK else res.white)[i] += 1
        (res.opener if o.our_role == Actor.OPENER else res.responder)[i] += 1
        if o.opener_color == Color.BLACK:
            res.opener_color_black += 1
        else:
            res.opener_color_white += 1
    return res


# ---------------------------------------------------------------------------
# Per-game determinism. game i gets a FIXED (role, rng) derived ONLY from
# (seed, i) — not from the order or process it runs in. role alternates exactly
# as the serial loop (game 0 opens, game 1 responds, ...), and the per-game rng
# is spawned from a SeedSequence(seed) child so it is independent of every other
# game's draws. This is the correctness guarantee for --jobs: the multiset of
# (role, rng) games is identical for jobs=1 and jobs=J, so any partition of the
# game indices across workers reproduces the SAME set of games — only faster.
# ---------------------------------------------------------------------------


def _game_role(i: int) -> Actor:
    """Negotiation role for game index ``i`` (game 0 opens, alternating)."""
    return Actor.OPENER if i % 2 == 0 else Actor.RESPONDER


def _game_rng(seed: int, n_games: int, i: int) -> np.random.Generator:
    """Independent per-game RNG for game ``i``, derived only from ``(seed, i)``.

    Spawned children of one ``SeedSequence(seed)`` are mutually independent and
    reproducible, so game ``i`` draws the SAME stream regardless of which worker
    (or the serial loop) runs it, or in what order. We spawn ``n_games`` children
    once and index by ``i`` (cheap; ``n_games`` is small for an eval gate).
    """
    children = np.random.SeedSequence(seed).spawn(n_games)
    return np.random.default_rng(children[i])


def _split_game_indices(n_games: int, jobs: int) -> list[list[int]]:
    """Partition ``range(n_games)`` into ``min(jobs, n_games)`` contiguous chunks.

    Contiguous (not strided) so each worker plays a block of game indices; the
    remainder is spread one-per-chunk over the leading chunks. Pure +
    unit-testable: the chunks concatenate back to ``range(n_games)`` exactly, so
    the union of games played is identical to the serial range.
    """
    if n_games <= 0:
        return []
    k = max(1, min(jobs, n_games))
    base, rem = divmod(n_games, k)
    chunks: list[list[int]] = []
    start = 0
    for c in range(k):
        size = base + (1 if c < rem else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


def _play_game_indices(
    *,
    checkpoint: str,
    engine_cfg,
    indices: list[int],
    n_games: int,
    sims: int,
    c_puct: float,
    seed: int,
    device: str | None,
) -> list[GameOutcome]:
    """Load the net (CPU/MPS per ``device``), build its OWN engine subprocess, and
    play exactly the games at ``indices``. The unit of work for one worker.

    Each call owns a fresh :class:`ExternalEnginePlayer` (engines are stateful
    subprocesses, NOT shareable across processes), loads the model once, and
    plays its assigned game indices with each game's deterministic (role, rng).
    Returns the per-game outcomes in ``indices`` order.
    """
    from gomoku.eval import mcts_picker
    from gomoku.external_engine import ExternalEngineConfig
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.self_play import _make_swap2_oracle
    from gomoku.util import pick_device

    if isinstance(engine_cfg, dict):
        engine_cfg = ExternalEngineConfig(**engine_cfg)

    dev = pick_device(device if device is not None else os.environ.get("GOMOKU_DEVICE"))
    model, _ = load_checkpoint(checkpoint, device=dev)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, dev)
    oracle = _make_swap2_oracle(evaluator)
    model_picker = mcts_picker(evaluator, n_simulations=sims, c_puct=c_puct)

    engine = ExternalEnginePlayer(engine_cfg)
    outcomes: list[GameOutcome] = []
    try:
        for i in indices:
            role = _game_role(i)
            rng = _game_rng(seed, n_games, i)
            outcomes.append(
                play_swap2_game(oracle, engine, role, rng, model_picker=model_picker)
            )
    finally:
        engine.close()
    return outcomes


def _worker_entry(task: dict) -> list[GameOutcome]:
    """Spawn-pool worker target: play a chunk of game indices. Top-level (and so
    picklable) for the ``spawn`` start method, which macOS+MPS/torch requires."""
    return _play_game_indices(
        checkpoint=task["checkpoint"],
        engine_cfg=task["engine_cfg"],
        indices=task["indices"],
        n_games=task["n_games"],
        sims=task["sims"],
        c_puct=task["c_puct"],
        seed=task["seed"],
        device=task["device"],
    )


def eval_swap2_vs_engine(
    checkpoint: str,
    engine_cfg,
    n_games: int,
    *,
    timeout_ms: int | None = None,
    sims: int = 100,
    c_puct: float = 1.5,
    seed: int = 0,
    device: str | None = None,
    jobs: int = 1,
) -> dict:
    """Score a checkpoint against an external engine under the swap2 protocol.

    ``engine_cfg`` is an :class:`gomoku.external_engine.ExternalEngineConfig` (or
    a dict of its kwargs). ``timeout_ms`` overrides the config's per-move budget
    when given. Roles ALTERNATE across games for balance (game 0: our net opens;
    game 1: our net responds; ...). Returns a result dict mirroring the
    eval_vs_rapfi shape, plus per-color / per-role / opener-color breakdowns.

    ``jobs`` (default 1 = serial) splits ``n_games`` across that many worker
    PROCESSES, each with its OWN Rapfi subprocess + a net loaded on the resolved
    device, to run far more games in the same wall-clock for an hourly gate. The
    per-game (role, rng) is derived ONLY from ``(seed, game_index)``, so the
    multiset of games is identical for any ``jobs`` and the aggregate for jobs=J
    is EQUIVALENT to jobs=1 at the same ``seed``/``n_games`` — just faster. Mirrors
    ``scripts/eval_vs_rapfi.py``'s ``--jobs`` concurrency (a ``spawn`` pool).
    """
    from gomoku.external_engine import ExternalEngineConfig

    if isinstance(engine_cfg, dict):
        engine_cfg = ExternalEngineConfig(**engine_cfg)
    if timeout_ms is not None:
        engine_cfg.timeout_ms = timeout_ms

    t0 = time.time()
    if jobs <= 1 or n_games <= 1:
        outcomes = _play_game_indices(
            checkpoint=checkpoint,
            engine_cfg=engine_cfg,
            indices=list(range(n_games)),
            n_games=n_games,
            sims=sims,
            c_puct=c_puct,
            seed=seed,
            device=device,
        )
    else:
        import multiprocessing as mp

        chunks = _split_game_indices(n_games, jobs)
        tasks = [
            {
                "checkpoint": checkpoint,
                "engine_cfg": engine_cfg,
                "indices": idxs,
                "n_games": n_games,
                "sims": sims,
                "c_puct": c_puct,
                "seed": seed,
                "device": device,
            }
            for idxs in chunks
        ]
        # spawn (macOS default): fork + torch/MPS is unsafe. Each worker builds
        # its OWN engine subprocess + net; the pool is sized to the chunk count.
        ctx = mp.get_context("spawn")
        per_chunk: list[list[GameOutcome]] = [None] * len(tasks)  # type: ignore[list-item]
        with ctx.Pool(processes=len(tasks)) as pool:
            for ci, outs in enumerate(pool.imap(_worker_entry, tasks)):
                per_chunk[ci] = outs
        # imap preserves task order; concatenating in chunk order reassembles the
        # games in ascending game-index order (chunks are contiguous), matching
        # the serial outcome ORDER as well as its multiset.
        outcomes = [o for chunk in per_chunk for o in chunk]
    wall = time.time() - t0

    res = aggregate_outcomes(outcomes)
    out = res.to_dict()
    out.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": os.path.abspath(checkpoint),
            "engine": engine_cfg.label,
            "engine_cmd": engine_cfg.cmd,
            "timeout_ms": engine_cfg.timeout_ms,
            "board_size": engine_cfg.board_size,
            "rule": engine_cfg.rule,
            "model_sims": sims,
            "model_c_puct": c_puct,
            "protocol": "swap2",
            "jobs": jobs,
            "wall_secs": round(wall, 2),
        }
    )
    return out


def main() -> None:  # pragma: no cover - thin CLI wrapper around the driver
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    ap.add_argument("--engine", required=True, help="Path to the engine brain (e.g. pbrain-rapfi).")
    ap.add_argument("--label", default="engine", help="Engine label for the record.")
    ap.add_argument("--timeout-ms", type=int, default=1000, help="Engine per-move budget (ms).")
    ap.add_argument("--sims", type=int, default=100, help="Our MCTS sims per normal move.")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--n-games", type=int, default=20, help="Games (roles alternated).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", type=int, default=0, help="0 = freestyle.")
    ap.add_argument("--size", type=int, default=BOARD_SIZE)
    ap.add_argument(
        "--jobs", type=int, default=1, metavar="J",
        help=(
            "Parallel worker processes (1 = serial, default). J>1 splits the "
            "games across J spawn-workers, each with its own Rapfi subprocess + "
            "net, so an hourly gate runs more games in the same wall-clock. The "
            "aggregate is equivalent to --jobs 1 at the same --seed/--n-games."
        ),
    )
    ap.add_argument("--out", default=None, help="JSONL output path (append).")
    args = ap.parse_args()

    from gomoku.external_engine import ExternalEngineConfig

    engine_path = os.path.abspath(args.engine)
    if not os.path.exists(engine_path):
        raise SystemExit(f"engine binary not found: {engine_path}")
    cfg = ExternalEngineConfig(
        cmd=engine_path, timeout_ms=args.timeout_ms, label=args.label,
        rule=args.rule, board_size=args.size,
    )
    out = eval_swap2_vs_engine(
        args.checkpoint, cfg, args.n_games,
        sims=args.sims, c_puct=args.c_puct, seed=args.seed, jobs=args.jobs,
    )
    print(
        f"{args.label}: {out['wins']}W-{out['losses']}L-{out['draws']}D / "
        f"{out['n_games']} games  win_rate={out['win_rate']:.2%}  "
        f"({out['wall_secs']:.1f}s)\n"
        f"  by color: black {out['by_color']['black']} white {out['by_color']['white']}\n"
        f"  by role:  opener {out['by_role']['opener']} responder {out['by_role']['responder']}\n"
        f"  opener_color: {out['opener_color_dist']}"
    )
    if args.out:
        import json

        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(out) + "\n")
        print(f"wrote 1 record to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
