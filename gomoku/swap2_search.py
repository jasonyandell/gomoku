"""Swap2 opening NEGOTIATOR — drives the negotiation with the net (v1).

What this is
============
``gomoku.swap2`` is the *pure game-logic* state machine (no net, no search).
This module is the v1 **negotiator**: it walks that state machine using a
network oracle to play the opening, then hands a normal position to existing
self-play / eval.

Why a negotiator at all
=======================
15x15 freestyle gomoku is a first-player win, so plain self-play collapses
(white is hopeless → no winnable white data → the value/policy heads never see
balanced games). Swap2 is a *negotiated* opening: a player is never forced onto
the lost side, so self-play yields ~50/50 data again. This negotiator produces
that opening, then ``Swap2Result.normal_state`` is the position normal play
begins from.

The v1 algorithm (two node kinds, two very different treatments)
================================================================
The negotiation has two node phases (see :mod:`gomoku.swap2`):

* **PLACE nodes are SAMPLED for diversity** — *not* searched and *not* recorded
  as training examples. We read the oracle's policy at ``s.net_state()``, mask
  to legal empty squares, renormalize, optionally mix Dirichlet noise at the
  first opener placement, and sample a square with temperature. Placements stay
  a plain spatial decision over the existing board-policy head; v1 collects no
  target for them.

* **CHOICE nodes are selected by VALUE and ARE recorded** — they are the only
  non-spatial decisions, and v1 has no trained choice head yet, so we pick by a
  one-ply value comparison and record a :class:`ChoiceRecord` as a target for a
  future choice head. For each legal slot we compute the *chooser's* value by
  greedily resolving the resulting state to DONE (placements take argmax legal
  policy; nested choice nodes recurse and each maximizes its *own* chooser
  value — proper minimax), evaluating the resulting normal position with the
  oracle, and re-signing the value to the chooser's perspective with
  :func:`gomoku.swap2.backup_sign`. The recorded target is a softmax over those
  values; the played slot is the argmax (greedy) or a sample.

Value attribution
==================
All value sign-flipping is delegated to :func:`gomoku.swap2.backup_sign` — the
opening breaks the "flip perspective every ply" invariant (the opener acts three
times in a row), so signs are driven by each node's ``to_act`` actor, never by
ply parity. We never reinvent that logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

from gomoku.board_config import BOARD_SIZE
from gomoku.game import GameState
from gomoku.swap2 import (
    Actor,
    Color,
    OpeningState,
    Phase,
    backup_sign,
    initial_opening,
)

# An oracle maps a (canonical) GameState to (policy_probs, value):
#   policy_probs : (N_ACTIONS,) float, a probability distribution over squares.
#   value        : float in [-1, 1], from gs's SIDE-TO-MOVE perspective.
# Real callers pass a thin wrapper around the net; tests pass deterministic
# fakes. Declared both as a Protocol (for typing) and a callable alias.
class Oracle(Protocol):
    def __call__(self, gs: GameState) -> tuple[np.ndarray, float]:
        ...


OracleFn = Callable[[GameState], tuple[np.ndarray, float]]


@dataclass
class ChoiceRecord:
    """A recorded negotiation choice — a future choice-head training example.

    ``planes`` frames the choice node for the net (``cs.net_state().to_planes()``,
    shape ``(N_INPUT_PLANES, N, N)``). ``target`` is the value-derived soft label
    over the 3 choice slots (illegal slots zeroed, sums to 1 over legal slots);
    ``chosen`` is the slot actually played at negotiation time.
    """

    planes: np.ndarray          # (N_INPUT_PLANES, N, N) float32
    to_act: Actor               # chooser at this node (RESPONDER or OPENER)
    legal_mask: np.ndarray      # (N_CHOICES,) bool
    target: np.ndarray          # (N_CHOICES,) float, sums to 1 over legal slots
    chosen: int                 # the slot actually applied


@dataclass
class Swap2Result:
    """The outcome of one full negotiation.

    ``normal_state`` is the canonical, perspective-correct position normal play
    begins from (hand straight to existing self-play / eval). ``opener_color`` /
    ``mover_actor`` carry the opener<->color mapping the color-blind GameState
    drops. ``choice_records`` are the negotiation choices to fold into a future
    choice-head's training set; placements are not recorded in v1.
    """

    normal_state: GameState
    opener_color: Color
    mover_actor: Actor
    choice_records: list[ChoiceRecord]


# ---------------------------------------------------------------------------
# Core per-node logic, shared by negotiate() and agent_act().
# ---------------------------------------------------------------------------


def _masked_renorm(probs: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Mask ``probs`` to legal squares and renormalize.

    Falls back to a uniform distribution over legal squares when the oracle puts
    no mass on any legal square (degenerate, but keeps sampling well-defined)."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1).copy()
    p[~legal_mask] = 0.0
    total = p.sum()
    if total <= 0.0:
        p = legal_mask.astype(np.float64)
        total = p.sum()
    return p / total


def _sample_square(
    oracle: OracleFn,
    s: OpeningState,
    rng: np.random.Generator,
    *,
    temperature: float,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    add_noise: bool,
) -> int:
    """Sample one legal square at a PLACE node from the oracle's policy.

    Diversity, not strength: masked+renormalized oracle policy, optional
    Dirichlet noise (root exploration at the first opener placement only),
    temperature applied to the probabilities. ``temperature == 0`` is greedy
    (argmax). Always returns a legal (empty) square."""
    legal_mask = s.legal_square_mask()
    probs, _ = oracle(s.net_state())
    p = _masked_renorm(probs, legal_mask)

    if add_noise and dirichlet_frac > 0.0:
        legal_idx = np.flatnonzero(legal_mask)
        if legal_idx.size > 0:
            noise = rng.dirichlet([dirichlet_alpha] * legal_idx.size)
            p[legal_idx] = (1.0 - dirichlet_frac) * p[legal_idx] + dirichlet_frac * noise

    if temperature <= 0.0:
        # Greedy: argmax over legal mass (ties broken by first index).
        return int(np.argmax(p))

    if temperature != 1.0:
        # Temperature on probabilities: p^(1/T), renormalized.
        with np.errstate(divide="ignore"):
            p = np.power(p, 1.0 / temperature)
        total = p.sum()
        p = p / total if total > 0 else legal_mask.astype(np.float64) / legal_mask.sum()

    return int(rng.choice(len(p), p=p))


def _greedy_square(oracle: OracleFn, s: OpeningState) -> int:
    """The argmax legal square under the oracle's policy (for choice rollouts)."""
    legal_mask = s.legal_square_mask()
    probs, _ = oracle(s.net_state())
    p = _masked_renorm(probs, legal_mask)
    return int(np.argmax(p))


def _resolve_greedy(oracle: OracleFn, s: OpeningState) -> OpeningState:
    """Greedily roll ``s`` forward to a DONE state for choice evaluation.

    At PLACE nodes take the argmax legal oracle policy. At CHOICE nodes (the only
    one reachable here is the opener's pick nested inside the responder's PLACE2
    line) pick the slot that maximizes THAT node's own chooser value — recursing
    into :func:`_choice_values`. This makes the whole rollout a proper minimax:
    every chooser maximizes its own value, so an outer chooser sees the value
    AFTER an inner adversary's best reply."""
    while not s.is_done():
        if s.phase is Phase.PLACE:
            s = s.apply(_greedy_square(oracle, s))
        else:  # Phase.CHOICE — pick the value-maximizing slot for THIS chooser.
            values, legal_mask = _choice_values(oracle, s)
            slot = _argmax_legal(values, legal_mask)
            s = s.apply(slot)
    return s


def _argmax_legal(values: np.ndarray, legal_mask: np.ndarray) -> int:
    """Argmax of ``values`` restricted to legal slots."""
    masked = np.where(legal_mask, values, -np.inf)
    return int(np.argmax(masked))


def _choice_values(
    oracle: OracleFn, cs: OpeningState
) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot value of a CHOICE node, in the chooser's (``cs.to_act``) frame.

    For each legal slot: apply it, greedily resolve to DONE (with minimax over
    any nested choice), evaluate the resulting normal position with the oracle,
    and re-sign the value to the chooser's perspective via ``backup_sign``.
    Illegal slots get ``-inf`` (never chosen, never weighted). This is the white-
    box primitive the value-attribution tests target."""
    legal_mask = cs.legal_choice_mask()
    values = np.full(legal_mask.shape, -np.inf, dtype=np.float64)
    for slot in np.flatnonzero(legal_mask):
        done = _resolve_greedy(oracle, cs.apply(int(slot)))
        gs = done.to_normal()
        _, v_norm = oracle(gs)
        # v_norm is from gs's side-to-move (== done.mover_actor()) perspective.
        # Re-sign to the chooser's perspective.
        values[slot] = backup_sign(cs, done.mover_actor()) * float(v_norm)
    return values, legal_mask


def _choice_target(values: np.ndarray, legal_mask: np.ndarray, choice_tau: float) -> np.ndarray:
    """Softmax(values / choice_tau) over legal slots; illegal slots → 0."""
    target = np.zeros(legal_mask.shape, dtype=np.float64)
    legal_idx = np.flatnonzero(legal_mask)
    v = values[legal_idx]
    tau = choice_tau if choice_tau > 0.0 else 1e-8
    # Numerically stable softmax over legal slots.
    z = v / tau
    z = z - z.max()
    e = np.exp(z)
    target[legal_idx] = e / e.sum()
    return target


def _decide_choice(
    oracle: OracleFn,
    cs: OpeningState,
    rng: np.random.Generator,
    *,
    choice_tau: float,
    greedy: bool,
) -> tuple[int, ChoiceRecord]:
    """Select a choice slot at ``cs`` and build its :class:`ChoiceRecord`."""
    values, legal_mask = _choice_values(oracle, cs)
    target = _choice_target(values, legal_mask, choice_tau)
    if greedy:
        chosen = _argmax_legal(values, legal_mask)
    else:
        chosen = int(rng.choice(len(target), p=target))
    record = ChoiceRecord(
        planes=cs.net_state().to_planes(),
        to_act=cs.to_act,
        legal_mask=legal_mask.copy(),
        target=target,
        chosen=chosen,
    )
    return chosen, record


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def negotiate(
    oracle: OracleFn,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_frac: float = 0.25,
    choice_tau: float = 0.1,
    greedy_choices: bool = False,
) -> Swap2Result:
    """Play the FULL swap2 negotiation with one oracle (both roles).

    For self-play: a single net drives both the opener and the responder.
    Placements are sampled (diversity, Dirichlet noise at the first opener
    placement); choices are selected by value and recorded.

    Returns a :class:`Swap2Result` whose ``normal_state`` is the position normal
    play begins from, and whose ``choice_records`` are choice-head training
    examples (placements are not recorded in v1).
    """
    s = initial_opening()
    records: list[ChoiceRecord] = []
    first_place_seen = False

    while not s.is_done():
        if s.phase is Phase.PLACE:
            add_noise = not first_place_seen  # root exploration at first opener placement
            square = _sample_square(
                oracle,
                s,
                rng,
                temperature=temperature,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_frac=dirichlet_frac,
                add_noise=add_noise,
            )
            first_place_seen = True
            s = s.apply(square)
        else:  # Phase.CHOICE
            chosen, record = _decide_choice(
                oracle, s, rng, choice_tau=choice_tau, greedy=greedy_choices
            )
            records.append(record)
            s = s.apply(chosen)

    return Swap2Result(
        normal_state=s.to_normal(),
        opener_color=s.opener_color,
        mover_actor=s.mover_actor(),
        choice_records=records,
    )


def agent_act(
    oracle: OracleFn,
    state: OpeningState,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    choice_tau: float = 0.1,
    greedy: bool = False,
) -> tuple[int, Optional[ChoiceRecord]]:
    """Decide a SINGLE node's action — for eval, where a harness alternates our
    turns with an external engine's.

    PLACE node → ``(square, None)`` (a sampled legal square; no Dirichlet noise
    here — eval wants the agent's own policy, not root exploration). CHOICE node
    → ``(slot, ChoiceRecord)``. Raises if ``state`` is already DONE.
    """
    if state.is_done():
        raise ValueError("agent_act called on a DONE opening")
    if state.phase is Phase.PLACE:
        square = _sample_square(
            oracle,
            state,
            rng,
            temperature=temperature,
            dirichlet_alpha=0.0,
            dirichlet_frac=0.0,
            add_noise=False,
        )
        return square, None
    chosen, record = _decide_choice(
        oracle, state, rng, choice_tau=choice_tau, greedy=greedy
    )
    return chosen, record
