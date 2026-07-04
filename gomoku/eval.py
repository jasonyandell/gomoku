"""Evaluation: play matches between two policies.

Two layers:

- `play_match_pickers` is the generic engine. Each player is a
  `Callable[[GameState, np.random.Generator], int]` (a "picker"). Random,
  heuristic, lookahead, and the MCTS-driven model all plug into this.

- `play_match` and `play_vs_random` are thin wrappers kept for backward
  compatibility with the training loop and any existing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from gomoku.game import BOARD_SIZE, GameState
from gomoku.mcts import Evaluator, MCTSGame, policy_from_visits, run_batched_mcts


Picker = Callable[[GameState, np.random.Generator], int]


@dataclass
class MatchResult:
    n_games: int
    wins: int
    losses: int
    draws: int
    # Per-color breakdown from the model's (player A's) perspective. A game is
    # counted under "black" if the model played black that game, else "white".
    # These are split tallies of the same outcomes the aggregates count, so by
    # construction black_w + white_w == wins (likewise losses/draws). They
    # default to 0 so existing MatchResult(...) call sites stay valid and the
    # aggregate fields remain the canonical strength signal; the split lets the
    # loss-tail analysis see whether losses cluster on white (second player).
    black_w: int = 0
    black_l: int = 0
    black_d: int = 0
    white_w: int = 0
    white_l: int = 0
    white_d: int = 0

    @property
    def win_rate(self) -> float:
        # Draws count as half-wins, standard chess scoring.
        return (self.wins + 0.5 * self.draws) / max(self.n_games, 1)


def _random_pick(state: GameState, rng: np.random.Generator) -> int:
    legal = state.legal_actions()
    return int(rng.choice(legal))


def vcf_overlay_picker(
    base: Picker,
    *,
    max_nodes: int,
    max_depth: int = 0,
) -> Picker:
    """Wrap a picker with an eval-time root VCF overlay.

    Before delegating to ``base``, run a bounded ``solve_vcf`` from the current
    root. If a forced continuous-four win is proven for the side-to-move, play
    the solver's recommended move; otherwise fall through to ``base`` unchanged.

    When ``max_nodes <= 0`` the overlay is OFF — this function returns ``base``
    UNCHANGED (same object, no wrapping), so OFF is byte-identical to the
    pre-lever path by construction.

    The solver is bounded by ``max_nodes`` (and ``max_depth`` when > 0; else
    ``vcf.DEFAULT_MAX_DEPTH``). Hitting the cap returns ``has_forced_win=False``
    with ``hit_cap=True`` — the overlay simply falls through to ``base``, so a
    tight cap can never block the picker indefinitely.

    This is EVAL-ONLY. The self-play / generation path uses ``--vcf-teacher`` on
    a separate seam (see ``selfplay_worker.py``); the two never collide.
    """
    if max_nodes <= 0:
        return base

    # Lazy import — keeps mcts_picker callers that pass eval_vcf_nodes=0
    # (the default) from paying the import cost.
    from gomoku.vcf import DEFAULT_MAX_DEPTH, solve_vcf

    depth = max_depth if max_depth > 0 else DEFAULT_MAX_DEPTH

    def pick(state: GameState, rng: np.random.Generator) -> int:
        # solve_vcf treats plane 0 as attacker (side-to-move) and plane 1 as
        # defender — which is exactly GameState's canonical board layout.
        res = solve_vcf(state.board, max_depth=depth, max_nodes=max_nodes)
        if res.has_forced_win and res.winning_move is not None:
            return int(res.winning_move)
        return base(state, rng)

    return pick


def _load_vct_solver():
    """Import the GPU VCT oracle (lazy; behind an indirection so tests can
    monkeypatch it without paying the MLX/Metal import + compile)."""
    try:
        from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb
    except ModuleNotFoundError:
        # Console entry points (gomoku-arena etc.) don't put the repo root on
        # sys.path the way `python -m gomoku.X` from the checkout does; the
        # editable install means gomoku/'s parent IS the repo root.
        import sys
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.vct_metal.mega_vct_bb import solve_vct_mega_bb
    return solve_vct_mega_bb


def vct_finish_picker(base: Picker, *, budget: int = 0) -> Picker:
    """Wrap a picker with an eval-time GPU-VCT FINISHER (issue #99).

    Before delegating to ``base``, run a bounded batched ``solve_vct_mega_bb`` (the
    validated GPU oracle) from the current root. If the side to move has a forced
    VCT (Victory by Continuous Threats), play the oracle's winning move; else fall
    through to ``base`` unchanged. Applied every turn, this drives a detected VCT
    to a REAL five-in-a-row rote — so a VCT-terminus net (issue #98) wins genuine
    games against ANY opponent through the standard match harness, with no special
    "VCT counts as a win" harness (the game still ends on an actual five, so wins
    are real). This is the deployable hybrid player: policy to the VCT, solver to
    the win.

    ``budget`` (default 0 = OFF → returns ``base`` UNCHANGED, byte-identical, and
    never imports MLX) is the per-board node cap; cap50 (=50) matches the first-VCT
    detection threshold the net is trained toward. A cap50 VCT's continuations are
    sub-proofs of the original <=50-node proof, so cap50 also CONVERTS turn by turn.
    ``hit_cap`` is treated as no-win (fall through — a tight cap never wedges the
    picker). VCT is a strict superset of the VCF overlay
    (:func:`vcf_overlay_picker`), so with both on this is checked first.

    EVAL / PLAY-ONLY. The self-play generation path terminates AT the VCT instead
    (``--vct-terminus``); the two seams never collide.
    """
    if budget <= 0:
        return base

    # Lazy import — only a VCT-finisher build pays the MLX/Metal compile; a
    # zero-torch `random vs heuristic` match never imports it.
    solve_vct_mega_bb = _load_vct_solver()

    def pick(state: GameState, rng: np.random.Generator) -> int:
        # state.board is already (2,N,N) side-to-move-relative: plane 0 attacker,
        # plane 1 defender — exactly the oracle's input (the layout the VCF overlay
        # above also relies on). Batch of 1 (eval is per-move single-board).
        boards = np.asarray(state.board, dtype=bool)[None]      # (1,2,N,N)
        win, _hit, move = solve_vct_mega_bb(
            boards, max_nodes=budget, return_move=True)
        m = int(move[0])
        if bool(win[0]) and 0 <= m and bool(np.asarray(state.legal_mask()).reshape(-1)[m]):
            return m
        return base(state, rng)

    return pick


def _infer_moves_from_diff(
    rs: GameState, state: GameState
) -> tuple[int, ...] | None:
    """Infer the sequence of 1 or 2 actions played to get from ``rs`` to ``state``.

    Returns the action tuple, or ``None`` if the diff doesn't match a clean
    1- or 2-ply continuation (e.g. opponent surprise via VCF-overlay short-
    circuit, or stale state). Caller falls back to rebuilding a fresh tree.

    Mechanics (canonical-perspective bookkeeping in ``GameState.apply``):
      - ``rs.board[0]``  = side-to-move at rs's stones; ``rs.board[1]`` = opp.
      - After 1 ply: side flips. ``state.board[1]`` now holds rs's mover's
        stones (incl new one). New cell = ``state.board[1] & ~rs.board[0]``.
      - After 2 plies: perspective flips back. ``state.board[0]`` holds rs's
        mover's stones (incl their move ``a``); ``state.board[1]`` holds the
        opponent's stones (incl their reply ``b``).
    """
    mc_diff = state.move_count - rs.move_count
    if mc_diff == 1:
        new_cells = state.board[1] & (~rs.board[0])
        if int(new_cells.sum()) != 1:
            return None
        coords = np.argwhere(new_cells)
        r, c = int(coords[0, 0]), int(coords[0, 1])
        a = r * BOARD_SIZE + c
        # Verify by replaying — defends against any sneaky board-shape mismatch.
        try:
            ns = rs.apply(a)
        except Exception:
            return None
        if not np.array_equal(ns.board, state.board):
            return None
        return (a,)
    if mc_diff == 2:
        new_a_cells = state.board[0] & (~rs.board[0])
        new_b_cells = state.board[1] & (~rs.board[1])
        if int(new_a_cells.sum()) != 1 or int(new_b_cells.sum()) != 1:
            return None
        ra, ca = np.argwhere(new_a_cells)[0]
        rb, cb = np.argwhere(new_b_cells)[0]
        a = int(ra) * BOARD_SIZE + int(ca)
        b = int(rb) * BOARD_SIZE + int(cb)
        try:
            ns = rs.apply(a).apply(b)
        except Exception:
            return None
        if not np.array_equal(ns.board, state.board):
            return None
        return (a, b)
    return None


def mcts_picker(
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.5,
    eval_vcf_nodes: int = 0,
    eval_vcf_depth: int = 0,
    fpu_reduction_c: float = 0.0,
    reuse_tree: bool = False,
    proven_prop: bool = False,
    proven_vcf_leaf_nodes: int = 0,
    vct_finish_nodes: int = 0,
) -> Picker:
    """Wrap a leaf evaluator into a one-shot picker for use in matches.

    Default (``reuse_tree=False``, BYTE-IDENTICAL legacy): builds a fresh MCTS
    tree per call — no cross-move reuse, no state held between picks.

    ``reuse_tree=True`` (the derby-jmi lever, EVAL-ONLY): returns a STATEFUL
    closure that holds one persistent ``MCTSGame`` across the match. On the
    first call it builds the tree at ``state``. On subsequent calls it infers
    the 1 or 2 actions played since the previous root (by board diff) and calls
    ``MCTSGame.advance_root`` for each — promoting the played child to be the
    new root and reusing its already-explored subtree (visits / W / priors).
    Effectively multiplies the per-move sim budget by the tree-reuse fraction
    at zero extra compute. Standard AlphaZero / KataGo / LCZero.

    Reset / safety rules (all rebuild a fresh tree):
      - First call (no game yet).
      - ``state.move_count`` decreased OR is 0 → new match.
      - Diff doesn't match a clean 1- or 2-ply continuation (e.g. opponent
        surprise via the VCF overlay short-circuit, mid-game pickup).

    EVAL-ONLY: the self-play / generation / training path NEVER calls this with
    ``reuse_tree=True`` — gen MCTS has its own tree-management semantics
    (Dirichlet noise per move, full sim budget per ply) and stays unchanged.

    ``eval_vcf_nodes`` (default 0 = OFF, byte-identical to the pre-lever path)
    enables an eval-time root VCF overlay: before MCTS picks a move, run a
    bounded ``solve_vcf`` from the current root; if a forced four-in-a-row win
    is proven, play that move; else fall through to the unchanged MCTS choice.
    ``eval_vcf_depth`` (default 0 → ``vcf.DEFAULT_MAX_DEPTH``) caps depth. The
    overlay is EVAL-ONLY — never used by self-play / generation / training.

    ``fpu_reduction_c`` (default 0.0 = OFF, byte-identical) enables KataGo
    First-Play Urgency reduction in MCTS selection: unvisited children inherit
    ``parent_V - c * sqrt(sum_visited_priors)`` instead of Q=0. Eval-only —
    self-play / gen MCTS stays untouched (their MCTSGame is constructed
    elsewhere with the default 0.0). KataGo uses c≈0.45 (subtree) / 0.20
    (root); LCZero 0.33.

    ``proven_prop`` (default False = OFF, byte-identical) enables KataGo-style
    proven-win/loss propagation through the MCTS tree (derby-b3n): terminal
    outcomes (and optionally bounded leaf-VCF results) propagate upward,
    selection deprioritizes proven-loss children (Q=-inf) and prioritizes
    proven-win children (Q=+inf), and on root short-circuit we play any proven
    winning move immediately. EVAL-ONLY: self-play / generation / training are
    NOT affected (their MCTSGame construction never sets this).

    ``proven_vcf_leaf_nodes`` (default 0 = OFF, byte-identical) enables a
    bounded ``solve_vcf`` call at MCTS leaf expansion (derby-b3n; KataGo's
    "graphSearch lite"). When >0, every newly-expanded leaf runs
    ``solve_vcf(state, max_nodes=N)``; on ``has_forced_win=True`` the leaf's
    ``proven_value`` is set to +1 immediately, propagating up via the proven-
    prop machinery. Requires ``proven_prop=True`` to have any effect. Reuses
    the bounded VCF solver from derby-b6r (same per-call work as
    ``--eval-vcf-nodes``).
    """

    if not reuse_tree:
        # Legacy path — byte-identical to pre-lever behavior when proven_prop
        # is OFF and proven_vcf_leaf_nodes is OFF. Same code shape otherwise.
        def pick(state: GameState, rng: np.random.Generator) -> int:
            g = MCTSGame(
                state,
                c_puct=c_puct,
                fpu_reduction_c=fpu_reduction_c,
                proven_prop=proven_prop,
                proven_vcf_leaf_nodes=proven_vcf_leaf_nodes,
                rng=rng,
            )
            run_batched_mcts(
                [g], evaluator, n_simulations=n_simulations, add_root_noise=False
            )
            # derby-b3n root short-circuit:
            #   1. If leaf-VCF proved a win at the root and supplied the
            #      first attacker move (root.proven_action), play it.
            #   2. Else, if any root CHILD is proven-winning for us
            #      (child.proven_value == -1 in the child's POV ⇒ opponent
            #      loses ⇒ we win), play that move.
            # Closes the "missing-one-sim" gap.
            if proven_prop:
                if (
                    g.root.proven_value == +1
                    and g.root.proven_action >= 0
                    and g.root.legal_mask[g.root.proven_action]
                ):
                    return int(g.root.proven_action)
                for a, child in g.root.children.items():
                    if child.proven_value == -1 and g.root.legal_mask[a]:
                        return int(a)
            pi = policy_from_visits(g.root, temperature=0.0)
            return int(np.argmax(pi))

        return vct_finish_picker(
            vcf_overlay_picker(
                pick, max_nodes=eval_vcf_nodes, max_depth=eval_vcf_depth
            ),
            budget=vct_finish_nodes,
        )

    # Stateful reuse-tree picker. Holds one MCTSGame across the match.
    # `state_holder` is a dict so the closure can rebind without `nonlocal`.
    state_holder: dict = {"game": None}

    def _build_fresh(state: GameState, rng: np.random.Generator) -> MCTSGame:
        return MCTSGame(
            state,
            c_puct=c_puct,
            fpu_reduction_c=fpu_reduction_c,
            proven_prop=proven_prop,
            proven_vcf_leaf_nodes=proven_vcf_leaf_nodes,
            rng=rng,
        )

    def pick(state: GameState, rng: np.random.Generator) -> int:
        g = state_holder["game"]
        rebuild = False
        if g is None:
            rebuild = True
        elif state.move_count == 0:
            # New match started — initial position. Even if g's root happens to
            # equal initial, rebuild to drop any stale stats from a prior game.
            rebuild = True
        elif state.move_count < g.root.state.move_count:
            # Time went backwards → new match or resumed earlier — rebuild.
            rebuild = True
        elif state.move_count == g.root.state.move_count:
            # Root already at this state (e.g. unusual replay) → no advance.
            if not np.array_equal(state.board, g.root.state.board):
                rebuild = True
        else:
            moves = _infer_moves_from_diff(g.root.state, state)
            if moves is None:
                rebuild = True
            else:
                for a in moves:
                    g.advance_root(int(a))
        if rebuild:
            g = _build_fresh(state, rng)
            state_holder["game"] = g
        run_batched_mcts(
            [g], evaluator, n_simulations=n_simulations, add_root_noise=False
        )
        chosen: int | None = None
        if proven_prop:
            if (
                g.root.proven_value == +1
                and g.root.proven_action >= 0
                and g.root.legal_mask[g.root.proven_action]
            ):
                chosen = int(g.root.proven_action)
            else:
                for a, child in g.root.children.items():
                    if child.proven_value == -1 and g.root.legal_mask[a]:
                        chosen = int(a)
                        break
        if chosen is None:
            pi = policy_from_visits(g.root, temperature=0.0)
            chosen = int(np.argmax(pi))
        # Advance through our own move so the next call (after opponent reply)
        # only needs to advance one more ply.
        g.advance_root(chosen)
        return chosen

    # Expose the holder for tests (state-invariant + reset-on-new-match) so
    # they can introspect the persistent MCTSGame without re-entering pick().
    pick._reuse_state = state_holder  # type: ignore[attr-defined]

    wrapped = vcf_overlay_picker(
        pick, max_nodes=eval_vcf_nodes, max_depth=eval_vcf_depth
    )
    wrapped = vct_finish_picker(wrapped, budget=vct_finish_nodes)
    # Propagate the holder onto whichever picker we hand back (the wrapper at
    # eval_vcf_nodes>0, or `pick` itself when the overlay is OFF and the wrapper
    # is `pick` unchanged) so callers always see ._reuse_state.
    if wrapped is not pick:
        wrapped._reuse_state = state_holder  # type: ignore[attr-defined]
    return wrapped


def _match_result_from_outcomes(
    outcomes: list[tuple[bool, str]], n_games: int
) -> MatchResult:
    """Aggregate per-game (a_is_black, outcome) pairs into a MatchResult.

    `outcome` is "win"/"loss"/"draw" from the model's (player A's) perspective;
    `a_is_black` is True when the model played black that game. Produces both the
    aggregate W/L/D and the per-color split, with the invariant that
    black_<x> + white_<x> == aggregate_<x> for each of win/loss/draw.
    """
    wins = losses = draws = 0
    black_w = black_l = black_d = 0
    white_w = white_l = white_d = 0
    for a_is_black, outcome in outcomes:
        if outcome == "win":
            wins += 1
            if a_is_black:
                black_w += 1
            else:
                white_w += 1
        elif outcome == "loss":
            losses += 1
            if a_is_black:
                black_l += 1
            else:
                white_l += 1
        elif outcome == "draw":
            draws += 1
            if a_is_black:
                black_d += 1
            else:
                white_d += 1
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown outcome {outcome!r}")
    return MatchResult(
        n_games=n_games,
        wins=wins,
        losses=losses,
        draws=draws,
        black_w=black_w,
        black_l=black_l,
        black_d=black_d,
        white_w=white_w,
        white_l=white_l,
        white_d=white_d,
    )


def play_match_pickers(
    picker_a: Picker,
    picker_b: Picker,
    *,
    n_games: int,
    seed: int = 0,
    random_opening_moves: int = 0,
    start_state: GameState | None = None,
) -> MatchResult:
    """Play A vs B, alternating colors. Returns result from A's perspective.

    Draws count as half-wins in `win_rate`. The result also carries a per-color
    split (see `MatchResult`) so analyses can see whether losses cluster on the
    color the model played.

    When ``random_opening_moves > 0`` each game pair shares the SAME random
    opening position: game pair (2k, 2k+1) both start from an identical K-stone
    opening drawn from a uniform-random playout, but swap which picker plays
    black vs white.  This mirrors the swap-2 convention's goal of neutralising
    first-mover advantage: neither player benefits from the opening stone
    placement because each side plays both colors from the same starting board.

    When ``start_state`` is given EVERY game starts from that exact position
    (with the same color-swap-per-pair seat alternation). This is the fixed-
    opening eval path — e.g. the swap2 idx-2 board the rulers use, where both
    players take the side-to-move seat across a pair so neither benefits from
    the (fixed) opening. ``start_state`` takes precedence over
    ``random_opening_moves``; a fixed opening does not compose with a random one.

    If ``n_games`` is odd the final game gets its own opening (no partner for
    the color swap), so keep ``n_games`` even for balanced evaluation.
    """
    from gomoku.self_play import _random_opening_state  # local to avoid heavy top-level dep

    rng = np.random.default_rng(seed)
    outcomes: list[tuple[bool, str]] = []
    # Pre-generate one opening state per game-pair so both games in a pair
    # share the same starting position before the pickers take over.
    opening_states: list[GameState] = []
    if random_opening_moves > 0:
        for pair_idx in range((n_games + 1) // 2):
            opening_state, _ = _random_opening_state(rng, random_opening_moves)
            opening_states.append(opening_state)

    for g_idx in range(n_games):
        a_is_black = (g_idx % 2 == 0)
        a_to_move = a_is_black
        # A fixed start position (e.g. the swap2 idx-2 board) wins over a random
        # opening: same seat-swap per pair, but the board is identical every game.
        if start_state is not None:
            state = start_state
            ply = state.move_count
            if ply % 2 == 1:
                # Odd-ply opening → white to move; flip from the base assignment.
                a_to_move = not a_is_black
        # Use the shared opening for both games in each color-swap pair.
        elif random_opening_moves > 0:
            pair_idx = g_idx // 2
            state = opening_states[pair_idx]
            # After a random opening, side-to-move is encoded in state.board[0].
            # The ply counter drives picker assignment; seed it from the opening
            # ply count so win/loss accounting stays correct regardless of how
            # many moves were pre-played.
            ply = state.move_count
            # After an even-ply opening black is still to move; after an
            # odd-ply opening white is to move.  Adjust a_to_move accordingly.
            if ply % 2 == 1:
                # White to move after the opening; flip from the base assignment.
                a_to_move = not a_is_black
        else:
            state = GameState.initial()
            ply = 0

        winner_side: int | None = None
        while True:
            picker = picker_a if a_to_move else picker_b
            action = picker(state, rng)
            side_just_moved = ply % 2
            state = state.apply(action)
            done, v = state.is_terminal()
            if done:
                if v == -1.0:
                    winner_side = side_just_moved
                break
            a_to_move = not a_to_move
            ply += 1

        if winner_side is None:
            outcomes.append((a_is_black, "draw"))
        else:
            a_side = 0 if a_is_black else 1
            outcomes.append((a_is_black, "win" if winner_side == a_side else "loss"))

    return _match_result_from_outcomes(outcomes, n_games)


# ---------------- Multi-process parallel match ----------------
#
# eval_worker plays n=20 games per baseline sequentially on one CPU core.
# When CPU is otherwise idle (trainer + self-play workers are on MPS), we
# can run those games in parallel via multiprocessing.Pool to cut the eval
# pass time by ~n_workers.
#
# The aggregate W/L/D counts are unbiased — each game's RNG is independent
# (seeded from game index + base seed) — but not byte-for-byte identical to
# the sequential path because the sequential version threads ONE shared RNG
# across all games, while the parallel version gives each game its own seed.

# Worker-process globals, populated by `_pool_init`.
_WORKER_MODEL_PICKER: Picker | None = None
_WORKER_OPP_PICKER: Picker | None = None


def _pool_init(checkpoint_path: str | None, sims: int, c_puct: float,
               device: str, opp_spec_str: str,
               eval_vcf_nodes: int = 0, eval_vcf_depth: int = 0,
               fpu_reduction_c: float = 0.0,
               reuse_tree: bool = False,
               proven_prop: bool = False,
               proven_vcf_leaf_nodes: int = 0) -> None:
    """Run once in each worker on Pool startup. Loads the model + opp picker.

    Each worker holds its own copy of the model (forked-then-loaded, so the
    parent's model isn't copied — workers re-load from disk). Acceptable
    overhead at eval-cycle granularity since the pool only spins up once
    per cycle per baseline.

    ``eval_vcf_nodes`` (default 0 = OFF, byte-identical) threads the eval-time
    root VCF overlay through to the model picker in each worker.

    ``fpu_reduction_c`` (default 0.0 = OFF, byte-identical) threads the eval-
    time FPU reduction through to the model picker in each worker.

    ``reuse_tree`` (default False = OFF, byte-identical) threads the eval-time
    cross-ply tree-reuse lever through to the model picker in each worker.
    Each worker reuses its own picker across games in the pool — the
    new-match reset rule (move_count==0) rebuilds between games so cross-game
    contamination is impossible.
    """
    global _WORKER_MODEL_PICKER, _WORKER_OPP_PICKER

    # Lazy imports so the worker process doesn't have to ship pytorch around
    # in the fork.
    from gomoku.match import build_player, parse_spec
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint

    if checkpoint_path:
        model, _ = load_checkpoint(checkpoint_path, device=device)
        model = fuse_model_for_inference(model)
        evaluator = make_torch_evaluator(model, device)
        _WORKER_MODEL_PICKER = mcts_picker(
            evaluator, n_simulations=sims, c_puct=c_puct,
            eval_vcf_nodes=eval_vcf_nodes, eval_vcf_depth=eval_vcf_depth,
            fpu_reduction_c=fpu_reduction_c,
            reuse_tree=reuse_tree,
            proven_prop=proven_prop,
            proven_vcf_leaf_nodes=proven_vcf_leaf_nodes,
        )
    _WORKER_OPP_PICKER = build_player(parse_spec(opp_spec_str))


def _pool_play_one(args: tuple[int, int]) -> str:
    """Play one game in a worker. Returns 'win'/'loss'/'draw' from the model's
    perspective. The model is `_WORKER_MODEL_PICKER`, opponent is
    `_WORKER_OPP_PICKER` (both set by `_pool_init`)."""
    g_idx, game_seed = args
    assert _WORKER_MODEL_PICKER is not None and _WORKER_OPP_PICKER is not None
    rng = np.random.default_rng(game_seed)
    a_is_black = (g_idx % 2 == 0)
    a_to_move = a_is_black
    state = GameState.initial()
    winner_side: int | None = None
    ply = 0
    while True:
        picker = _WORKER_MODEL_PICKER if a_to_move else _WORKER_OPP_PICKER
        action = picker(state, rng)
        side_just_moved = ply % 2
        state = state.apply(action)
        done, v = state.is_terminal()
        if done:
            if v == -1.0:
                winner_side = side_just_moved
            break
        a_to_move = not a_to_move
        ply += 1
    if winner_side is None:
        return "draw"
    a_side = 0 if a_is_black else 1
    return "win" if winner_side == a_side else "loss"


def play_match_parallel(
    *,
    checkpoint_path: str | None,
    opp_spec: str,
    n_games: int,
    seed: int = 0,
    n_workers: int = 4,
    sims: int = 100,
    c_puct: float = 1.5,
    device: str = "cpu",
    eval_vcf_nodes: int = 0,
    eval_vcf_depth: int = 0,
    fpu_reduction_c: float = 0.0,
    reuse_tree: bool = False,
    proven_prop: bool = False,
    proven_vcf_leaf_nodes: int = 0,
) -> MatchResult:
    """Play `n_games` of the model (loaded from `checkpoint_path`) vs the
    baseline described by `opp_spec`, in parallel via multiprocessing.Pool.

    Result from the model's perspective (wins = model won).

    Aggregate W/L/D counts are equivalent to the sequential `play_match_pickers`
    in expectation, but per-game RNG is independent per worker rather than
    shared across the sequence. That's a correctness-preserving change (each
    game is independent anyway), not a bug.

    ``eval_vcf_nodes`` (default 0 = OFF, byte-identical) threads the eval-time
    root VCF overlay through into each worker's model picker.

    ``fpu_reduction_c`` (default 0.0 = OFF, byte-identical) threads the eval-
    time FPU reduction through into each worker's model picker.

    ``reuse_tree`` (default False = OFF, byte-identical) threads the eval-time
    cross-ply tree-reuse lever through into each worker's model picker.
    """
    import multiprocessing as mp

    if n_workers < 2:
        raise ValueError(f"play_match_parallel needs n_workers >= 2, got {n_workers}")
    init_args = (checkpoint_path, sims, c_puct, device, opp_spec,
                 eval_vcf_nodes, eval_vcf_depth, fpu_reduction_c,
                 reuse_tree, proven_prop, proven_vcf_leaf_nodes)
    game_args = [(g_idx, seed + g_idx + 1) for g_idx in range(n_games)]
    # spawn (not fork) on macOS by default; spawn re-imports the module in
    # each worker, which is what we want for clean state.
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_pool_init, initargs=init_args) as pool:
        outcomes = pool.map(_pool_play_one, game_args)
    # Re-derive each game's model color from its index (same rule the worker
    # used: a_is_black = g_idx % 2 == 0). pool.map preserves input order, so
    # outcomes[i] corresponds to game_args[i].
    color_outcomes = [
        ((g_idx % 2 == 0), outcome)
        for (g_idx, _seed), outcome in zip(game_args, outcomes)
    ]
    return _match_result_from_outcomes(color_outcomes, n_games)


def play_vs_random(
    evaluator: Evaluator,
    *,
    n_games: int,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    seed: int = 0,
) -> MatchResult:
    """Play `n_games` against a random opponent, alternating colors.

    Thin wrapper over `play_match_pickers` — kept for backward compatibility
    with the training loop.
    """
    model_picker = mcts_picker(evaluator, n_simulations=n_simulations, c_puct=c_puct)
    return play_match_pickers(model_picker, _random_pick, n_games=n_games, seed=seed)


def play_match(
    eval_a: Evaluator,
    eval_b: Evaluator,
    *,
    n_games: int,
    n_simulations: int = 50,
    c_puct: float = 1.5,
    seed: int = 0,
) -> MatchResult:
    """Play A vs B (both MCTS-driven), alternating colors.

    Thin wrapper over `play_match_pickers` — kept for backward compatibility.
    """
    pa = mcts_picker(eval_a, n_simulations=n_simulations, c_puct=c_puct)
    pb = mcts_picker(eval_b, n_simulations=n_simulations, c_puct=c_puct)
    return play_match_pickers(pa, pb, n_games=n_games, seed=seed)
