"""Non-learned opponents: random, heuristic, and shallow alpha-beta.

A "player" is `Callable[[GameState, np.random.Generator], int]` — given a state
and an RNG (for tie-breaking only, so matches stay reproducible), it returns the
action to play. This is deliberately decoupled from `gomoku.mcts.Evaluator` so
that pure heuristic opponents don't drag in torch or batched leaf evaluation.

Intentionally torch-free: keep these usable as fast reference opponents.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from gomoku.game import BOARD_SIZE, GameState, WIN_LEN

Player = Callable[[GameState, np.random.Generator], int]


# ----- Pre-computed length-5 windows -----
# Each row is a (WIN_LEN,) array of flat-board indices forming one line of 5.

def _build_windows() -> np.ndarray:
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    wins: list[list[int]] = []
    n = BOARD_SIZE
    for dr, dc in dirs:
        for r0 in range(n):
            for c0 in range(n):
                r_end = r0 + dr * (WIN_LEN - 1)
                c_end = c0 + dc * (WIN_LEN - 1)
                if not (0 <= r_end < n and 0 <= c_end < n):
                    continue
                wins.append([(r0 + dr * k) * n + (c0 + dc * k) for k in range(WIN_LEN)])
    return np.asarray(wins, dtype=np.int32)


_WINDOWS = _build_windows()  # (n_windows, 5)
_N_WINDOWS = _WINDOWS.shape[0]

# For each square, the list of window-indices that pass through it. Used to
# focus the per-move update so we don't rescore the whole board for each
# candidate move.
def _build_windows_through_cell() -> list[np.ndarray]:
    by_cell: list[list[int]] = [[] for _ in range(BOARD_SIZE * BOARD_SIZE)]
    for wi in range(_N_WINDOWS):
        for cell in _WINDOWS[wi]:
            by_cell[int(cell)].append(wi)
    return [np.asarray(lst, dtype=np.int32) for lst in by_cell]


_WINDOWS_THROUGH_CELL = _build_windows_through_cell()
# Max number of windows through a single cell (used for padding the dense layout).
_MAX_WIN_PER_CELL = max(arr.size for arr in _WINDOWS_THROUGH_CELL)


def _build_dense_windows_through_cell() -> tuple[np.ndarray, np.ndarray]:
    """Dense (81, K) array of window-indices through each cell, plus a (81, K)
    bool mask of valid entries. Padded with index 0 where unused (mask zero)."""
    n = BOARD_SIZE * BOARD_SIZE
    k = _MAX_WIN_PER_CELL
    dense = np.zeros((n, k), dtype=np.int32)
    mask = np.zeros((n, k), dtype=bool)
    for c in range(n):
        arr = _WINDOWS_THROUGH_CELL[c]
        dense[c, : arr.size] = arr
        mask[c, : arr.size] = True
    return dense, mask


_DENSE_WIN_BY_CELL, _DENSE_WIN_MASK = _build_dense_windows_through_cell()


# Weights for "my_stones in a window where opp has 0 stones".
# Index = number of my stones in window (0..5). 5 is essentially won.
_MY_W = np.array([0, 1, 25, 500, 10000, 10_000_000], dtype=np.float64)
# Defensive weights: how much we fear a window with N opp stones (and 0 of ours).
# A bit heavier than offense so we tend to block strong threats.
_OPP_W = np.array([0, 1, 40, 750, 15000, 10_000_000], dtype=np.float64) * 1.5


# ----- Players -----


def random_player(state: GameState, rng: np.random.Generator) -> int:
    legal = state.legal_actions()
    return int(rng.choice(legal))


def _flat_planes(state: GameState) -> tuple[np.ndarray, np.ndarray]:
    """Return (mine, opp) as flat (81,) bool arrays for side-to-move."""
    return state.board[0].reshape(-1), state.board[1].reshape(-1)


def _window_counts(mine: np.ndarray, opp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (my_stones, opp_stones) counts. Each returned array is (n_windows,)."""
    # _WINDOWS is (n_windows, 5); indexing flat plane gives (n_windows, 5) bool.
    mw = mine[_WINDOWS]
    ow = opp[_WINDOWS]
    return mw.sum(axis=1), ow.sum(axis=1)


def evaluate_position(state: GameState) -> float:
    """Static eval from side-to-move's perspective.

    Sum of "live for me" window contributions minus "live for opp" contributions.
    Positive favours the side to move.
    """
    mine, opp = _flat_planes(state)
    my_cnt, opp_cnt = _window_counts(mine, opp)
    my_live = (opp_cnt == 0)
    opp_live = (my_cnt == 0)
    score = float(_MY_W[my_cnt[my_live]].sum()) - float(_OPP_W[opp_cnt[opp_live]].sum())
    return score


def _has_immediate_win(mine_after: np.ndarray, opp: np.ndarray) -> bool:
    """Did placing the stone produce a 5-in-a-row for `mine`?"""
    # Any window where mine has 5 stones is a win regardless of opp.
    mw = mine_after[_WINDOWS].sum(axis=1)
    return bool((mw >= WIN_LEN).any())


def _find_immediate_wins(mine: np.ndarray, opp: np.ndarray, legal_idx: np.ndarray) -> np.ndarray:
    """Return subset of `legal_idx` that immediately complete 5-in-a-row for `mine`.

    Vectorized: for every cell, take the max `mine`-stones-after-placement across
    the windows through it (masking padded entries), then keep legal cells whose
    max reaches WIN_LEN. Replaces the per-legal-cell Python loop that was ~52% of
    deep-lookahead time (called once per leaf, scanning every empty square).
    """
    if legal_idx.size == 0:
        return np.asarray([], dtype=np.int32)
    mw = mine[_WINDOWS].sum(axis=1)                     # (n_windows,)
    cell_my_after = mw[_DENSE_WIN_BY_CELL] + 1          # (81, K): +1 for the new stone
    # Padded (invalid) window slots must not count — zero them before the max.
    best_after = np.where(_DENSE_WIN_MASK, cell_my_after, 0).max(axis=1)  # (81,)
    win_cells = best_after >= WIN_LEN                   # (81,) bool
    return legal_idx[win_cells[legal_idx]].astype(np.int32)


def _score_all_moves(mine: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Vectorized: return (81,) array of per-cell scores under the same heuristic
    as `_score_move`. Cells that aren't legal still get a score (caller masks)."""
    # Window-level counts (n_windows,)
    mw = mine[_WINDOWS].sum(axis=1)
    ow = opp[_WINDOWS].sum(axis=1)

    # For each cell c, gather (k,) counts of the windows through it.
    # Shape (81, K).
    cell_my = mw[_DENSE_WIN_BY_CELL]
    cell_opp = ow[_DENSE_WIN_BY_CELL]
    # The current cell is empty in the input planes (we're scoring placements
    # on empty cells). Stones in `mine` are pre-placement; placing adds 1.
    cell_my_after = cell_my + 1

    # Offensive: windows where opp has 0 stones AND the entry is valid.
    my_live = (cell_opp == 0) & _DENSE_WIN_MASK
    my_clipped = np.minimum(cell_my_after, WIN_LEN)
    # Where mask is False, contribution must be 0 — gather weights then zero-out.
    off_w = _MY_W[my_clipped]
    offense = np.where(my_live, off_w, 0.0).sum(axis=1)

    # Defensive: windows where I had 0 stones pre-placement and entry is valid.
    opp_threat = (cell_my == 0) & _DENSE_WIN_MASK
    opp_clipped = np.minimum(cell_opp, WIN_LEN)
    def_w = _OPP_W[opp_clipped]
    defense = np.where(opp_threat, def_w, 0.0).sum(axis=1)

    return offense + defense


def _score_cells(mine: np.ndarray, opp: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Per-cell heuristic score for just `cells`, aligned to `cells`.

    Identical per-cell values to `_score_all_moves` (each cell's score is
    independent of the others), but the dense (cell, window) gather is
    restricted to the candidate rows instead of all 81 — the move-ordering
    hot path inside the alpha-beta tree only needs the candidate scores.
    """
    mw = mine[_WINDOWS].sum(axis=1)
    ow = opp[_WINDOWS].sum(axis=1)
    dense = _DENSE_WIN_BY_CELL[cells]          # (n_cells, K)
    msk = _DENSE_WIN_MASK[cells]
    cell_my = mw[dense]
    cell_opp = ow[dense]

    my_live = (cell_opp == 0) & msk
    off_w = _MY_W[np.minimum(cell_my + 1, WIN_LEN)]
    offense = np.where(my_live, off_w, 0.0).sum(axis=1)

    opp_threat = (cell_my == 0) & msk
    def_w = _OPP_W[np.minimum(cell_opp, WIN_LEN)]
    defense = np.where(opp_threat, def_w, 0.0).sum(axis=1)

    return offense + defense


def _score_move(mine: np.ndarray, opp: np.ndarray, a: int) -> float:
    """Local score for playing `a`. Considers only windows through `a` plus a
    global defensive term for opp threats that don't pass through `a`.

    For speed we use a delta-style score: relative to "do nothing", the only
    windows that change are those passing through `a`. (Opp threats not passing
    through `a` don't change, so they cancel out in argmax across candidate `a`s
    and can be omitted.)
    """
    win_idxs = _WINDOWS_THROUGH_CELL[a]
    if win_idxs.size == 0:
        return 0.0
    w = _WINDOWS[win_idxs]  # (k, 5)
    my_cnt = mine[w].sum(axis=1) + 1  # after placement
    opp_cnt = opp[w].sum(axis=1)

    # Offensive contribution: windows still live for me after placing (opp == 0).
    my_live = (opp_cnt == 0)
    my_clipped = np.minimum(my_cnt[my_live], WIN_LEN)
    offense = float(_MY_W[my_clipped].sum())

    # Defensive contribution: windows opp could have completed (my pre-placement
    # was 0). Placing my stone there kills the threat — score how big the threat was.
    pre_my = my_cnt - 1
    opp_threat = (pre_my == 0)
    opp_clipped = np.minimum(opp_cnt[opp_threat], WIN_LEN)
    defense = float(_OPP_W[opp_clipped].sum())

    return offense + defense


def heuristic_player(state: GameState, rng: np.random.Generator) -> int:
    """Fast rule-based player.

    1) Take an immediate win if available.
    2) Else block one of the opponent's immediate winning moves.
    3) Else play argmax of pattern-based move score (rng ties).
    """
    mine, opp = _flat_planes(state)
    legal = state.legal_actions()
    all_scores = _score_all_moves(mine, opp)

    # 1) Immediate win for us?
    my_wins = _find_immediate_wins(mine, opp, legal)
    if my_wins.size > 0:
        return _argmax_with_tiebreak(my_wins, all_scores[my_wins], rng)

    # 2) Immediate win for opponent? Must block. (Opponent placing at `a` makes
    #    them complete 5; since blockable squares are exactly their winning moves,
    #    play one of those — preferring the one with the best secondary score for us.)
    opp_wins = _find_immediate_wins(opp, mine, legal)
    if opp_wins.size > 0:
        return _argmax_with_tiebreak(opp_wins, all_scores[opp_wins], rng)

    # 3) Argmax of score across all legal moves.
    return _argmax_with_tiebreak(legal, all_scores[legal], rng)


def defensive_player(state: GameState, rng: np.random.Generator) -> int:
    """Defense-only variant: blocks opponent threats; ignores own offense.

    Use as a self-play opponent to force the MODEL to learn how to break
    through resistance and convert its own threats — the defensive player
    never builds its own attack, so when it loses, it's because the model
    actually constructed a 5-in-a-row against constant blocking pressure.

    1) Block an immediate opp winning move if any.
    2) Else play the move that maximally neutralises the opp's strongest
       remaining threats (defense term only from `_score_all_moves`).
    """
    mine, opp = _flat_planes(state)
    legal = state.legal_actions()

    # 1) Immediate win for opponent? Must block.
    opp_wins = _find_immediate_wins(opp, mine, legal)
    if opp_wins.size > 0:
        # Tiebreak among blocking moves by which one cuts the most other threats.
        defense_scores = _defense_only_scores(mine, opp)
        return _argmax_with_tiebreak(opp_wins, defense_scores[opp_wins], rng)

    # 2) Argmax of defense-only score over legal moves.
    defense_scores = _defense_only_scores(mine, opp)
    return _argmax_with_tiebreak(legal, defense_scores[legal], rng)


def _max_own_window_after(mine: np.ndarray, _opp: np.ndarray) -> np.ndarray:
    """For each cell, returns the maximum # of my stones in any 5-window
    THROUGH that cell, AFTER placing one of my stones there.

    Used by `pacifist_blocker` to avoid moves that grow its own line to 4+.
    """
    mw = mine[_WINDOWS].sum(axis=1)
    cell_my = mw[_DENSE_WIN_BY_CELL]
    cell_my_after = (cell_my + 1)
    # Cells not in any valid window: pretend they're 0 so they don't dominate.
    cell_my_after = np.where(_DENSE_WIN_MASK, cell_my_after, 0)
    return cell_my_after.max(axis=1)


def pacifist_blocker(state: GameState, rng: np.random.Generator,
                     *, max_own_line: int = 3) -> int:
    """Like `defensive_player` but refuses to grow its own line beyond
    `max_own_line` (default 3 — never threatens an open four).

    The intent: force the MODEL to actually finish a 5-in-a-row against
    constant defensive pressure that won't accidentally win first. When
    `defensive_player` ends up forming its own 5 (50% of games against the
    e50 model), `pacifist_blocker` will reject those moves.

    Priority order:
    1) If opponent has an immediate 5 threat: must block (overriding the
       self-line cap is fine here; that move would have been forced anyway).
    2) Among legal moves whose placement keeps my longest window ≤ max_own_line,
       pick argmax defense.
    3) If no such move exists (board crowded with my stones), fall back to
       the legal move with the smallest own-line after, breaking ties by
       defense score.
    """
    mine, opp = _flat_planes(state)
    legal = state.legal_actions()

    opp_wins = _find_immediate_wins(opp, mine, legal)
    if opp_wins.size > 0:
        defense_scores = _defense_only_scores(mine, opp)
        return _argmax_with_tiebreak(opp_wins, defense_scores[opp_wins], rng)

    own_after = _max_own_window_after(mine, opp)
    defense_scores = _defense_only_scores(mine, opp)

    pacifist_mask = own_after[legal] <= max_own_line
    if pacifist_mask.any():
        cands = legal[pacifist_mask]
        return _argmax_with_tiebreak(cands, defense_scores[cands], rng)

    # No safe move — find the legal move with the smallest own-line, break
    # ties by defense.
    own_for_legal = own_after[legal]
    smallest = own_for_legal.min()
    candidates = legal[own_for_legal == smallest]
    return _argmax_with_tiebreak(candidates, defense_scores[candidates], rng)


def _defense_only_scores(mine: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Pure defense: per-cell score equals how much opp-threat the placement kills.
    Computed using the same window infrastructure as `_score_all_moves`, but
    drops the offensive term entirely.
    """
    mw = mine[_WINDOWS].sum(axis=1)
    ow = opp[_WINDOWS].sum(axis=1)
    cell_my = mw[_DENSE_WIN_BY_CELL]
    cell_opp = ow[_DENSE_WIN_BY_CELL]
    # Defensive: windows where I had 0 stones pre-placement and entry is valid.
    opp_threat = (cell_my == 0) & _DENSE_WIN_MASK
    opp_clipped = np.minimum(cell_opp, WIN_LEN)
    def_w = _OPP_W[opp_clipped]
    return np.where(opp_threat, def_w, 0.0).sum(axis=1)


def _argmax_with_tiebreak(actions: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> int:
    m = scores.max()
    winners = actions[scores == m]
    if winners.size == 1:
        return int(winners[0])
    return int(rng.choice(winners))


# ----- Neighbor-only move generation (huge alpha-beta speedup on sparse boards) -----

def _neighbor_offsets() -> np.ndarray:
    """All (dr, dc) within Chebyshev distance 2, excluding (0,0)."""
    out = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            out.append((dr, dc))
    return np.asarray(out, dtype=np.int32)


_NEIGHBORS = _neighbor_offsets()


def _build_neighbor_mask() -> np.ndarray:
    """(81, 81) bool: row i is True at every cell within Chebyshev-2 of cell i.

    Precomputed so the per-node candidate dilation is a single gather+reduce
    instead of a Python loop over the 24 neighbor offsets (which was ~29% of
    deep-lookahead time — the fancy-index + bounds-mask per offset dominated).
    """
    n = BOARD_SIZE * BOARD_SIZE
    mask = np.zeros((n, n), dtype=bool)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            i = r * BOARD_SIZE + c
            for dr, dc in _NEIGHBORS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                    mask[i, rr * BOARD_SIZE + cc] = True
    return mask


_NEIGHBOR_MASK = _build_neighbor_mask()


def _candidate_moves(state: GameState) -> np.ndarray:
    """Return legal moves restricted to neighbors of existing stones.

    Falls back to the full legal set when the board is empty (start of game).
    """
    occ_flat = (state.board[0] | state.board[1]).reshape(-1)
    occ_idx = np.flatnonzero(occ_flat)
    if occ_idx.size == 0:
        # Opening — center is fine. Return only e5 to keep the tree tiny on move 1.
        return np.asarray([(BOARD_SIZE // 2) * BOARD_SIZE + (BOARD_SIZE // 2)], dtype=np.int32)
    # Chebyshev-2 dilation as a single vectorized gather over the precomputed
    # neighbor mask: any occupied cell's neighborhood, intersected with empties.
    near = _NEIGHBOR_MASK[occ_idx].any(axis=0)
    cand = near & ~occ_flat
    if not cand.any():
        # Pathological: board is full of stones in occupied's neighborhood. Fallback.
        return state.legal_actions()
    return np.flatnonzero(cand).astype(np.int32)


# ----- Negamax with alpha-beta -----

_MATE = 1_000_000_000.0  # large sentinel, distinct from any heuristic eval


def _negamax(state: GameState, depth: int, alpha: float, beta: float) -> float:
    """Return value from side-to-move's perspective.

    `depth` is plies remaining. Higher returned value is better for side-to-move.
    """
    done, v = state.is_terminal()
    if done:
        # v == -1 means the player who JUST moved (i.e. our opponent) won.
        # From side-to-move's perspective, we lost. Use a mate-distance score so
        # shorter wins (and longer losses) are preferred.
        if v < 0:
            return -(_MATE + depth)  # losing — prefer to lose LATER (smaller magnitude w/ larger depth).
        return 0.0  # draw

    if depth == 0:
        # 1-ply quiescence: if our opponent (the side that just moved) has any
        # immediate winning threat (a 4-in-a-row completable to 5 next move),
        # we are forced to block one of those squares before evaluating.
        # Without this, the static `evaluate_position` over-credits the
        # just-moved side's "live 4" — there's no distinction between an
        # open four (actually mate) and a half-open four (trivially blocked).
        # That hallucinated-threat bias is what made odd-depth lookahead
        # weaker than even-depth in our calibration tournament: at odd depth
        # the searcher gets the last move and builds a threat the opponent
        # never gets to refute before the eval.
        mine, opp = _flat_planes(state)
        legal = state.legal_actions()
        opp_wins = _find_immediate_wins(opp, mine, legal)
        if opp_wins.size == 0:
            return evaluate_position(state)
        # Forced-reply: try blocking each of opp's winning squares and take
        # the best resulting evaluation. Bounded at ~5 moves in practice
        # since multi-way threats end games quickly.
        best_q = -np.inf
        for a in opp_wins:
            child = state.apply(int(a))
            done2, v2 = child.is_terminal()
            if done2 and v2 < 0:
                # Our block itself won (shouldn't happen given opp_wins is
                # opp's threats, but covers the corner case).
                return _MATE
            # evaluate_position returns from current-side perspective; after
            # we block, side flips to opp, so we negate to get back to ours.
            q = -evaluate_position(child)
            if q > best_q:
                best_q = q
        return best_q

    candidates = _candidate_moves(state)
    # Move ordering: try moves with higher static score first to maximise alpha-beta cuts.
    mine, opp = _flat_planes(state)
    move_scores = _score_cells(mine, opp, candidates)
    order = np.argsort(-move_scores)
    candidates = candidates[order]
    # Restrict the branching at internal nodes: only keep top-K. The full neighbor
    # set still gets considered at the ROOT (caller's loop); this affects deeper
    # plies only. K is chosen to stay tactically safe: 12 still covers all
    # plausible threat-responses on a 9x9 board.
    _MAX_BRANCH = 12
    if candidates.size > _MAX_BRANCH:
        candidates = candidates[:_MAX_BRANCH]

    best = -np.inf
    for a in candidates:
        child = state.apply(int(a))
        # Quick terminal check post-move for immediate wins (saves a recursive call).
        done2, v2 = child.is_terminal()
        if done2 and v2 < 0:
            # We just won — return mate score immediately, preferring shorter mates.
            return _MATE + depth
        val = -_negamax(child, depth - 1, -beta, -alpha)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def lookahead_player(depth: int = 4) -> Player:
    """Return a player that runs negamax+alpha-beta to `depth` plies."""

    def play(state: GameState, rng: np.random.Generator) -> int:
        mine, opp = _flat_planes(state)
        legal = state.legal_actions()
        all_scores = _score_all_moves(mine, opp)

        # Step 1: shortcut for immediate wins (cheap and avoids depth=0 noise).
        my_wins = _find_immediate_wins(mine, opp, legal)
        if my_wins.size > 0:
            return _argmax_with_tiebreak(my_wins, all_scores[my_wins], rng)

        candidates = _candidate_moves(state)
        # Order by static score for better pruning.
        m_scores = all_scores[candidates]
        order = np.argsort(-m_scores)
        candidates = candidates[order]

        best_val = -np.inf
        best_actions: list[int] = []
        alpha = -np.inf
        beta = np.inf
        for a in candidates:
            child = state.apply(int(a))
            done2, v2 = child.is_terminal()
            if done2 and v2 < 0:
                # Immediate win (should have been caught by my_wins; defensive).
                val = _MATE + depth
            else:
                val = -_negamax(child, depth - 1, -beta, -alpha)
            if val > best_val:
                best_val = val
                best_actions = [int(a)]
            elif val == best_val:
                best_actions.append(int(a))
            if best_val > alpha:
                alpha = best_val
        if len(best_actions) == 1:
            return best_actions[0]
        return int(rng.choice(np.asarray(best_actions)))

    return play
