"""PUCT MCTS for gomoku, with cross-game leaf batching for GPU saturation.

Design: each game gets its own MCTS tree. Per simulation round, every active
game does a select-down-to-leaf step, all leaves are batched into one network
forward pass, then we backprop in parallel. This keeps the MPS batch size large
even though each game's tree is independent.

State perspective: canonical states have plane 0 = side-to-move. Network values
are from side-to-move's perspective in the LEAF state, so when we backprop we
flip the sign at every parent (their opponent's value is the negation of ours).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from gomoku import state_ops
from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES, GameState


class Evaluator(Protocol):
    """Batch leaf evaluator.

    Takes a list of canonical states; returns (priors, values).
      priors: (B, N_ACTIONS) float32, raw (unmasked, unnormalized over illegal moves OK)
      values: (B,) float32 in [-1, 1] from side-to-move's perspective
    """

    def __call__(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class Node:
    state: GameState
    parent: "Node | None" = None
    parent_action: int = -1
    # Per-child stats indexed by action (0..80). We use dense arrays of size 81 even
    # though most actions are illegal — the legal mask zeroes them out in selection.
    N: np.ndarray = field(default_factory=lambda: np.zeros(N_ACTIONS, dtype=np.int32))
    W: np.ndarray = field(default_factory=lambda: np.zeros(N_ACTIONS, dtype=np.float32))
    P: np.ndarray = field(default_factory=lambda: np.zeros(N_ACTIONS, dtype=np.float32))
    children: dict[int, "Node"] = field(default_factory=dict)
    legal_mask: np.ndarray = field(default_factory=lambda: np.zeros(N_ACTIONS, dtype=bool))
    expanded: bool = False
    # Cached terminal info on first visit
    is_terminal: bool = False
    terminal_value: float = 0.0  # from side-to-move's perspective in THIS node's state

    def total_visits(self) -> int:
        return int(self.N.sum())


def _select_action(node: Node, c_puct_init: float, c_puct_base: float) -> int:
    """PUCT selection over legal actions only.

    AlphaGo Zero log-schedule:

        pb_c = log((1 + N_parent + c_puct_base) / c_puct_base) + c_puct_init

    pb_c replaces the constant c_puct in the standard PUCT formula. At small
    N_parent it equals c_puct_init; as the parent accumulates visits it grows
    slowly, tilting toward more exploration. With c_puct_base=19652 it's nearly
    constant for the typical sims-per-move budget but matches the AGZ recipe.
    """
    total = node.total_visits()
    pb_c = float(np.log((1.0 + total + c_puct_base) / c_puct_base) + c_puct_init)
    sqrt_total = np.sqrt(total + 1e-8)
    Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
    U = pb_c * node.P * sqrt_total / (1.0 + node.N)
    score = Q + U
    # Mask illegal moves to -inf so they're never selected.
    score = np.where(node.legal_mask, score, -np.inf)
    return int(np.argmax(score))


def _set_priors(node: Node, raw_priors: np.ndarray) -> None:
    """Softmax over legal actions only, store in node.P."""
    masked = np.where(node.legal_mask, raw_priors, -1e9)
    masked = masked - masked.max()
    exp = np.exp(masked) * node.legal_mask
    s = exp.sum()
    node.P = (exp / s).astype(np.float32) if s > 0 else node.legal_mask.astype(np.float32) / max(node.legal_mask.sum(), 1)


def _add_dirichlet_noise(node: Node, alpha: float, eps: float, rng: np.random.Generator) -> None:
    legal_idx = np.flatnonzero(node.legal_mask)
    if len(legal_idx) == 0:
        return
    noise = rng.dirichlet([alpha] * len(legal_idx))
    p = node.P.copy()
    p[legal_idx] = (1 - eps) * p[legal_idx] + eps * noise
    node.P = p.astype(np.float32)


def _init_node(node: Node) -> None:
    """Compute terminal info and legal mask for a fresh node."""
    done, v, legal_mask = state_ops.init_node_status(
        node.state.board,
        node.state.move_count,
    )
    node.is_terminal = done
    node.terminal_value = v
    if legal_mask is not None:
        node.legal_mask = legal_mask


@dataclass
class _PendingLeaf:
    """A leaf reached by selection, awaiting evaluation."""

    leaf: Node
    path: list[tuple[Node, int]]  # (parent, action_from_parent) for backprop


def _select_one(root: Node, c_puct_init: float, c_puct_base: float) -> _PendingLeaf:
    """Descend tree until we reach either an unexpanded node or a terminal."""
    node = root
    path: list[tuple[Node, int]] = []
    while True:
        if node.is_terminal:
            return _PendingLeaf(leaf=node, path=path)
        if not node.expanded:
            return _PendingLeaf(leaf=node, path=path)
        a = _select_action(node, c_puct_init, c_puct_base)
        if a not in node.children:
            # Lazy child creation
            child_state = node.state.apply(a)
            child = Node(state=child_state, parent=node, parent_action=a)
            _init_node(child)
            node.children[a] = child
        path.append((node, a))
        node = node.children[a]


def _backprop(path: list[tuple[Node, int]], leaf_value: float) -> None:
    """Backpropagate leaf_value, flipping sign at every level.

    leaf_value is from the LEAF state's side-to-move perspective. The parent
    along the path has the opposite side-to-move (because state.apply() flips
    planes), so its value contribution is -leaf_value, etc.
    """
    v = leaf_value
    for parent, action in reversed(path):
        # In `parent`, we're recording the value of taking `action`.
        # The child's value is +v from the child's perspective, which is -v from parent's.
        v = -v
        parent.N[action] += 1
        parent.W[action] += v


class MCTSGame:
    """Per-game MCTS state. Reused across multiple `search()` calls as the game progresses.

    The `c_puct` parameter is the c_puct_init term of the AGZ log schedule (see
    `_select_action`). `c_puct_base` controls how fast pb_c grows with parent
    visits; the AGZ default 19652 makes it nearly constant for our sim budgets.
    """

    def __init__(
        self,
        state: GameState,
        *,
        c_puct: float = 1.25,
        c_puct_base: float = 19652.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        rng: np.random.Generator | None = None,
    ):
        self.c_puct = c_puct  # acts as c_puct_init in the AGZ log schedule
        self.c_puct_base = c_puct_base
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = rng or np.random.default_rng()
        self.root = Node(state=state)
        _init_node(self.root)

    def advance_root(self, action: int) -> None:
        """Move root to child after action is played (reuses subtree)."""
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
            self.root.parent_action = -1
        else:
            child_state = self.root.state.apply(action)
            self.root = Node(state=child_state)
            _init_node(self.root)


def run_batched_mcts(
    games: list[MCTSGame],
    evaluator: Evaluator,
    *,
    n_simulations: int,
    add_root_noise: bool = True,
) -> None:
    """Run n_simulations rounds of MCTS across all games, batching leaf evals.

    On the first simulation, also expand the root and (optionally) add Dirichlet
    noise — this is the standard AlphaZero exploration prior at the root.
    """
    # Step 1: expand any unexpanded roots in one batched eval, with optional Dirichlet noise.
    unexpanded = [g for g in games if not g.root.expanded and not g.root.is_terminal]
    if unexpanded:
        states = [g.root.state for g in unexpanded]
        priors, values = evaluator(states)
        for g, p in zip(unexpanded, priors):
            _set_priors(g.root, p)
            g.root.expanded = True
            if add_root_noise:
                _add_dirichlet_noise(g.root, g.dirichlet_alpha, g.dirichlet_eps, g.rng)
        # We do NOT backprop the root's own value; the root is not a leaf in the AlphaZero sense.

    for _ in range(n_simulations):
        pending = [_select_one(g.root, g.c_puct, g.c_puct_base) for g in games]

        # Split into terminal (no eval needed) and to-evaluate.
        to_eval: list[_PendingLeaf] = []
        for p in pending:
            if p.leaf.is_terminal:
                _backprop(p.path, p.leaf.terminal_value)
            else:
                to_eval.append(p)

        if not to_eval:
            continue

        states = [p.leaf.state for p in to_eval]
        priors, values = evaluator(states)
        for p, prior, value in zip(to_eval, priors, values):
            if not p.leaf.expanded:
                _set_priors(p.leaf, prior)
                p.leaf.expanded = True
            _backprop(p.path, float(value))


# ---------------- Wave-batched MCTS (virtual loss) ----------------
#
# Inspired by mk5-main/forge/zeb/{mcts.py,batched_mcts.py}. Instead of evaluating
# one leaf per game per sim, we collect a "wave" of K leaves per game with virtual
# loss applied along each path, then batch-evaluate all G*K leaves in a single
# evaluator call. The total number of evaluator calls per move drops by ~K, which
# is the big win on MPS (where per-call sync dominates).
#
# Trade-off: within a wave, the W stats are stale (only N is incremented via
# virtual loss until the eval comes back). Larger waves => more staleness =>
# slightly noisier MCTS targets. Empirically wave_size 8–32 is a sweet spot at
# n_simulations=100.


def _select_one_vloss(root: Node, c_puct_init: float, c_puct_base: float) -> _PendingLeaf:
    """Like _select_one but applies virtual loss (N += 1) along the path.

    Soft virtual loss: we only increment N, not W. Within a wave, that drops
    Q = W/N (toward 0 from above) and U via the (1+N) denominator, which
    discourages reselecting the same action. AGZ-style strong vloss would also
    do `W -= 1` to push Q toward -1 more aggressively; we don't, partly because
    our wave-size-16/32 benchmarks already match wave=1 in outcomes, partly
    because soft vloss is forgiving when terminal leaves cause the path to be
    re-credited without a paired eval.

    Caller MUST backprop with `_backprop_value_only` (W only) since N has
    already been incremented here.
    """
    node = root
    path: list[tuple[Node, int]] = []
    while True:
        if node.is_terminal:
            return _PendingLeaf(leaf=node, path=path)
        if not node.expanded:
            return _PendingLeaf(leaf=node, path=path)
        a = _select_action(node, c_puct_init, c_puct_base)
        if a not in node.children:
            child_state = node.state.apply(a)
            child = Node(state=child_state, parent=node, parent_action=a)
            _init_node(child)
            node.children[a] = child
        path.append((node, a))
        node.N[a] += 1  # virtual loss — keep subsequent wave picks off this path
        node = node.children[a]


def _backprop_value_only(path: list[tuple[Node, int]], leaf_value: float) -> None:
    """Update W along path; N already incremented by virtual loss in selection."""
    v = leaf_value
    for parent, action in reversed(path):
        v = -v
        parent.W[action] += v


def _bfs_descend_one_per_game(
    games_subset: list[MCTSGame],
) -> list[_PendingLeaf]:
    """BFS-vectorized descent: one descent per game in `games_subset`,
    all advanced in lockstep level-by-level with vectorized PUCT.

    At each BFS level we collect every still-descending game's current node,
    stack their (W, N, P, legal_mask) into (P, N_ACTIONS) arrays, and do ONE
    batched argmax instead of P individual `_select_action` calls. Per-game
    soft virtual loss (N += 1) is applied as we descend, matching the
    sequential `_select_one_vloss` behavior.

    Trees are independent across games, so this is byte-for-byte equivalent
    to calling `_select_one_vloss` for each game serially under any RNG.

    Returns one `_PendingLeaf` per input game, in the same order.
    """
    if not games_subset:
        return []

    # Per-descent state, indexed by position in games_subset.
    nodes: list[Node] = [g.root for g in games_subset]
    paths: list[list[tuple[Node, int]]] = [[] for _ in games_subset]
    leaves: list[_PendingLeaf | None] = [None] * len(games_subset)
    max_rows = len(games_subset)
    N_stack = np.empty((max_rows, N_ACTIONS), dtype=np.int32)
    W_stack = np.empty((max_rows, N_ACTIONS), dtype=np.float32)
    P_stack = np.empty((max_rows, N_ACTIONS), dtype=np.float32)
    L_stack = np.empty((max_rows, N_ACTIONS), dtype=bool)
    c_puct_init = np.empty(max_rows, dtype=np.float64)
    c_puct_base = np.empty(max_rows, dtype=np.float64)

    # Active indices into games_subset.
    active = list(range(len(games_subset)))

    while active:
        # Partition: terminal / unexpanded → leaf; otherwise → still descending.
        still: list[int] = []
        for i in active:
            node = nodes[i]
            if node.is_terminal or not node.expanded:
                leaves[i] = _PendingLeaf(leaf=node, path=paths[i])
            else:
                still.append(i)
        if not still:
            break

        # Copy current-node action arrays into reusable scratch buffers. This
        # preserves the vectorized PUCT math while avoiding four fresh
        # np.stack allocations at every BFS level.
        n_still = len(still)
        for row, i in enumerate(still):
            node = nodes[i]
            N_stack[row] = node.N
            W_stack[row] = node.W
            P_stack[row] = node.P
            L_stack[row] = node.legal_mask
            game = games_subset[i]
            c_puct_init[row] = game.c_puct
            c_puct_base[row] = game.c_puct_base

        N_rows = N_stack[:n_still]                                     # (P, 81) int32
        W_rows = W_stack[:n_still]                                     # (P, 81) float32
        P_rows = P_stack[:n_still]                                     # (P, 81) float32
        L_rows = L_stack[:n_still]                                     # (P, 81) bool
        cpi = c_puct_init[:n_still]                                    # (P,)
        cpb = c_puct_base[:n_still]                                    # (P,)

        # Vectorized PUCT scoring (one np.argmax across all still-going descents).
        totals = N_rows.sum(axis=1, dtype=np.float64)                 # (P,)
        pb_c = np.log((1.0 + totals + cpb) / cpb) + cpi               # (P,)
        sqrt_totals = np.sqrt(totals + 1e-8)                          # (P,)
        denom = 1.0 + N_rows                                          # (P, 81)
        Q = np.where(N_rows > 0, W_rows / np.maximum(N_rows, 1), 0.0)
        U = (pb_c * sqrt_totals)[:, None] * P_rows / denom            # (P, 81)
        score = np.where(L_rows, Q + U, -np.inf)
        actions = np.argmax(score, axis=1)                            # (P,) int

        # Descend each still-going descent by one level, applying soft vloss.
        next_active: list[int] = []
        for j, i in enumerate(still):
            a = int(actions[j])
            node = nodes[i]
            if a not in node.children:
                child_state = node.state.apply(a)
                child = Node(state=child_state, parent=node, parent_action=a)
                _init_node(child)
                node.children[a] = child
            paths[i].append((node, a))
            node.N[a] += 1  # virtual loss — same as _select_one_vloss
            nodes[i] = node.children[a]
            next_active.append(i)
        active = next_active

    # Anything still in `active` after the loop has been written into `leaves`.
    # Anything finished above is already in `leaves`. Just collect.
    out: list[_PendingLeaf] = []
    for i in range(len(games_subset)):
        if leaves[i] is not None:
            out.append(leaves[i])
        else:
            # Shouldn't happen (every descent terminates at a leaf or terminal).
            out.append(_PendingLeaf(leaf=nodes[i], path=paths[i]))
    return out


def run_batched_mcts_waves(
    games: list[MCTSGame],
    evaluator: Evaluator,
    *,
    n_simulations: int,
    wave_size: int = 16,
    add_root_noise: bool = True,
) -> None:
    """Wave-batched MCTS: per round, collect `wave_size` leaves per game with
    virtual loss, then one batched evaluator call for all G*wave_size leaves.

    Equivalent to `run_batched_mcts` when wave_size == 1.

    Implementation note: within a wave we iterate wave-slots sequentially (slot
    0 of all games, then slot 1, …) so each slot's descent in a given game sees
    the cumulative virtual loss from prior slots — this matches the original
    sequential ordering. Within each slot we BFS-descend one descent per game
    in lockstep, vectorizing PUCT across games (see `_bfs_descend_one_per_game`).
    """
    if wave_size < 1:
        raise ValueError(f"wave_size must be >= 1, got {wave_size}")

    # 1. Expand unexpanded roots (one shared batched call).
    unexpanded = [g for g in games if not g.root.expanded and not g.root.is_terminal]
    if unexpanded:
        states = [g.root.state for g in unexpanded]
        priors, _ = evaluator(states)
        for g, p in zip(unexpanded, priors):
            _set_priors(g.root, p)
            g.root.expanded = True
            if add_root_noise:
                _add_dirichlet_noise(g.root, g.dirichlet_alpha, g.dirichlet_eps, g.rng)

    sims_done = [0] * len(games)
    while True:
        # How many sims to do this round, per game (cap at wave_size).
        wave_counts = [min(wave_size, n_simulations - sims_done[i]) for i in range(len(games))]
        if not any(wave_counts):
            break

        # Iterate wave-slots sequentially. At each slot, BFS-descend one
        # descent per game (only games that still need this slot).
        max_slots = max(wave_counts)
        wave_pending: list[_PendingLeaf] = []
        for slot in range(max_slots):
            slot_games = [games[i] for i in range(len(games)) if slot < wave_counts[i]]
            if not slot_games:
                continue
            wave_pending.extend(_bfs_descend_one_per_game(slot_games))

        # Split terminal vs to-evaluate.
        to_eval: list[_PendingLeaf] = []
        for p in wave_pending:
            if p.leaf.is_terminal:
                _backprop_value_only(p.path, p.leaf.terminal_value)
            else:
                to_eval.append(p)

        # Single batched evaluator call across all games in this wave.
        if to_eval:
            states = [p.leaf.state for p in to_eval]
            priors, values = evaluator(states)
            for p, prior, value in zip(to_eval, priors, values):
                if not p.leaf.expanded:
                    _set_priors(p.leaf, prior)
                    p.leaf.expanded = True
                _backprop_value_only(p.path, float(value))

        for i, w in enumerate(wave_counts):
            sims_done[i] += w


def policy_from_visits(root: Node, temperature: float) -> np.ndarray:
    """Return MCTS visit-based policy over actions.

    temperature: 0 = greedy (one-hot on max visits), 1 = proportional to visits,
                 small values sharpen.
    """
    counts = root.N.astype(np.float64)
    if temperature <= 0:
        out = np.zeros_like(counts)
        # Break ties uniformly among the most-visited actions.
        m = counts.max()
        winners = np.flatnonzero(counts == m)
        out[winners] = 1.0 / len(winners)
        return out.astype(np.float32)
    if temperature == 1.0:
        s = counts.sum()
        return (counts / s).astype(np.float32) if s > 0 else (root.legal_mask.astype(np.float32) / max(root.legal_mask.sum(), 1))
    counts = counts ** (1.0 / temperature)
    s = counts.sum()
    return (counts / s).astype(np.float32) if s > 0 else (root.legal_mask.astype(np.float32) / max(root.legal_mask.sum(), 1))


def make_torch_evaluator(
    model,
    device,
    *,
    fp16: bool = False,
) -> Evaluator:
    """Wrap a GomokuNet for use as a leaf evaluator."""
    import torch

    def states_to_batch(states: list[GameState]) -> np.ndarray:
        x = np.empty(
            (len(states), N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
        for i, state in enumerate(states):
            x[i] = state.to_planes()
        return x

    @torch.no_grad()
    def evaluate_planes(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        was_training = model.training
        if was_training:
            model.eval()
        try:
            x = np.asarray(x, dtype=np.float32)
            t = torch.from_numpy(x).to(device)
            if fp16:
                t = t.half()
            logits, values = model(t)
            B = t.shape[0]
            # One device->host transfer is much cheaper than two on MPS:
            # each .cpu() forces a stream sync, and the data is tiny (~20KB)
            # so syncs dominate over bandwidth. Pack logits + values into a
            # single 1-D tensor, transfer once, then split.
            combo = torch.cat(
                [logits.reshape(B * N_ACTIONS).float(), values.reshape(B).float()]
            ).cpu().numpy()
            priors = combo[: B * N_ACTIONS].reshape(B, N_ACTIONS)
            vals = combo[B * N_ACTIONS:]
            # Sanitize non-finite outputs. NaN values propagated through
            # native MCTS backup leave all select_action scores NaN, and
            # the C extension's default best_action=0 then plays an
            # illegal move when (0,0) is occupied. Guards against archived
            # mid-game positions (no history) producing pathological model
            # outputs.
            if not np.all(np.isfinite(priors)):
                priors = np.nan_to_num(priors, nan=0.0, posinf=1e6, neginf=-1e6)
            if not np.all(np.isfinite(vals)):
                vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=-1.0)
            return priors, vals
        finally:
            if was_training:
                model.train()

    @torch.no_grad()
    def evaluate(states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        return evaluate_planes(states_to_batch(states))

    evaluate.evaluate_planes = evaluate_planes  # type: ignore[attr-defined]
    return evaluate


def make_random_evaluator(seed: int = 0) -> Evaluator:
    """Uniform priors, zero values. Useful for testing the MCTS plumbing."""
    rng = np.random.default_rng(seed)

    def evaluate(states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        n = len(states)
        priors = np.zeros((n, N_ACTIONS), dtype=np.float32)
        for i, s in enumerate(states):
            legal = s.legal_mask()
            if legal.any():
                priors[i, legal] = 1.0 / legal.sum()
        values = np.zeros(n, dtype=np.float32)
        return priors, values

    # rng kept for future stochastic variants
    _ = rng
    return evaluate
