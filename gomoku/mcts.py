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
    # KataGo-style proven-value bookkeeping (derby-b3n). From THIS node's
    # side-to-move perspective: +1 = proven win, -1 = proven loss, 0 = unknown.
    # Set at terminal-init (where terminal_value -1.0 ≡ side-to-move just lost),
    # optionally at leaf expansion via solve_vcf, and propagated upward in
    # backup. EVAL-ONLY: gen MCTS leaves it at 0 (it stays unused).
    proven_value: int = 0
    # When proven_value == +1 from a leaf-VCF call, the solver also returns
    # the first attacker move of the forcing line — store it so the picker
    # can play it directly (we know the move without descending). -1 = unset.
    proven_action: int = -1

    def total_visits(self) -> int:
        return int(self.N.sum())


def _select_action(
    node: Node,
    c_puct_init: float,
    c_puct_base: float,
    fpu_reduction_c: float = 0.0,
    proven_prop: bool = False,
) -> int:
    """PUCT selection over legal actions only.

    AlphaGo Zero log-schedule:

        pb_c = log((1 + N_parent + c_puct_base) / c_puct_base) + c_puct_init

    pb_c replaces the constant c_puct in the standard PUCT formula. At small
    N_parent it equals c_puct_init; as the parent accumulates visits it grows
    slowly, tilting toward more exploration. With c_puct_base=19652 it's nearly
    constant for the typical sims-per-move budget but matches the AGZ recipe.

    ``fpu_reduction_c`` (default 0.0 = OFF, byte-identical legacy) enables
    KataGo-style First-Play Urgency reduction: unvisited children inherit
    ``parent_V - c * sqrt(sum_visited_priors)`` instead of Q=0. Eval-only knob;
    self-play / gen never set this (they take the c=0 branch).

    ``proven_prop`` (default False = OFF, byte-identical) enables KataGo-style
    proven-value selection overrides: an existing child with
    ``proven_value == -1`` (the OPPONENT at that child is proven-losing, so
    taking that action proves a win for US) gets effective Q=+inf — MCTS will
    always pick it if available. An existing child with ``proven_value == +1``
    (opponent proven-winning ⇒ we proven-lose down this branch) gets effective
    Q=-inf — we never voluntarily pick it. Degenerate case (every legal child
    is proven-loss for us): score is all -inf over legal, we fall back to the
    legal policy-prior argmax so selection still returns SOMETHING (the
    slowest-mate proxy). EVAL-ONLY: gen MCTS never sets ``proven_prop=True``.
    """
    total = node.total_visits()
    pb_c = float(np.log((1.0 + total + c_puct_base) / c_puct_base) + c_puct_init)
    sqrt_total = np.sqrt(total + 1e-8)
    if fpu_reduction_c > 0.0 and total > 0:
        parent_V = float(node.W.sum()) / float(total)
        sum_visited_priors = float(node.P[node.N > 0].sum())
        fpu_q = parent_V - fpu_reduction_c * float(np.sqrt(max(sum_visited_priors, 0.0)))
        Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), fpu_q)
    else:
        Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
    U = pb_c * node.P * sqrt_total / (1.0 + node.N)
    score = Q + U
    # Mask illegal moves to -inf so they're never selected.
    score = np.where(node.legal_mask, score, -np.inf)
    if proven_prop and node.children:
        # Override score at expanded children with a proven value. The child's
        # proven_value is from the CHILD's side-to-move POV (the opponent of
        # this node), so child=-1 (opp loses) ⇒ this action wins ⇒ +inf;
        # child=+1 (opp wins) ⇒ this action loses ⇒ -inf.
        for a, child in node.children.items():
            pv = child.proven_value
            if pv == -1:
                score[a] = np.inf
            elif pv == +1:
                score[a] = -np.inf
        # Degenerate: if all legal scores are -inf (every legal expanded child
        # is proven-loss AND no unexpanded child remains), fall back to
        # legal-policy argmax so we still return a valid action.
        if not np.any(np.isfinite(score) & node.legal_mask) and not np.any(
            np.isposinf(score)
        ):
            legal_p = np.where(node.legal_mask, node.P, -np.inf)
            return int(np.argmax(legal_p))
    return int(np.argmax(score))


def _n_forced_visits(k: float, prior: float, root_total: int) -> int:
    """KataGo forced-playout count for one root child (Wu 2019).

    n_forced(a) = ceil(sqrt(k * P(a) * N_root)). Returns 0 when disabled.
    """
    if k <= 0.0 or prior <= 0.0 or root_total <= 0:
        return 0
    return int(np.ceil(np.sqrt(k * prior * root_total)))


def _select_root_action_forced(node: Node, k: float) -> int:
    """Root-only forced-playout override. Returns the legal action to force,
    or -1 if no already-visited child is below its forced quota (caller falls
    back to plain PUCT). Mirrors the native C `select_forced_root_action`:
    only children with N>0 are forced (top-up of explored moves), and ties on
    deficit are broken by prior so selection is deterministic.
    """
    total = node.total_visits()
    if k <= 0.0 or total <= 0:
        return -1
    best_action = -1
    best_deficit = 0
    best_prior = -1.0
    for a in np.flatnonzero(node.legal_mask):
        n = int(node.N[a])
        if n <= 0:
            continue
        nf = _n_forced_visits(k, float(node.P[a]), total)
        deficit = nf - n
        if deficit <= 0:
            continue
        p = float(node.P[a])
        if deficit > best_deficit or (deficit == best_deficit and p > best_prior):
            best_deficit = deficit
            best_prior = p
            best_action = int(a)
    return best_action


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
    """Compute terminal info and legal mask for a fresh node.

    Also seeds ``proven_value`` (derby-b3n) from the terminal outcome when
    applicable: a terminal with ``terminal_value == -1.0`` means this node's
    side-to-move just lost (the previous mover, now plane 1, has five), so the
    node is proven-loss for the side to move (``proven_value = -1``). Defensive
    +1.0 case (won't arise from natural play but guarded for solver-set leaves)
    maps to ``+1``. Draws (``0.0``) leave ``proven_value = 0``.
    """
    done, v, legal_mask = state_ops.init_node_status(
        node.state.board,
        node.state.move_count,
    )
    node.is_terminal = done
    node.terminal_value = v
    if legal_mask is not None:
        node.legal_mask = legal_mask
    if done:
        if v <= -0.5:
            node.proven_value = -1
        elif v >= 0.5:
            node.proven_value = +1
        # else (draw): leave proven_value = 0; a draw is not a proven outcome.


@dataclass
class _PendingLeaf:
    """A leaf reached by selection, awaiting evaluation."""

    leaf: Node
    path: list[tuple[Node, int]]  # (parent, action_from_parent) for backprop


def _select_one(
    root: Node,
    c_puct_init: float,
    c_puct_base: float,
    forced_playout_k: float = 0.0,
    fpu_reduction_c: float = 0.0,
    proven_prop: bool = False,
) -> _PendingLeaf:
    """Descend tree until we reach either an unexpanded node or a terminal."""
    node = root
    path: list[tuple[Node, int]] = []
    while True:
        if node.is_terminal:
            return _PendingLeaf(leaf=node, path=path)
        # Proven nodes act as "virtual terminals" once proven_prop is on — no
        # point descending further into a subtree whose value is fixed.
        if proven_prop and node.proven_value != 0:
            return _PendingLeaf(leaf=node, path=path)
        if not node.expanded:
            return _PendingLeaf(leaf=node, path=path)
        a = -1
        if forced_playout_k > 0.0 and node is root:
            a = _select_root_action_forced(node, forced_playout_k)
        if a < 0:
            a = _select_action(
                node,
                c_puct_init,
                c_puct_base,
                fpu_reduction_c=fpu_reduction_c,
                proven_prop=proven_prop,
            )
        if a not in node.children:
            # Lazy child creation
            child_state = node.state.apply(a)
            child = Node(state=child_state, parent=node, parent_action=a)
            _init_node(child)
            node.children[a] = child
        path.append((node, a))
        node = node.children[a]


def _maybe_propagate_proven(parent: Node) -> None:
    """KataGo-style proven-value propagation rules (derby-b3n).

    Called bottom-up after a backup step touches ``parent.children[action]``.
    From ``parent``'s side-to-move POV:

    - If ANY existing child has ``proven_value == -1`` (the opponent at that
      child is proven-losing), playing that move proves a win for ``parent``:
      ``parent.proven_value = +1``.
    - Else, if every LEGAL child is expanded AND every one has
      ``proven_value == +1`` (the opponent always wins from there), then no
      matter what ``parent`` plays it loses: ``parent.proven_value = -1``.
    - Otherwise leave it unknown.

    Idempotent: once ``parent.proven_value != 0`` we never overwrite it (the
    first proof is correct under MCTS's tree-search semantics).
    """
    if parent.proven_value != 0:
        return
    # Fast path: any expanded child with proven_value == -1 ⇒ parent wins.
    for child in parent.children.values():
        if child.proven_value == -1:
            parent.proven_value = +1
            return
    # All-children-proven-+1 ⇒ parent loses. Requires every LEGAL action to
    # have a corresponding expanded child AND each child proven_value=+1.
    n_legal = int(parent.legal_mask.sum())
    if n_legal == 0 or len(parent.children) < n_legal:
        return
    for a in np.flatnonzero(parent.legal_mask):
        c = parent.children.get(int(a))
        if c is None or c.proven_value != +1:
            return
    parent.proven_value = -1


def _backprop(
    path: list[tuple[Node, int]],
    leaf_value: float,
    *,
    leaf_proven_value: int = 0,
    proven_prop: bool = False,
) -> None:
    """Backpropagate leaf_value, flipping sign at every level.

    leaf_value is from the LEAF state's side-to-move perspective. The parent
    along the path has the opposite side-to-move (because state.apply() flips
    planes), so its value contribution is -leaf_value, etc.

    When ``proven_prop=True`` (derby-b3n) AND a node along the path becomes
    proven (either the leaf was set to ``leaf_proven_value != 0``, or a child
    along the walk becomes proven via the propagation rule), every step uses
    the proven value with INFINITE confidence — the running ``v`` is replaced
    by ``-child.proven_value`` whenever the child we just visited is proven,
    overriding ``W/N`` along the path so future selection sees the fixed
    value. After updating each ``parent`` we re-check whether it itself
    becomes proven via :func:`_maybe_propagate_proven` and cascade upward.
    """
    v = leaf_value
    # leaf_proven_value lets the caller seed a non-terminal proven outcome
    # (e.g. solve_vcf at leaf-expansion sets +1) on a leaf that has no parent
    # yet on the path; downstream the backup picks it up via parent.children.
    if proven_prop:
        # Walk bottom-up; at each level let the child's proven_value (if any)
        # override the running value, then update parent and check propagation.
        # Note: in path, each pair is (parent, action) where the CHILD is
        # parent.children[action]. The leaf is implicitly the child of the
        # deepest path entry — leaf_proven_value is its proven_value if set.
        child_pv: int = leaf_proven_value
        for parent, action in reversed(path):
            v = -v
            if child_pv != 0:
                # Proven override: contribution is the proven value from
                # parent's POV (negation of child's POV).
                v = float(-child_pv)
            parent.N[action] += 1
            parent.W[action] += v
            _maybe_propagate_proven(parent)
            # If parent itself became proven, that propagates further up: the
            # NEXT iteration's "child" is this parent.
            child_pv = parent.proven_value
        return
    # Legacy (proven_prop OFF) — byte-identical to the pre-lever path.
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
        forced_playout_k: float = 0.0,
        fpu_reduction_c: float = 0.0,
        proven_prop: bool = False,
        proven_vcf_leaf_nodes: int = 0,
        rng: np.random.Generator | None = None,
    ):
        self.c_puct = c_puct  # acts as c_puct_init in the AGZ log schedule
        self.c_puct_base = c_puct_base
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        # KataGo forced-playouts constant. 0.0 == OFF == byte-identical legacy.
        self.forced_playout_k = forced_playout_k
        # KataGo FPU-reduction constant. 0.0 == OFF == byte-identical legacy
        # (unvisited children take Q=0). Eval-only knob.
        self.fpu_reduction_c = fpu_reduction_c
        # derby-b3n: KataGo proven-value propagation. False == OFF ==
        # byte-identical legacy. Eval-only knob.
        self.proven_prop = proven_prop
        # derby-b3n: leaf-expansion bounded VCF. 0 == OFF == byte-identical
        # legacy. When >0, expanded leaves run a bounded solve_vcf to
        # potentially seed proven_value=+1 directly. Requires proven_prop=True
        # to be useful (the proof just sits dormant otherwise). Eval-only.
        self.proven_vcf_leaf_nodes = proven_vcf_leaf_nodes
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
            # derby-b3n: bounded leaf-VCF at root expansion. The root IS a
            # leaf the first time it is expanded; seed proven_value=+1 if
            # solve_vcf proves a win for the side-to-move.
            pp = getattr(g, "proven_prop", False)
            vcf_n = getattr(g, "proven_vcf_leaf_nodes", 0)
            if pp and vcf_n > 0 and g.root.proven_value == 0:
                from gomoku.vcf import solve_vcf
                res = solve_vcf(g.root.state.board, max_nodes=vcf_n)
                if res.has_forced_win:
                    g.root.proven_value = +1
                    if res.winning_move is not None:
                        g.root.proven_action = int(res.winning_move)
        # We do NOT backprop the root's own value; the root is not a leaf in the AlphaZero sense.

    for _ in range(n_simulations):
        pending: list[tuple[MCTSGame, _PendingLeaf]] = []
        for g in games:
            pl = _select_one(
                g.root,
                g.c_puct,
                g.c_puct_base,
                getattr(g, "forced_playout_k", 0.0),
                getattr(g, "fpu_reduction_c", 0.0),
                proven_prop=getattr(g, "proven_prop", False),
            )
            pending.append((g, pl))

        # Split into terminal/proven (no eval needed) and to-evaluate.
        to_eval: list[tuple[MCTSGame, _PendingLeaf]] = []
        for g, p in pending:
            pp = getattr(g, "proven_prop", False)
            if p.leaf.is_terminal:
                _backprop(
                    p.path,
                    p.leaf.terminal_value,
                    leaf_proven_value=p.leaf.proven_value if pp else 0,
                    proven_prop=pp,
                )
            elif pp and p.leaf.proven_value != 0:
                # Proven non-terminal leaf (e.g. from leaf-VCF). Use the
                # proven value as the backed-up signal with infinite confidence.
                _backprop(
                    p.path,
                    float(p.leaf.proven_value),
                    leaf_proven_value=p.leaf.proven_value,
                    proven_prop=True,
                )
            else:
                to_eval.append((g, p))

        if not to_eval:
            continue

        states = [p.leaf.state for _g, p in to_eval]
        priors, values = evaluator(states)
        for (g, p), prior, value in zip(to_eval, priors, values):
            if not p.leaf.expanded:
                _set_priors(p.leaf, prior)
                p.leaf.expanded = True
                # derby-b3n: optional bounded VCF at leaf expansion. When
                # vcf_leaf_nodes > 0 we run solve_vcf on the side-to-move at
                # this leaf; if it proves a win, seed proven_value=+1 directly.
                pp = getattr(g, "proven_prop", False)
                vcf_n = getattr(g, "proven_vcf_leaf_nodes", 0)
                if pp and vcf_n > 0 and p.leaf.proven_value == 0:
                    from gomoku.vcf import solve_vcf
                    res = solve_vcf(p.leaf.state.board, max_nodes=vcf_n)
                    if res.has_forced_win:
                        p.leaf.proven_value = +1
                        if res.winning_move is not None:
                            p.leaf.proven_action = int(res.winning_move)
            pp = getattr(g, "proven_prop", False)
            leaf_pv = p.leaf.proven_value if pp else 0
            # If the leaf is proven (from leaf-VCF), override the value
            # contribution with the proven value (+1 from leaf POV) instead of
            # the network's V — this is the "infinite confidence" override.
            v_in = float(leaf_pv) if leaf_pv != 0 else float(value)
            _backprop(
                p.path, v_in, leaf_proven_value=leaf_pv, proven_prop=pp,
            )


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


def _select_one_vloss(
    root: Node, c_puct_init: float, c_puct_base: float, forced_playout_k: float = 0.0
) -> _PendingLeaf:
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
        a = -1
        if forced_playout_k > 0.0 and node is root:
            a = _select_root_action_forced(node, forced_playout_k)
        if a < 0:
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

        # KataGo forced playouts at the ROOT only: for any descent currently
        # sitting at its game's root (empty path so far), override the PUCT
        # pick with the forced action when this game has k>0 and an
        # already-visited root child is under its forced quota. Default k==0
        # leaves `actions` untouched → byte-identical legacy behavior.
        # Descends to root only at the first level of each descent.
        for j, i in enumerate(still):
            k = getattr(games_subset[i], "forced_playout_k", 0.0)
            if k > 0.0 and nodes[i] is games_subset[i].root:
                fa = _select_root_action_forced(nodes[i], k)
                if fa >= 0:
                    actions[j] = fa

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


def _pruned_root_visits(
    root: Node, k: float, c_puct_init: float, c_puct_base: float
) -> np.ndarray:
    """KataGo policy-target pruning (Wu 2019). Return root child visit counts
    after subtracting forced playouts. Most-visited child keeps all visits; for
    each other child, subtract up to its n_forced quota WITHOUT letting its PUCT
    selection value exceed the best child's. k<=0 returns root.N unchanged
    (byte-identical legacy target). Mirrors native `compute_pruned_root_visits`.
    """
    counts = root.N.astype(np.int64).copy()
    total = int(root.N.sum())
    if k <= 0.0 or total <= 0:
        return counts
    legal = np.flatnonzero(root.legal_mask)
    if len(legal) == 0:
        return counts
    best_action = int(legal[np.argmax(root.N[legal])])
    pb_c = float(np.log((1.0 + total + c_puct_base) / c_puct_base) + c_puct_init)
    sqrt_total = float(np.sqrt(total + 1e-8))
    best_n = int(root.N[best_action])
    best_q = float(root.W[best_action]) / best_n if best_n > 0 else 0.0
    best_u = pb_c * float(root.P[best_action]) * sqrt_total / (1.0 + best_n)
    best_value = best_q + best_u
    for a in legal:
        a = int(a)
        if a == best_action or root.N[a] <= 0:
            continue
        nf = _n_forced_visits(k, float(root.P[a]), total)
        if nf <= 0:
            continue
        q = float(root.W[a]) / float(root.N[a])
        n = int(root.N[a])
        subtracted = 0
        while subtracted < nf and n > 1:
            u_at = pb_c * float(root.P[a]) * sqrt_total / (1.0 + (n - 1))
            if q + u_at > best_value:
                break
            n -= 1
            subtracted += 1
        counts[a] = n
    return counts


def policy_from_visits(
    root: Node,
    temperature: float,
    *,
    forced_playout_k: float = 0.0,
    c_puct_init: float = 1.25,
    c_puct_base: float = 19652.0,
) -> np.ndarray:
    """Return MCTS visit-based policy over actions.

    temperature: 0 = greedy (one-hot on max visits), 1 = proportional to visits,
                 small values sharpen.

    forced_playout_k > 0 enables KataGo policy-target pruning: forced-playout
    visits are subtracted from the visit counts before forming the target.
    Default 0.0 == OFF == legacy byte-identical behavior.
    """
    if forced_playout_k > 0.0:
        counts = _pruned_root_visits(
            root, forced_playout_k, c_puct_init, c_puct_base
        ).astype(np.float64)
    else:
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
