"""Gumbel AlphaZero root action selection + Sequential Halving.

Reference: Danihelka, Guez, Schrittwieser, Silver, "Policy improvement by
planning with Gumbel" (ICLR 2022). This is the *root-only* formulation (the
standard one used by Gumbel MuZero / AlphaZero): the root samples `m` candidate
actions with the Gumbel-top-k trick, distributes the simulation budget across
them with Sequential Halving, and produces a "completed-Q" policy improvement
target. Internal (non-root) nodes keep ordinary PUCT.

Why this matters here: vanilla MCTS at low simulation counts ("never grokked")
gives a weak visit-count target. Gumbel provably improves the policy even at
n=2..16 sims, which directly attacks our generation-bound regime.

This module reuses the tree machinery from `gomoku.mcts` (`Node`, `_init_node`,
`_set_priors`, `_select_one`/`_select_one_vloss`, `_backprop*`). The ONLY thing
it changes is *root* scheduling and the policy target; everything below the root
is the unchanged PUCT descent. The standard (non-Gumbel) path in `mcts.py` is
untouched and byte-identical.

Constants (paper defaults): c_visit=50.0, c_scale=1.0,
    sigma(q) = (c_visit + max_visit_at_root) * c_scale * q.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gomoku.game import N_ACTIONS, GameState
from gomoku.mcts import (
    Evaluator,
    Node,
    _backprop,
    _init_node,
    _select_one,
    _set_priors,
)

C_VISIT_DEFAULT = 50.0
C_SCALE_DEFAULT = 1.0
M_DEFAULT = 16


def _root_logits(node: Node, raw_priors: np.ndarray) -> np.ndarray:
    """Center the raw network logits over the legal actions.

    Gumbel works on *logits*, not the softmaxed priors. We keep the raw logits
    (illegal actions set to -inf so they're never sampled) and center them by
    subtracting the legal max — Gumbel-top-k and the softmax target are both
    shift-invariant, so centering only improves numerical conditioning.
    """
    logits = np.where(node.legal_mask, raw_priors.astype(np.float64), -np.inf)
    finite = logits[np.isfinite(logits)]
    if finite.size:
        logits = logits - finite.max()
    return logits


def _gumbel_topk(
    logits: np.ndarray,
    legal_idx: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (without replacement) the top-`m` actions by logits + Gumbel noise.

    Returns (sampled_actions, gumbel) where:
      sampled_actions: (k,) int array, k = min(m, #legal), the chosen candidates
                       (the top-k of logits+g, descending).
      gumbel:          (N_ACTIONS,) float64, the Gumbel noise drawn per legal
                       action (illegal => 0.0). Needed later because the
                       Sequential-Halving / final-selection score is
                       g(a) + logits(a) + sigma(q_hat(a)) — the SAME Gumbel
                       sample is reused throughout (this is the Gumbel-top-k
                       trick that makes SH an unbiased policy-improvement op).
    """
    gumbel = np.zeros(N_ACTIONS, dtype=np.float64)
    # Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1).
    u = rng.random(len(legal_idx))
    u = np.clip(u, 1e-300, 1.0)
    g = -np.log(-np.log(u))
    gumbel[legal_idx] = g

    scores = gumbel + logits  # logits already -inf on illegal
    k = int(min(m, len(legal_idx)))
    # argsort descending, restricted to finite (legal) entries.
    order = np.argsort(scores)[::-1]
    sampled = order[:k].astype(np.int64)
    return sampled, gumbel


def _sigma(q: np.ndarray, max_visit: int, c_visit: float, c_scale: float) -> np.ndarray:
    """Monotone transform of Q used in the Gumbel selection score.

    sigma(q) = (c_visit + max_visit) * c_scale * q,  with q in [-1, 1].
    max_visit is the largest visit count over the root's children so far.
    """
    return (c_visit + float(max_visit)) * c_scale * q


def _root_q_completed(
    node: Node,
    value_logits_v: float,
) -> np.ndarray:
    """Completed Q over ALL legal root actions.

    For visited actions: q_hat(a) = W[a] / N[a]  (note: W is stored from the
    PARENT's perspective via _backprop's sign flip, so W[a]/N[a] is already the
    value of playing `a` from the root's side-to-move — exactly what we want).

    For unvisited actions: use the "v_mix" value completion from the paper —
    the visit-weighted mix of the root value estimate and the visited children's
    Q. With no visits anywhere v_mix == value_logits_v (the network's root value).
    This keeps the completed-policy target well-defined for every legal action
    even at very low sim budgets.
    """
    legal = node.legal_mask
    N = node.N
    visited = legal & (N > 0)
    q = np.zeros(N_ACTIONS, dtype=np.float64)
    q[visited] = node.W[visited] / np.maximum(N[visited], 1)

    sum_n = int(N[legal].sum())
    if sum_n > 0:
        # sum over visited of pi(a) * q(a); pi here = prior P (already softmaxed,
        # legal-only). Paper uses the policy prior weights.
        pi = node.P
        weighted_q = float((pi[visited] * q[visited]).sum())
        sum_pi_visited = float(pi[visited].sum())
        if sum_pi_visited > 0:
            v_mix = (value_logits_v + (sum_n / sum_pi_visited) * weighted_q) / (1.0 + sum_n)
        else:
            v_mix = value_logits_v
    else:
        v_mix = value_logits_v

    unvisited = legal & (N == 0)
    q[unvisited] = v_mix
    return q


def completed_policy_target(
    node: Node,
    root_logits: np.ndarray,
    value_root: float,
    *,
    c_visit: float,
    c_scale: float,
) -> np.ndarray:
    """The Gumbel policy-improvement TARGET (NOT visit counts).

    pi'(a) = softmax over legal a of [ logits(a) + sigma(completed_q(a)) ].

    Returns an (N_ACTIONS,) float32 pmf that is non-negative and sums to 1 over
    legal actions (0 on illegal). This is what gets stored as the training
    policy target. `value_root` is the network's value estimate at the root,
    used to complete Q for unvisited actions.
    """
    legal = node.legal_mask
    max_visit = int(node.N.max()) if legal.any() else 0
    q = _root_q_completed(node, value_root)
    sig = _sigma(q, max_visit, c_visit, c_scale)
    raw = np.where(legal, root_logits + sig, -np.inf)
    finite = raw[np.isfinite(raw)]
    out = np.zeros(N_ACTIONS, dtype=np.float64)
    if finite.size == 0:
        # degenerate: uniform over legal
        if legal.any():
            out[legal] = 1.0 / int(legal.sum())
        return out.astype(np.float32)
    m = finite.max()
    exp = np.where(legal, np.exp(raw - m), 0.0)
    s = exp.sum()
    if s > 0:
        out = exp / s
    elif legal.any():
        out[legal] = 1.0 / int(legal.sum())
    return out.astype(np.float32)


def _sequential_halving_schedule(m: int, budget: int) -> list[int]:
    """Per-phase visits-per-candidate for Sequential Halving.

    Standard SH: log2(m) phases; each phase visits every surviving candidate
    `n_per` times (round-robin), then keeps the top half by score. The budget is
    split as evenly as possible across phases. Returns a list of `n_per` values,
    one per phase, where the candidate set halves each phase (m, m/2, m/4, ... 2).

    Guarantees: never schedule fewer than 1 visit/candidate/phase; total visits
    <= budget is approximate (we cap the LAST phase so the grand total never
    exceeds `budget`). With m candidates and budget>=m, every candidate gets at
    least one visit.
    """
    if m <= 1 or budget <= 0:
        return []
    import math

    n_phases = max(1, int(math.floor(math.log2(m))))
    # Candidate-set sizes per phase: m, ceil(m/2), ... down to >=2.
    sizes = []
    cur = m
    for _ in range(n_phases):
        sizes.append(cur)
        cur = max(2, cur // 2)
    # Even split of budget across phases, weighted by candidate-set size so each
    # phase gets ~ budget/n_phases total visits (paper's allocation).
    per_phase_total = max(1, budget // n_phases)
    schedule = []
    used = 0
    for i, size in enumerate(sizes):
        if i == len(sizes) - 1:
            remaining = budget - used
            n_per = max(1, remaining // size)
        else:
            n_per = max(1, per_phase_total // size)
        schedule.append(n_per)
        used += n_per * size
    return schedule


@dataclass
class GumbelResult:
    action: int          # the selected move (argmax of the SH score)
    pi: np.ndarray       # (N_ACTIONS,) completed-policy training target, float32
    visits: int          # total root simulations actually spent


def _simulate_from_child(
    root: Node,
    action: int,
    evaluator: Evaluator,
) -> None:
    """Run ONE simulation whose first edge from the root is forced to `action`.

    The child (and everything below it) uses ordinary PUCT via `_select_one`.
    This is the Gumbel formulation: root edge is scheduled by Sequential
    Halving; internal nodes are vanilla MCTS.
    """
    # Lazily create the child if needed.
    if action not in root.children:
        child_state = root.state.apply(action)
        child = Node(state=child_state, parent=root, parent_action=action)
        _init_node(child)
        root.children[action] = child
    child = root.children[action]

    if child.is_terminal:
        # Backprop terminal directly through the single root edge.
        _backprop([(root, action)], child.terminal_value)
        return

    # Descend from the child with PUCT to a leaf, evaluate, backprop.
    pending = _select_one(child, root_c_puct_init(root), root_c_puct_base(root))
    leaf = pending.leaf
    # Full path from root: root->action, then child's internal path.
    full_path = [(root, action)] + pending.path
    if leaf.is_terminal:
        _backprop(full_path, leaf.terminal_value)
        return
    priors, values = evaluator([leaf.state])
    if not leaf.expanded:
        _set_priors(leaf, priors[0])
        leaf.expanded = True
    _backprop(full_path, float(values[0]))


# The Node/MCTSGame don't carry c_puct themselves; the Gumbel driver receives
# it explicitly. We stash it on the root node via attributes set by the driver.
def root_c_puct_init(root: Node) -> float:
    return getattr(root, "_gumbel_c_puct_init", 1.25)


def root_c_puct_base(root: Node) -> float:
    return getattr(root, "_gumbel_c_puct_base", 19652.0)


def gumbel_search_root(
    root: Node,
    evaluator: Evaluator,
    *,
    n_simulations: int,
    m: int = M_DEFAULT,
    c_visit: float = C_VISIT_DEFAULT,
    c_scale: float = C_SCALE_DEFAULT,
    c_puct_init: float = 1.25,
    c_puct_base: float = 19652.0,
    rng: np.random.Generator | None = None,
) -> GumbelResult:
    """Gumbel root selection + Sequential Halving on a single tree `root`.

    Steps:
      1. Ensure the root is expanded (network priors + value).
      2. Sample m candidate root actions via Gumbel-top-k over the logits.
      3. Sequential Halving: round-robin visits over surviving candidates,
         halve by score g(a)+logits(a)+sigma(q_hat(a)) each phase.
      4. Selected action = argmax of that score over final survivors.
      5. Training target = completed-policy (softmax of logits + sigma(q_hat)).

    The root must already be `_init_node`'d. Internal nodes use PUCT.
    """
    rng = rng or np.random.default_rng()
    # Stash PUCT params on the node so the internal descent can read them.
    root._gumbel_c_puct_init = c_puct_init  # type: ignore[attr-defined]
    root._gumbel_c_puct_base = c_puct_base  # type: ignore[attr-defined]

    if root.is_terminal:
        return GumbelResult(action=-1, pi=np.zeros(N_ACTIONS, dtype=np.float32), visits=0)

    # 1. Expand root if needed; capture raw logits + root value.
    priors, values = evaluator([root.state])
    raw_priors = priors[0]
    value_root = float(values[0])
    if not root.expanded:
        _set_priors(root, raw_priors)
        root.expanded = True
    logits = _root_logits(root, raw_priors)

    legal_idx = np.flatnonzero(root.legal_mask)
    if len(legal_idx) == 0:
        return GumbelResult(action=-1, pi=np.zeros(N_ACTIONS, dtype=np.float32), visits=0)
    if len(legal_idx) == 1:
        a = int(legal_idx[0])
        pi = np.zeros(N_ACTIONS, dtype=np.float32)
        pi[a] = 1.0
        return GumbelResult(action=a, pi=pi, visits=0)

    # 2. Gumbel-top-k candidate sampling.
    candidates, gumbel = _gumbel_topk(logits, legal_idx, m, rng)

    # 3. Sequential Halving.
    visits_spent = 0
    if len(candidates) <= 1 or n_simulations <= 0:
        survivors = candidates
    else:
        schedule = _sequential_halving_schedule(len(candidates), n_simulations)
        survivors = candidates
        for n_per in schedule:
            if len(survivors) <= 1:
                break
            # Round-robin: visit each surviving candidate n_per times.
            for _ in range(n_per):
                for a in survivors:
                    if visits_spent >= n_simulations:
                        break
                    _simulate_from_child(root, int(a), evaluator)
                    visits_spent += 1
                if visits_spent >= n_simulations:
                    break
            # Score survivors: g(a) + logits(a) + sigma(q_hat(a)).
            max_visit = int(root.N.max())
            scores = []
            for a in survivors:
                a = int(a)
                n = int(root.N[a])
                q = float(root.W[a] / n) if n > 0 else 0.0
                s = gumbel[a] + logits[a] + _sigma(np.array([q]), max_visit, c_visit, c_scale)[0]
                scores.append(s)
            scores = np.array(scores, dtype=np.float64)
            keep = max(1, len(survivors) // 2)
            top = np.argsort(scores)[::-1][:keep]
            survivors = survivors[top]
            if visits_spent >= n_simulations:
                break

    # 4. Final selection: argmax of g + logits + sigma(q_hat) over survivors.
    max_visit = int(root.N.max()) if root.legal_mask.any() else 0
    best_a = int(survivors[0])
    best_score = -np.inf
    for a in survivors:
        a = int(a)
        n = int(root.N[a])
        q = float(root.W[a] / n) if n > 0 else 0.0
        s = gumbel[a] + logits[a] + _sigma(np.array([q]), max_visit, c_visit, c_scale)[0]
        if s > best_score:
            best_score = s
            best_a = a

    # 5. Completed-policy training target.
    pi = completed_policy_target(
        root, logits, value_root, c_visit=c_visit, c_scale=c_scale
    )
    return GumbelResult(action=best_a, pi=pi, visits=visits_spent)


# ===========================================================================
# C-PORT SPEC for gomoku/_mcts_native.c  (to make Gumbel a FAIR derby lever)
# ===========================================================================
#
# The Python path above is correct + readable but runs the pure-Python tree
# (~5x slower per game than native PUCT in CPU smoke). For an honest wall-clock
# derby lever it must run in the native path. Below is exactly what to change.
#
# WHY A NEW SEARCH ENTRY (not a flag on `native_search_batch`):
#   The existing native loop is a wave-batched PUCT scheduler: every game does
#   the SAME thing each sim (descend from root via PUCT). Gumbel's root schedule
#   is per-game STATEFUL (which candidates survive, how many visits each gets
#   this phase) and forces the FIRST edge to a Sequential-Halving-chosen action,
#   not PUCT. That's a different control flow at the root only; internal nodes
#   are byte-identical PUCT (reuse `select_one_vloss` UNCHANGED below the root).
#
# DATA the root selection needs (all already in CNode + NativeMCTSGameObject):
#   - Root logits: `set_priors` currently softmaxes raw priors into `node.P`.
#     ADD a `float logits[N_ACTIONS]` field to CNode and have `set_priors` store
#     the centered raw logits there too (illegal -> -INFINITY). Gumbel needs raw
#     logits, the completed-policy target needs them, and `node.P` (softmaxed)
#     stays for the v_mix prior weights and any PUCT below root.
#   - Root value: `call_evaluator` already returns `values`; the root's own
#     value is currently discarded. Capture it into a new
#     `float root_value` field on the game (or pass through) for v_mix.
#   - Per-game Gumbel state: add to NativeMCTSGameObject:
#       int   gumbel_candidates[N_ACTIONS];   // current survivor action ids
#       int   gumbel_n_candidates;
#       double gumbel_noise[N_ACTIONS];       // g(a), drawn once per root search
#     Reuse the existing `rng` (splitmix64) — add `rng_gumbel()` =
#     -log(-log(rng_uniform())) clamped like rng_normal's u-guard.
#
# FUNCTIONS to add to _mcts_native.c:
#   1. static void gumbel_sample_topk(game, root, m):
#        draw gumbel_noise[a] for each legal a; compute score = g + logits;
#        partial-sort legal actions desc by score; keep top min(m, n_legal) into
#        gumbel_candidates / gumbel_n_candidates.
#   2. static double sigma_q(double q, int max_visit, double c_visit,
#                            double c_scale): (c_visit+max_visit)*c_scale*q.
#   3. static int gumbel_select_score_argmax(game, root, c_visit, c_scale):
#        over survivors, argmax of g[a]+logits[a]+sigma_q(W[a]/N[a], rootMaxN).
#   4. static PyObject *NativeMCTSGame_gumbel_policy(self, c_visit, c_scale):
#        the completed-policy TARGET. Mirror `_root_q_completed` +
#        `completed_policy_target` exactly:
#          q[a] = W[a]/N[a] for visited (already root-perspective via backprop);
#          v_mix = (root_value + (sum_N/sum_P_visited)*sum(P[a]*q[a])) / (1+sum_N);
#          q[unvisited] = v_mix;
#          target = softmax over legal of logits[a] + sigma_q(q[a], maxN).
#        Return (N_ACTIONS,) float32 pmf (matches NativeMCTSGame_policy shape).
#
# NEW BATCH ENTRY `native_gumbel_search_batch(games, evaluator, n_simulations,
#                  m, c_visit, c_scale, add_root_noise=0)`:
#   - Expand+evaluate all unexpanded roots in ONE batched evaluator call (reuse
#     `evaluate_and_expand`, but also stash root_value + logits). add_root_noise
#     stays 0 — Gumbel uses Gumbel noise, NOT Dirichlet.
#   - For each game: gumbel_sample_topk(m).
#   - Sequential Halving phases (same schedule as `_sequential_halving_schedule`
#     here): for each phase, round-robin over survivors; for each scheduled visit
#     of candidate `a`, run ONE simulation whose FIRST edge from root is forced
#     to `a` (create child if child[a]<0, push (root,a) onto the path, N[root,a]+=1
#     virtual loss), THEN descend the child with the EXISTING `select_one_vloss`
#     starting at child (refactor it to take a start node_idx + seed path). Batch
#     all games' leaves into one evaluator call per slot exactly like the current
#     wave loop, backprop via `backprop_value_only` (UNCHANGED). After the phase,
#     re-score survivors and keep top half.
#   - The cross-game batching key: schedule the SAME phase/slot index across all
#     games so the per-slot leaf batch stays large (preserves MPS saturation).
#     Games with fewer survivors / exhausted budget simply contribute no leaf
#     that slot (same pattern as `wave_counts[i]` today).
#
# PYTHON GLUE (gomoku/self_play.py::_generate_games_gumbel):
#   - Swap NativeMCTSGame in (like `_generate_games_native`), call
#     native_gumbel_search_batch(...), then per game: action =
#     g.gumbel_selected_action() (store the argmax in step 3 and expose it like
#     `visit_counts`), pi = g.gumbel_policy(c_visit, c_scale); g.advance_root(a).
#   - Route `_can_use_native_mcts(evaluator) and gumbel_root` to it; keep this
#     pure-Python implementation as the no-native fallback (identical target math
#     => same training signal, just slower).
#
# VALIDATION after the port: assert native gumbel_policy matches this module's
# completed_policy_target to ~1e-5 on a handful of fixed (priors, values, visits)
# fixtures; assert sums-to-1 / non-negative; re-run tests/test_gumbel.py against
# the native path with GOMOKU_DISABLE_NATIVE_MCTS unset.
# ===========================================================================
