"""walt on gomoku — a deliberate misapplication, built as well as we can.

texas-42 walt (level-k best response over sampled deals, played from trick 1)
transplanted onto a perfect-information game. Gomoku has no hidden deal, so the
hidden thing we sample is the field's *tape*: the seed of a stochastic opponent
model. Tickertape semantics (texas-42 Def 3.5): the field's randomness is keyed
on (position, tape), not on the path — the same tape at the same position draws
the same move in every branch, so tapes PARTITION into buckets by drawn move
rather than multiplying branches.

Level 1 at the root:
  value(my node, W)    = max_a value(child_a, W)            one action covers every tape
  value(field node, W) = sum_m value(child_m, W_m)          W_m = tapes that draw m
  value(leaf, W)       = sum_{r in W} outcome(leaf, r)      VCT oracle, else batched rollout
Payoff is an exact integer per tape: walt win 2, draw 1, loss 0.

The field = policy (uniform-near or the net at temperature T) + the VCT finisher
(if the mover has a proven forced win, it plays it). The finisher makes the VCT
oracle an *exact* statement about the modeled field, which is walt's semantics.

Batching: the tree is expanded breadth-first (one net call per depth), every leaf
board goes to one VCT call, and all unresolved (leaf, tape) rollouts advance in
lockstep — one net call + (every k plies) one VCT call per ply for the lot.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import numpy as np

from gomoku.game import BOARD_SIZE, GameState

N_ACT = BOARD_SIZE * BOARD_SIZE
WIN, DRAW, LOSS = 2, 1, 0


def _tape_u(state: GameState, tape: int) -> float:
    """Uniform in [0,1) keyed on (position, tape) — the tickertape."""
    h = hashlib.blake2b(state.board.tobytes(), digest_size=8,
                        key=int(tape).to_bytes(8, "little")).digest()
    return (int.from_bytes(h, "little") >> 11) / float(1 << 53)


def _near_mask(state: GameState, radius: int = 2) -> np.ndarray:
    occ = state.board[0] | state.board[1]
    legal = ~occ
    if not occ.any():
        m = np.zeros_like(legal)
        c = BOARD_SIZE // 2
        m[c, c] = True
        return m.reshape(-1)
    near = np.zeros_like(occ)
    rs, cs = np.nonzero(occ)
    for r, c in zip(rs, cs):
        near[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1] = True
    m = (near & legal).reshape(-1)
    return m if m.any() else legal.reshape(-1)


@dataclass
class Field:
    """Level-0 mind: a stochastic policy, sampled through the tape."""
    kind: str                 # "uniform" (near-stone uniform) | "net"
    evaluator: object = None  # make_torch_evaluator(...) for kind="net" (and candidates)
    temperature: float = 1.0

    def probs(self, states: list[GameState]) -> np.ndarray:
        out = np.zeros((len(states), N_ACT), dtype=np.float64)
        if not states:
            return out
        if self.kind == "uniform":
            for i, s in enumerate(states):
                m = _near_mask(s)
                out[i, m] = 1.0 / m.sum()
            return out
        logits, _ = self.evaluator(states)
        logits = np.asarray(logits, dtype=np.float64)
        for i, s in enumerate(states):
            lm = s.legal_mask()
            z = np.where(lm, logits[i] / max(self.temperature, 1e-6), -np.inf)
            z -= z.max()
            p = np.exp(z)
            out[i] = p / p.sum()
        return out


def _draw(p: np.ndarray, u: float) -> int:
    c = np.cumsum(p)
    return int(min(np.searchsorted(c, u * c[-1], side="right"), len(p) - 1))


class VCT:
    """Batched GPU VCT oracle wrapper with accounting."""

    def __init__(self, nodes: int):
        from gomoku.eval import _load_vct_solver
        self.solve = _load_vct_solver()
        self.nodes = nodes
        self.calls = 0
        self.boards = 0
        self.secs = 0.0
        self.solve(np.zeros((1, 2, BOARD_SIZE, BOARD_SIZE), bool), max_nodes=1)  # warm

    def __call__(self, states: list[GameState]):
        """-> (win (B,) bool for side-to-move, move (B,) int, hit_cap (B,) bool)."""
        if not states:
            z = np.zeros(0, bool)
            return z, np.zeros(0, np.int32), z
        t = time.perf_counter()
        boards = np.stack([s.board for s in states]).astype(bool)
        win, hit, move = self.solve(boards, max_nodes=self.nodes, return_move=True)
        self.secs += time.perf_counter() - t
        self.calls += 1
        self.boards += len(states)
        return np.asarray(win, bool), np.asarray(move), np.asarray(hit, bool)


@dataclass
class Node:
    state: GameState
    mine: bool                      # walt to move here
    tapes: np.ndarray               # tape indices reaching this node
    depth: int
    term: int | None = None         # terminal payoff (walt POV) if game over here
    children: list = field(default_factory=list)   # list[(action, Node)]
    value: int = 0


@dataclass
class WaltConfig:
    n_tapes: int = 32
    horizon: int = 2                # plies expanded before rollouts (>=1)
    n_cand: int = 8                 # candidates at walt's nodes (top-M by net prior)
    vct_nodes: int = 50             # oracle budget (leaf + finisher)
    rollout_vct_every: int = 1      # VCT terminus check cadence inside rollouts (1 = both sides; 2 at an even horizon only ever checks walt — optimism bug)
    seed: int = 0


class WaltPlayer:
    """Picker: (state, rng) -> action. Keeps per-move diagnostics in self.log."""

    def __init__(self, cfg: WaltConfig, field_: Field, cand_eval, vct: VCT, label="walt"):
        self.cfg, self.field, self.cand_eval, self.vct, self.label = cfg, field_, cand_eval, vct, label
        self.log: list[dict] = []
        self._tape_rng = np.random.default_rng(cfg.seed)

    # ---- helpers -------------------------------------------------------
    def _candidates(self, states: list[GameState]) -> list[np.ndarray]:
        logits, _ = self.cand_eval(states)
        out = []
        for i, s in enumerate(states):
            lm = s.legal_mask()
            z = np.where(lm, np.asarray(logits[i], np.float64), -np.inf)
            k = min(self.cfg.n_cand, int(lm.sum()))
            out.append(np.argsort(-z)[:k])
        return out

    @staticmethod
    def _terminal(state: GameState, mover_is_walt: bool) -> int | None:
        done, v = state.is_terminal()
        if not done:
            return None
        if v == -1.0:  # player who just moved won
            return WIN if mover_is_walt else LOSS
        return DRAW

    # ---- the decision --------------------------------------------------
    def __call__(self, state: GameState, rng: np.random.Generator) -> int:
        t0 = time.perf_counter()
        vct0 = (self.vct.calls, self.vct.boards, self.vct.secs)
        diag: dict = {"ply": state.move_count}

        # Root finisher (same as the champion product): proven win -> take it.
        w, mv, _ = self.vct([state])
        if w[0] and mv[0] >= 0 and state.legal_mask()[mv[0]]:
            diag.update(kind="finisher", secs=time.perf_counter() - t0)
            self.log.append(diag)
            return int(mv[0])

        tapes = self._tape_rng.integers(1, 2**62, size=self.cfg.n_tapes)
        all_idx = np.arange(self.cfg.n_tapes)
        root = Node(state, True, all_idx, 0)
        frontier = [root]
        leaves: list[Node] = []

        # Breadth-first expansion: one batched net call per depth.
        while frontier:
            open_ = [n for n in frontier if n.depth < self.cfg.horizon]
            leaves += [n for n in frontier if n.depth >= self.cfg.horizon]
            mine = [n for n in open_ if n.mine]
            theirs = [n for n in open_ if not n.mine]
            nxt: list[Node] = []
            if mine:
                for n, cands in zip(mine, self._candidates([n.state for n in mine])):
                    for a in cands:
                        s2 = n.state.apply(int(a))
                        c = Node(s2, False, n.tapes, n.depth + 1, term=self._terminal(s2, True))
                        n.children.append((int(a), c))
                        (nxt if c.term is None else leaves).append(c)
            if theirs:
                moves = self._field_moves([n.state for n in theirs], [n.tapes for n in theirs], tapes)
                for n, mv_per_tape in zip(theirs, moves):
                    for a in np.unique(mv_per_tape):
                        sub = n.tapes[mv_per_tape == a]
                        s2 = n.state.apply(int(a))
                        c = Node(s2, True, sub, n.depth + 1, term=self._terminal(s2, False))
                        n.children.append((int(a), c))
                        (nxt if c.term is None else leaves).append(c)
            frontier = nxt

        # Leaves: terminal -> exact; else VCT oracle; else rollouts.
        open_leaves = [n for n in leaves if n.term is None]
        for n in leaves:
            if n.term is not None:
                n.value = n.term * len(n.tapes)
        lw, _, lhit = self.vct([n.state for n in open_leaves])
        rollout_leaves = []
        for n, win in zip(open_leaves, lw):
            if win:  # side to move at leaf has a forced win; the field's finisher plays it too
                n.value = (WIN if n.mine else LOSS) * len(n.tapes)
            else:
                rollout_leaves.append(n)
        diag["leaves"] = len(leaves)
        diag["leaf_terminal"] = len(leaves) - len(open_leaves)
        diag["leaf_vct_resolved"] = int(lw.sum())
        diag["leaf_vct_hitcap"] = int(lhit.sum())
        tr = time.perf_counter()
        roll_stats = self._rollouts(rollout_leaves, tapes)
        diag.update(roll_stats)
        diag["rollout_secs"] = time.perf_counter() - tr

        # Backup: field nodes sum buckets, walt nodes max over actions.
        def backup(n: Node) -> int:
            if not n.children:
                return n.value
            vals = [backup(c) for _, c in n.children]
            n.value = max(vals) if n.mine else sum(vals)
            return n.value

        backup(root)
        vals = np.array([c.value for _, c in root.children])
        acts = [a for a, _ in root.children]
        best = vals.max()
        tied = np.flatnonzero(vals == best)
        choice = acts[int(tied[0])]  # ties -> highest net prior (candidates are prior-sorted)
        full = 2 * len(all_idx)
        diag.update(kind="search", values=vals.tolist(), acts=acts, choice=choice,
                    net_top=acts[0], n_tied=int(len(tied)), saturated=bool(best in (0, full) or len(tied) == len(vals)),
                    best_frac=float(best) / full,
                    vct_calls=self.vct.calls - vct0[0], vct_boards=self.vct.boards - vct0[1],
                    vct_secs=self.vct.secs - vct0[2], secs=time.perf_counter() - t0)
        self.log.append(diag)
        return choice

    def _field_moves(self, states, tape_sets, tapes) -> list[np.ndarray]:
        """Field's drawn move per tape at each state (finisher first, then policy via tape)."""
        w, mv, _ = self.vct(states)
        probs = self.field.probs([s for s, win in zip(states, w) if not win])
        out, j = [], 0
        for s, ts, win, m in zip(states, tape_sets, w, mv):
            if win and m >= 0:
                out.append(np.full(len(ts), int(m)))
                continue
            p = probs[j]; j += 1
            out.append(np.array([_draw(p, _tape_u(s, tapes[t])) for t in ts]))
        return out

    def _rollouts(self, leaves: list[Node], tapes) -> dict:
        """Lockstep-batched rollouts of every (leaf, tape) pair; both sides play the field."""
        items = []  # [state, walt_to_move, leaf, tape]
        for n in leaves:
            for t in n.tapes:
                items.append([n.state, n.mine, n, int(t)])
        n_items = len(items)
        lengths = np.zeros(n_items, int)
        live = list(range(n_items))
        ply = 0
        resolved_by_vct = 0
        while live:
            states = [items[i][0] for i in live]
            if ply % self.cfg.rollout_vct_every == 0:
                w, mv, _ = self.vct(states)
            else:
                w, mv = np.zeros(len(live), bool), np.full(len(live), -1)
            still = []
            need_policy = []
            for k, i in enumerate(live):
                s, walt_tm, leaf, t = items[i]
                if w[k]:
                    leaf.value += WIN if walt_tm else LOSS
                    resolved_by_vct += 1
                    lengths[i] = ply
                else:
                    need_policy.append(k)
            probs = self.field.probs([states[k] for k in need_policy])
            for j, k in enumerate(need_policy):
                i = live[k]
                s, walt_tm, leaf, t = items[i]
                a = _draw(probs[j], _tape_u(s, tapes[t]))
                s2 = s.apply(a)
                term = self._terminal(s2, walt_tm)
                if term is not None:
                    leaf.value += term
                    lengths[i] = ply + 1
                else:
                    items[i] = [s2, not walt_tm, leaf, t]
                    still.append(i)
            live = still
            ply += 1
        return {"rollouts": n_items, "rollout_vct_resolved": resolved_by_vct,
                "rollout_len_mean": float(lengths.mean()) if n_items else 0.0}
