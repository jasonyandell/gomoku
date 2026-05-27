"""Cross-game value sidecar (Derby 'position-stats').

LEVER: de-noise the single-game value target ``z`` by blending it with a
low-variance aggregate of the (discounted) outcomes across ALL games that
passed through the same position. A single game's ±1 punishes a good move for a
blunder 10 plies later; aggregating across games turns that into a confident
Monte-Carlo discounted-value estimate, sharpening conversion and the opening.

ARCHITECTURE (trainer-owned, single-writer — no multi-process locking):
  - The 8 self-play workers keep dropping atomic game-record .pt files exactly
    as today. The TRAINER aggregates on ingest into this in-process store.
  - Canonical key = min over the 8 D4 symmetries of the side-to-move-relative
    board (reuse the same D4 transforms as ``game.augment``). Because the board
    is stored side-relative (plane 0 = current player), the key matches ``z``'s
    convention, so aggregation is correct by construction. The key is an EXACT
    base-3 packing of the 81 cells (trit per cell: 0 empty / 1 mine / 2 opp)
    into 129 bits = (hi:uint64, lo:uint64, top:uint16) ≈ 18 bytes — collision
    free, NOT a hash.
  - store = canonical_key -> [visits, sum_value]. Each contributing example adds
    its already-(value-)discounted ``z`` to sum_value (the example's ``z`` IS the
    discounted return when --value-discount<1, so the aggregate is a discounted
    Monte-Carlo value). The 8 D4 augmentations of one position share a key and a
    ``z``, so they contribute 8 visits with identical ``z`` — the MEAN is
    unbiased; the visit GATE accounts for the uniform 8× via its threshold.
  - Recency-decay: before each ingest the whole store is multiplied by
    ``recency_decay`` (a poor-man's reanalyze with no GPU) so old/weak-net games
    fade and the aggregate tracks the current policy.
  - Relabel on sample: the buffer stores each row's canonical key as a column
    (computed once at ingest). The read path is an O(1) dict lookup; the value
    target becomes a confidence-weighted blend(z_game, aggregate) gated by visit
    count — low-visit positions fall back to single-game ``z``.

OPT-IN + BYTE-IDENTICAL-WHEN-OFF: nothing here is constructed or called unless
the trainer flag --cross-game-value is set (aux-head discipline). The store is a
SIDE-CAR file (e.g. latest.position_stats.pkl), never embedded in latest.pt; on
resume it is loaded if present, else rebuilt from the buffer already embedded in
latest.pt.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from gomoku.game import BOARD_SIZE, N_INPUT_PLANES

# Powers of 3 for the base-3 trit packing of an 81-cell board. Precomputed once.
# 3**81 ≈ 2**128.4, so the packed value needs 129 bits = two uint64 + a few bits.
_N_CELLS = BOARD_SIZE * BOARD_SIZE
_POW3 = [3 ** i for i in range(_N_CELLS)]
_OPP_PLANE = N_INPUT_PLANES // 2  # current-frame opponent plane index in to_planes()

# Canonical key type: a 3-tuple (hi:int<2**64, lo:int<2**64, top:int<2**16).
PosKey = tuple


# ----- D4 symmetry over the raw 9x9 occupancy (independent of game.py's
# float-plane helpers so we can operate on the compact (2, N, N) board) -----

def _sym(board2: np.ndarray, sym: int) -> np.ndarray:
    """Apply one of 8 D4 symmetries to a (2, N, N) occupancy array. Matches the
    transform order used by gomoku.game._sym_board (rot then optional flip)."""
    rot = sym % 4
    flip = sym // 4
    out = np.rot90(board2, rot, axes=(-2, -1))
    if flip:
        out = np.flip(out, axis=-1)
    return out


def _pack_trits(mine: np.ndarray, opp: np.ndarray) -> PosKey:
    """Exact base-3 pack of one orientation: trit per cell (0 empty/1 mine/2 opp).
    Returns (hi, lo, top) ints. Python big-int math keeps this exact."""
    mine_flat = mine.reshape(-1).astype(np.int64)
    opp_flat = opp.reshape(-1).astype(np.int64)
    trits = mine_flat + 2 * opp_flat  # 0,1,2 (a cell is never both)
    val = 0
    for i in range(_N_CELLS):
        t = int(trits[i])
        if t:
            val += t * _POW3[i]
    lo = val & 0xFFFFFFFFFFFFFFFF
    hi = (val >> 64) & 0xFFFFFFFFFFFFFFFF
    top = (val >> 128) & 0xFFFF
    return (hi, lo, top)


def canonical_key_from_board(board2: np.ndarray) -> PosKey:
    """Canonical D4-invariant key for a (2, N, N) occupancy array where
    board2[0] = side-to-move stones, board2[1] = opponent stones. The key is the
    LEXICOGRAPHIC min of the packed value over all 8 D4 orientations -> identical
    for any rotation/reflection of the same position."""
    best: PosKey | None = None
    for s in range(8):
        t = _sym(board2, s)
        key = _pack_trits(t[0], t[1])
        if best is None or key < best:
            best = key
    return best  # type: ignore[return-value]


def canonical_key_from_planes(planes: np.ndarray) -> PosKey:
    """Canonical key from a full (N_INPUT_PLANES, N, N) float32 input stack.
    Uses only the CURRENT-frame occupancy: plane 0 (side-to-move) and plane
    N_INPUT_PLANES//2 (opponent). History planes are intentionally ignored so the
    key identifies a board position, not a move-order path."""
    board2 = np.stack(
        [planes[0] > 0.5, planes[_OPP_PLANE] > 0.5]
    )
    return canonical_key_from_board(board2)


@dataclass
class PositionStats:
    """Trainer-owned single-writer cross-game value store.

    store: PosKey -> [visits(float), sum_value(float)]. Floats because the
    recency decay multiplies the counts. The aggregate value of a position is
    sum_value / visits.
    """

    recency_decay: float = 0.999      # multiply all counts by this each ingest
    min_visits: float = 8.0           # below this, fall back to single-game z
    max_blend: float = 0.5            # cap on the aggregate's weight in the blend

    def __post_init__(self) -> None:
        # key -> np.array([visits, sum_value], float64)
        self.store: dict[PosKey, np.ndarray] = {}

    # ---- ingest (writer) ----
    def decay(self) -> None:
        """Recency-decay the whole store (call once per ingest cycle before
        adding the new games). A no-op when recency_decay >= 1.0."""
        if self.recency_decay >= 1.0:
            return
        d = self.recency_decay
        for v in self.store.values():
            v *= d

    def add(self, key: PosKey, z: float) -> None:
        """Add one contribution: a single example's already-discounted return."""
        cur = self.store.get(key)
        if cur is None:
            self.store[key] = np.array([1.0, float(z)], dtype=np.float64)
        else:
            cur[0] += 1.0
            cur[1] += float(z)

    def add_many(self, keys, zs) -> None:
        """Vectorized-ish batch add for an ingest cycle (after decay())."""
        for key, z in zip(keys, zs):
            self.add(key, float(z))

    # ---- read (relabel on sample) ----
    def aggregate(self, key: PosKey) -> tuple[float, float]:
        """Return (visits, mean_value) for a key; (0.0, 0.0) if unseen."""
        cur = self.store.get(key)
        if cur is None:
            return 0.0, 0.0
        visits = float(cur[0])
        if visits <= 0.0:
            return 0.0, 0.0
        return visits, float(cur[1]) / visits

    def blend(self, key: PosKey, z_game: float) -> float:
        """Confidence-weighted blend of the single-game z and the cross-game
        aggregate, gated by visit count. Below ``min_visits`` -> pure z_game.
        Above, the aggregate weight ramps from 0 toward ``max_blend`` with
        log-visits so heavily-transited positions (the opening) trust the
        aggregate most. z_game is always retained at weight (1 - w)."""
        visits, mean_v = self.aggregate(key)
        if visits < self.min_visits:
            return float(z_game)
        # Confidence ramp: 0 at min_visits, -> max_blend as visits grows.
        # log2(visits/min_visits) reaches 1.0 at 2x min_visits, 2.0 at 4x, ...
        conf = np.log2(visits / self.min_visits)
        w = self.max_blend * conf / (conf + 1.0)  # saturating in [0, max_blend)
        return float((1.0 - w) * z_game + w * mean_v)

    def relabel_z(self, keys, z_batch):
        """Vectorized relabel of a sampled z batch given each row's canonical
        key. Returns a new numpy float32 array; rows whose key is unseen / below
        the gate are returned unchanged. Pure-python loop (≈ tens of µs per 256
        batch, negligible vs the MPS step)."""
        out = np.asarray(z_batch, dtype=np.float32).copy()
        for i, key in enumerate(keys):
            out[i] = self.blend(key, float(out[i]))
        return out

    # ---- persistence (side-car, NOT embedded in latest.pt) ----
    def save(self, path: str) -> None:
        payload = {
            "store": {k: v.tolist() for k, v in self.store.items()},
            "recency_decay": self.recency_decay,
            "min_visits": self.min_visits,
            "max_blend": self.max_blend,
        }
        tmp = str(path) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        import os
        os.replace(tmp, str(path))

    @classmethod
    def load(cls, path: str) -> "PositionStats":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        ps = cls(
            recency_decay=float(payload.get("recency_decay", 0.999)),
            min_visits=float(payload.get("min_visits", 8.0)),
            max_blend=float(payload.get("max_blend", 0.5)),
        )
        ps.store = {
            tuple(k): np.array(v, dtype=np.float64)
            for k, v in payload["store"].items()
        }
        return ps

    def __len__(self) -> int:
        return len(self.store)
