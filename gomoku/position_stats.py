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
  - Recency-decay: a LAZY GLOBAL SCALE. Instead of multiplying every entry by
    ``recency_decay`` each ingest (an O(store-size) traversal that made the
    Derby-v8 epoch wall grow 14s->128s as the store hit 14MB — bead derby-eda),
    we keep one ``scale`` float and multiply IT by ``recency_decay`` per cycle
    (O(1)). Entries store RAW visits/sum_value; the effective (decayed) values
    are ``raw * scale``. New contributions this cycle are stored as
    ``contribution / scale`` so older entries decay correctly RELATIVE to newer
    ones. The blend ratio ``sum_value/visits`` is scale-invariant; the visit
    GATE compares against the EFFECTIVE decayed visits (``raw_visits * scale``).
    When ``scale`` underflows it is folded back into the store in one O(N) pass
    and reset to 1.0 (amortized negligible). The store is BOUNDED by periodic
    compaction that prunes cold entries (effective visits below a floor).
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


# When the lazy global scale drops below this, fold it into the store (one O(N)
# pass) and reset to 1.0. Far above float64 underflow; amortized negligible.
_SCALE_RENORM_FLOOR = 1e-6


@dataclass
class PositionStats:
    """Trainer-owned single-writer cross-game value store.

    store: PosKey -> [raw_visits(float), raw_sum_value(float)]. The EFFECTIVE
    (recency-decayed) values are ``raw * scale`` where ``scale`` is a single
    lazy global float (see module docstring). The aggregate value of a position
    is ``raw_sum_value / raw_visits`` — scale-invariant, so the mean is
    unaffected; the visit gate uses the effective ``raw_visits * scale``.
    """

    recency_decay: float = 0.999      # multiply the GLOBAL scale by this each ingest
    min_visits: float = 8.0           # below this (EFFECTIVE), fall back to single-game z
    max_blend: float = 0.5            # cap on the aggregate's weight in the blend
    compact_every: int = 64           # ingest cycles between bound-the-store compactions
    compact_min_visits: float = 1.0   # prune entries with effective visits below this

    def __post_init__(self) -> None:
        # key -> np.array([raw_visits, raw_sum_value], float64)
        self.store: dict[PosKey, np.ndarray] = {}
        # Lazy global recency-decay scale: effective visits/sum = raw * scale.
        self.scale: float = 1.0
        # Ingest-cycle counter for periodic compaction.
        self._cycles: int = 0

    # ---- ingest (writer) ----
    def decay(self) -> None:
        """Recency-decay (call once per ingest cycle before adding the new
        games). O(1): multiply the GLOBAL scale, not every entry. A no-op when
        recency_decay >= 1.0. Also drives periodic compaction so the store stays
        bounded (cold entries pruned), and renormalizes on scale underflow."""
        self._cycles += 1
        if self.recency_decay < 1.0:
            self.scale *= self.recency_decay
            if self.scale < _SCALE_RENORM_FLOOR:
                self._renormalize()
        if self.compact_every > 0 and self._cycles % self.compact_every == 0:
            self.compact()

    def _renormalize(self) -> None:
        """Fold the global scale into the raw counts in one O(N) pass and reset
        scale=1.0. Effective values (raw*scale) are preserved exactly."""
        s = self.scale
        if s == 1.0:
            return
        for v in self.store.values():
            v *= s
        self.scale = 1.0

    def add(self, key: PosKey, z: float) -> None:
        """Add one contribution (a single example's already-discounted return).

        Stored as ``contribution / scale`` so this NEW entry decays correctly
        relative to OLDER (already smaller-scale) entries: its effective weight
        this cycle is ``(1/scale) * scale == 1`` while older entries that were
        added at scale=1 have effective weight ``scale < 1``."""
        inv = 1.0 / self.scale
        cur = self.store.get(key)
        if cur is None:
            self.store[key] = np.array([inv, float(z) * inv], dtype=np.float64)
        else:
            cur[0] += inv
            cur[1] += float(z) * inv

    def add_many(self, keys, zs) -> None:
        """Vectorized-ish batch add for an ingest cycle (after decay())."""
        for key, z in zip(keys, zs):
            self.add(key, float(z))

    # ---- bound the store ----
    def compact(self) -> None:
        """Prune cold entries (effective visits below ``compact_min_visits``) so
        the store can't grow unbounded. O(N) but only every ``compact_every``
        cycles — amortized O(1) per cycle, vastly cheaper than the old O(N)
        per-cycle full-store decay this replaces (bead derby-eda)."""
        if self.compact_min_visits <= 0.0:
            return
        thresh = self.compact_min_visits / self.scale  # compare raw to effective floor
        self.store = {k: v for k, v in self.store.items() if v[0] >= thresh}

    # ---- read (relabel on sample) ----
    def aggregate(self, key: PosKey) -> tuple[float, float]:
        """Return (effective_visits, mean_value) for a key; (0.0, 0.0) if unseen.
        Effective visits = raw_visits * scale; the mean is scale-invariant."""
        cur = self.store.get(key)
        if cur is None:
            return 0.0, 0.0
        raw_visits = float(cur[0])
        if raw_visits <= 0.0:
            return 0.0, 0.0
        return raw_visits * self.scale, float(cur[1]) / raw_visits

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
        # Fold scale into the raw counts before persisting so the side-car is
        # scale-1.0 canonical (a loaded store needs no scale-history context).
        self._renormalize()
        payload = {
            "store": {k: v.tolist() for k, v in self.store.items()},
            "recency_decay": self.recency_decay,
            "min_visits": self.min_visits,
            "max_blend": self.max_blend,
            "compact_every": self.compact_every,
            "compact_min_visits": self.compact_min_visits,
            "scale": self.scale,
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
            compact_every=int(payload.get("compact_every", 64)),
            compact_min_visits=float(payload.get("compact_min_visits", 1.0)),
        )
        ps.store = {
            tuple(k): np.array(v, dtype=np.float64)
            for k, v in payload["store"].items()
        }
        # Older side-cars have no "scale" (counts already effective) -> 1.0.
        ps.scale = float(payload.get("scale", 1.0))
        return ps

    def __len__(self) -> int:
        return len(self.store)
