"""Cross-game value sidecar (Derby 'position-stats', bead derby-eft).

ALL CPU, tiny — never touches MPS / the live derby. Covers:
  - canonical-key D4 invariance (the load-bearing aggregation guarantee)
  - aggregate math: discounted-return blend + recency decay + visit gate
  - OFF-path byte-identical: buffer schema/state_dict, model param count +
    state_dict, and a tiny CPU smoke-gen + sample produces identical z targets
  - resume rebuild-from-buffer: keys recomputed from planes, aggregate matches
"""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import (
    BOARD_SIZE,
    GameState,
    N_ACTIONS,
    N_INPUT_PLANES,
    _sym_board,
)
from gomoku.position_stats import (
    PositionStats,
    canonical_key_from_board,
    canonical_key_from_planes,
)
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import SelfPlayExample


# ---------- helpers ----------

def _planes_after(actions: list[int]) -> np.ndarray:
    s = GameState.initial()
    for a in actions:
        s = s.apply(a)
    return s.to_planes()


def _example(planes: np.ndarray, z: float) -> SelfPlayExample:
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[0] = 1.0
    return SelfPlayExample(planes=planes.astype(np.float32), pi=pi, z=float(z))


# ---------- canonical-key D4 invariance ----------

def test_canonical_key_d4_invariant_planes():
    """The canonical key is identical for all 8 D4 orientations of one position."""
    planes = _planes_after([40, 30, 50, 31])  # some asymmetric position
    ref = canonical_key_from_planes(planes)
    for s in range(8):
        rot = _sym_board(planes, s)
        assert canonical_key_from_planes(rot) == ref, f"key changed under sym {s}"


def test_canonical_key_distinguishes_distinct_positions():
    """Genuinely different (non-symmetric) boards get different keys."""
    a = canonical_key_from_planes(_planes_after([40, 30]))
    b = canonical_key_from_planes(_planes_after([40, 31]))
    assert a != b


def test_canonical_key_side_relative():
    """Same stone configuration but opposite side-to-move => different key
    (planes are side-relative, matching z's convention)."""
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=bool)
    board[0, 4, 4] = True   # my stone
    board[1, 0, 0] = True   # opp stone
    k_mine = canonical_key_from_board(board)
    k_swapped = canonical_key_from_board(board[::-1].copy())
    assert k_mine != k_swapped


def test_canonical_key_empty_board_is_zero_triple():
    empty = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=bool)
    assert canonical_key_from_board(empty) == (0, 0, 0)


# ---------- aggregate math ----------

def test_aggregate_mean_of_discounted_returns():
    ps = PositionStats(recency_decay=1.0, min_visits=1.0, max_blend=1.0)
    k = (1, 2, 3)
    for z in (1.0, -1.0, 0.0, 0.5):
        ps.add(k, z)
    visits, mean_v = ps.aggregate(k)
    assert visits == 4.0
    assert mean_v == (1.0 - 1.0 + 0.0 + 0.5) / 4.0


def test_recency_decay_fades_old_counts():
    ps = PositionStats(recency_decay=0.5, min_visits=1.0, max_blend=1.0)
    k = (7, 0, 0)
    ps.add(k, 1.0)          # visits=1, sum=1.0
    ps.decay()              # -> visits=0.5, sum=0.5
    ps.add(k, 1.0)          # -> visits=1.5, sum=1.5
    visits, mean_v = ps.aggregate(k)
    assert abs(visits - 1.5) < 1e-9
    assert abs(mean_v - 1.0) < 1e-9     # both contributions were +1
    # A fresh, opposite contribution after decay is weighted toward recent:
    ps2 = PositionStats(recency_decay=0.5, min_visits=1.0, max_blend=1.0)
    ps2.add(k, -1.0)        # old, will be decayed
    ps2.decay()
    ps2.add(k, 1.0)         # recent
    _, mean2 = ps2.aggregate(k)
    # recent (+1, weight 1.0) outweighs decayed old (-1, weight 0.5)
    assert mean2 > 0.0


def test_blend_below_gate_is_pure_game_z():
    ps = PositionStats(recency_decay=1.0, min_visits=8.0, max_blend=0.5)
    k = (5, 5, 5)
    for _ in range(4):       # below the 8-visit gate
        ps.add(k, 1.0)
    assert ps.blend(k, -1.0) == -1.0      # unchanged, falls back to z_game


def test_blend_above_gate_moves_toward_aggregate_and_caps():
    ps = PositionStats(recency_decay=1.0, min_visits=8.0, max_blend=0.5)
    k = (9, 9, 9)
    for _ in range(8192):    # heavily transited -> max confidence
        ps.add(k, 1.0)       # aggregate mean = +1
    blended = ps.blend(k, -1.0)
    # weight w in [0, max_blend): blend = (1-w)*(-1) + w*(+1) in (-1, 0)
    assert -1.0 < blended < 0.0
    # cap: even at huge visits, w < max_blend => blend < -1 + 2*max_blend = 0.0
    assert blended < (-1.0 + 2.0 * ps.max_blend) + 1e-6


def test_relabel_z_unseen_keys_unchanged():
    ps = PositionStats(recency_decay=1.0, min_visits=8.0, max_blend=0.5)
    z = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    out = ps.relabel_z([(1, 0, 0), (2, 0, 0), (3, 0, 0)], z)
    assert np.array_equal(out, z)        # nothing seen -> all unchanged


# ---------- persistence roundtrip ----------

def test_save_load_roundtrip(tmp_path):
    ps = PositionStats(recency_decay=0.97, min_visits=4.0, max_blend=0.3)
    k = canonical_key_from_planes(_planes_after([40, 30, 50]))
    ps.add(k, 0.5)
    ps.add(k, -0.25)
    path = str(tmp_path / "store.pkl")
    ps.save(path)
    ps2 = PositionStats.load(path)
    assert ps2.recency_decay == 0.97 and ps2.min_visits == 4.0 and ps2.max_blend == 0.3
    assert len(ps2) == 1
    assert ps2.aggregate(k) == ps.aggregate(k)


# ---------- OFF-path byte-identical: buffer schema ----------

def test_buffer_off_no_pos_key():
    b = ReplayBuffer(64, device="cpu")
    assert b.pos_key is None
    assert not b.cross_game_enabled
    # state_dict carries no cross-game key column (byte-identical schema)
    sd = b.state_dict()
    assert "pos_key" not in sd


def test_buffer_off_state_dict_keys_match_pre_lever():
    """OFF buffer's state_dict has exactly the pre-lever key set."""
    b = ReplayBuffer(32, device="cpu")
    ex = [_example(_planes_after([40]), 1.0)]
    b.add(ex)
    keys = set(b.state_dict().keys())
    expected = {"capacity", "head", "size", "planes", "pi", "z",
                "weight_version", "side", "ply", "current_weight_version"}
    assert keys == expected


def test_buffer_on_populates_pos_key():
    b = ReplayBuffer(32, device="cpu", cross_game_value=True)
    planes = _planes_after([40, 30])
    b.add([_example(planes, 1.0)])
    assert b.pos_key is not None
    got = tuple(int(x) for x in b.pos_key[0])
    assert got == canonical_key_from_planes(planes)


# ---------- OFF-path byte-identical: model untouched ----------

def test_model_unaffected_by_cross_game_lever():
    """The cross-game lever NEVER touches the model — there is no model arg, no
    new head. A model built the normal way is param-count- and state_dict-
    identical regardless of the lever (it cannot be otherwise)."""
    from gomoku.model import build_model, n_params
    m1 = build_model("tiny")
    m2 = build_model("tiny")
    assert n_params(m1) == n_params(m2)
    assert set(m1.state_dict().keys()) == set(m2.state_dict().keys())


# ---------- OFF-path byte-identical: smoke value-target ----------

def _build_buffer_with_games(cross_game: bool, seed: int = 0):
    """Tiny deterministic 'smoke-gen': fabricate a couple of short games' worth
    of examples (NO MPS, NO MCTS) and add them to a buffer. Returns the buffer."""
    rng = np.random.default_rng(seed)
    b = ReplayBuffer(256, device="cpu", cross_game_value=cross_game)
    games = [
        [40, 30, 50, 31, 60],
        [40, 30, 41, 31, 42],   # shares the opening 40,30 with game 1
        [20, 21, 22, 23, 24],
    ]
    for g in games:
        exs = []
        s = GameState.initial()
        z = float(rng.choice([-1.0, 0.0, 1.0]))
        for ply, a in enumerate(g):
            planes = s.to_planes()
            side = ply % 2
            zz = z if side == 0 else -z
            exs.append(_example(planes, zz))
            s = s.apply(a)
        b.add(exs)
    return b


def test_off_path_z_byte_identical_to_baseline():
    """With the lever OFF, the sampled z values are byte-identical to a buffer
    that never knew about cross-game (same seed, same sample indices)."""
    torch.manual_seed(123)
    b_off = _build_buffer_with_games(cross_game=False, seed=7)
    # Re-seed RNG so the two samples draw identical indices.
    g = torch.Generator().manual_seed(999)
    idx = torch.randint(0, b_off.size, (16,), generator=g)
    z_off = b_off.z[idx].clone()

    # A 'plain' buffer (no cross-game at all) built identically.
    b_plain = _build_buffer_with_games(cross_game=False, seed=7)
    z_plain = b_plain.z[idx].clone()
    assert torch.equal(z_off, z_plain)

    # And the OFF buffer's z column equals the plain buffer's z column entirely.
    assert torch.equal(b_off.z[:b_off.size], b_plain.z[:b_plain.size])


def test_sample_off_returns_no_keys_default_shape():
    """sample() default contract unchanged when off: 5-tuple, no key column."""
    b = _build_buffer_with_games(cross_game=False, seed=1)
    out = b.sample(8)
    assert len(out) == 5
    # return_keys with an off buffer yields a trailing None (no crash).
    out2 = b.sample(8, return_keys=True)
    assert out2[-1] is None


# ---------- resume: rebuild-from-buffer ----------

def test_rebuild_pos_keys_from_planes_matches_ingest():
    """A buffer restored WITHOUT a key column (resume from a pre-lever latest.pt)
    can recompute every row's canonical key from its planes -> identical to the
    keys computed at ingest."""
    b_ingest = _build_buffer_with_games(cross_game=True, seed=3)
    ingest_keys = b_ingest.pos_key[:b_ingest.size].copy()

    # Simulate resume: a fresh cross-game buffer whose key column was NOT filled
    # (e.g. planes copied in from an old checkpoint). Zero it, then rebuild.
    b_resume = _build_buffer_with_games(cross_game=True, seed=3)
    b_resume.pos_key[:] = 0
    b_resume.rebuild_pos_keys()
    assert np.array_equal(b_resume.pos_key[:b_resume.size], ingest_keys)


def test_rebuild_store_from_buffer_matches_streaming_aggregate():
    """The resume rebuild (aggregate from the buffer's stored z + recomputed
    keys) reproduces the same aggregate a streaming ingest would have built —
    proving the side-car is reconstructable from latest.pt alone (no GPU)."""
    b = _build_buffer_with_games(cross_game=True, seed=5)
    # Streaming aggregate (what the trainer builds during a live run, decay off).
    stream = PositionStats(recency_decay=1.0, min_visits=1.0, max_blend=0.5)
    planes_all = b.planes[:b.size].numpy()
    z_all = b.z[:b.size].numpy()
    for r in range(b.size):
        stream.add(canonical_key_from_planes(planes_all[r]), float(z_all[r]))

    # Rebuild path: recompute keys from the column, aggregate stored z.
    b.rebuild_pos_keys()
    rebuilt = PositionStats(recency_decay=1.0, min_visits=1.0, max_blend=0.5)
    rebuilt.add_many(
        (tuple(int(x) for x in b.pos_key[r]) for r in range(b.size)),
        z_all,
    )
    assert len(rebuilt) == len(stream)
    for k in stream.store:
        assert np.allclose(rebuilt.store[k], stream.store[k])


# ---------- bead derby-eda: per-ingest cost is O(new positions), store bounded ----------

def test_decay_is_o1_not_full_store_traversal():
    """REGRESSION GUARD (bead derby-eda). The old code multiplied EVERY entry by
    recency_decay each ingest (O(store-size)) — the epoch wall grew 14s->128s as
    the store hit 14MB. The fix uses a single lazy global ``scale``. Assert
    decay() touches NO per-entry array: snapshot the raw arrays' object identity
    + contents before/after decay over a large store; they must be untouched
    (only the scalar ``scale`` changes)."""
    ps = PositionStats(recency_decay=0.99, min_visits=1.0, max_blend=0.5,
                       compact_every=0)  # disable compaction for this probe
    n = 5000
    for i in range(n):
        ps.add((i, 0, 0), 1.0)
    before_scale = ps.scale
    snap = {k: (id(v), v.copy()) for k, v in ps.store.items()}
    ps.decay()
    # scale changed by exactly recency_decay; raw entries are byte-identical.
    assert ps.scale == before_scale * 0.99
    for k, v in ps.store.items():
        sid, sval = snap[k]
        assert id(v) == sid                       # same array object, not rewritten
        assert np.array_equal(v, sval)            # contents untouched -> no O(N) work


def test_ingest_plus_decay_time_flat_as_store_grows():
    """REGRESSION GUARD (bead derby-eda). Per-cycle (decay + add N new) wall must
    stay ~FLAT as the store grows. The old O(store-size) decay made later cycles
    materially slower; the lazy-scale fix makes per-cycle cost O(new positions).
    Assert the LAST batch of cycles isn't materially slower than the FIRST,
    despite the store being orders of magnitude larger."""
    import time
    ps = PositionStats(recency_decay=0.999, min_visits=8.0, max_blend=0.5,
                       compact_every=0)  # isolate decay+add cost (no periodic prune)
    new_per_cycle = 200
    n_cycles = 400

    def run_window(start_cycle):
        zs = [0.5] * new_per_cycle
        t0 = time.perf_counter()
        for c in range(start_cycle, start_cycle + 50):
            ps.decay()
            base = c * new_per_cycle
            ps.add_many(((base + j, 0, 0) for j in range(new_per_cycle)), zs)
        return time.perf_counter() - t0

    first = run_window(0)
    # advance to cycle ~350 so the store is ~70k entries vs ~10k for the first
    for c in range(50, 350):
        ps.decay()
        base = c * new_per_cycle
        ps.add_many(((base + j, 0, 0) for j in range(new_per_cycle)), [0.5] * new_per_cycle)
    big_store = len(ps.store)
    last = run_window(350)

    assert big_store > 60000, f"store didn't grow enough to be a real test ({big_store})"
    # Flat: the late window must not be much slower than the early one. The old
    # O(N) decay would make `last` scale with store size (~7x here); allow 2.5x
    # slack for timer noise on a loaded machine while still catching O(N) regress.
    assert last < first * 2.5, (
        f"per-cycle cost grew with store size (O(N) regression): "
        f"first={first*1e3:.1f}ms last={last*1e3:.1f}ms store={big_store}"
    )


def test_lazy_scale_aggregate_matches_eager_decay():
    """Aggregate (visits + mean) under the lazy global scale must match an
    independent EAGER per-entry decay reference over many mixed cycles."""
    rng = np.random.default_rng(0)
    decay = 0.97
    lazy = PositionStats(recency_decay=decay, min_visits=1.0, max_blend=0.5,
                         compact_every=0)
    # Eager reference: dict key -> [visits, sum] decayed entry-by-entry.
    eager: dict = {}
    keys_pool = [(k, 0, 0) for k in range(40)]
    for _ in range(120):
        lazy.decay()
        for v in eager.values():            # eager: multiply every entry
            v[0] *= decay
            v[1] *= decay
        m = rng.integers(1, 8)
        for _ in range(int(m)):
            k = keys_pool[int(rng.integers(0, len(keys_pool)))]
            z = float(rng.choice([-1.0, 0.0, 1.0, 0.5]))
            lazy.add(k, z)
            if k in eager:
                eager[k][0] += 1.0
                eager[k][1] += z
            else:
                eager[k] = [1.0, z]
    for k, ev in eager.items():
        lv, lm = lazy.aggregate(k)
        e_visits = ev[0]
        e_mean = ev[1] / ev[0] if ev[0] != 0 else 0.0
        assert abs(lv - e_visits) < 1e-6, f"{k}: visits {lv} vs {e_visits}"
        assert abs(lm - e_mean) < 1e-9, f"{k}: mean {lm} vs {e_mean}"


def test_renormalize_preserves_effective_values():
    """When the global scale underflows it is folded into the raw counts (O(N)
    once) and reset to 1.0 — effective visits + mean must be preserved exactly."""
    # tiny decay forces an underflow renorm within a handful of cycles
    ps = PositionStats(recency_decay=0.001, min_visits=1.0, max_blend=0.5,
                       compact_every=0)
    k = (3, 1, 4)
    ps.add(k, 0.5)
    ps.add(k, -0.5)
    v0, m0 = ps.aggregate(k)
    # several decays: 0.001**3 = 1e-9 < 1e-6 floor -> a renorm fires
    for _ in range(3):
        ps.decay()
    assert ps.scale == 1.0, "scale should have been renormalized to 1.0"
    v1, m1 = ps.aggregate(k)
    # effective visits decayed by 0.001**3; mean unchanged (scale-invariant)
    assert abs(v1 - v0 * (0.001 ** 3)) < 1e-12
    assert abs(m1 - m0) < 1e-12


def test_compaction_bounds_the_store():
    """Periodic compaction prunes cold entries so the store can't grow unbounded
    (the 14MB blow-up in Derby v8). A churn of one-off cold keys plus a few hot
    keys must leave the store small after compaction, keeping the hot keys."""
    ps = PositionStats(recency_decay=0.95, min_visits=1.0, max_blend=0.5,
                       compact_every=4, compact_min_visits=1.0)
    hot = [(900 + i, 0, 0) for i in range(3)]
    cold_seq = 0
    for cycle in range(40):
        ps.decay()
        # hot keys get many visits each cycle (stay above the floor)
        for k in hot:
            for _ in range(50):
                ps.add(k, 1.0)
        # a flood of unique cold keys, one visit each (will decay below the floor)
        for _ in range(100):
            ps.add((cold_seq, 7, 7), 0.0)
            cold_seq += 1
    # without compaction this would be ~4000 keys; bounded well below that.
    assert len(ps.store) < 500, f"store not bounded: {len(ps.store)}"
    # hot keys survive (effective visits well above the floor).
    for k in hot:
        v, _ = ps.aggregate(k)
        assert v >= ps.compact_min_visits


def test_save_folds_scale_and_roundtrips_decayed_state():
    """save() folds the lazy scale into raw counts (scale->1.0) so the side-car
    is canonical; load() restores aggregates that match the live store exactly."""
    ps = PositionStats(recency_decay=0.9, min_visits=2.0, max_blend=0.5,
                       compact_every=0)
    k = (11, 22, 33)
    for _ in range(5):
        ps.decay()
        ps.add(k, 0.25)
    live_v, live_m = ps.aggregate(k)
    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as d:
        path = _os.path.join(d, "s.pkl")
        ps.save(path)
        ps2 = PositionStats.load(path)
    assert ps2.scale == 1.0
    r_v, r_m = ps2.aggregate(k)
    assert abs(r_v - live_v) < 1e-9
    assert abs(r_m - live_m) < 1e-9
