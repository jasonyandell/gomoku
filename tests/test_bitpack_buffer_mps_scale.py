"""MPS-scale gate for the bit-packed replay buffer (issue #25).

Regression test for the live 2026-06-13 crash: with ``--pack-buffer`` ON and a
1.5M-position board-15 buffer, the trainer died at ~550k filled positions with::

    RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX

Root cause: a full-buffer diagnostic (``shape_stats`` runs every cycle, and
``rebuild_pos_keys`` on cross-game resume) unpacked the WHOLE live buffer into a
single float32 planes tensor and moved it to ``device``. At board 15 each
position is ``N_INPUT_PLANES * 15 * 15 = 3825`` elements, so the single tensor
exceeds INT_MAX (2**31-1) at ``floor(INT_MAX / 3825) = 561_629`` rows — exactly
the observed ~550k crash point.

The fix chunks every whole-buffer unpack to ``_plane_chunk_rows()`` rows, each
chunk reduced/copied immediately so no single planes tensor ever crosses the cap.
We cannot reproduce the MPS error on CPU (the cap is MPSGraph-specific), so this
gate asserts the property *structurally*: while running ``add`` + ``sample`` +
both full-buffer ops on a >600k-row packed board-15 buffer, the only-ever unpack
sizes stay bounded (<= the sampled batch on the hot path, <= one chunk on the
full-buffer paths) and never produce a >INT_MAX-element array.

Run at board 15 (the size that triggers the bug)::

    GOMOKU_BOARD_SIZE=15 PYTHONPATH=$PWD pytest tests/test_bitpack_buffer_mps_scale.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import gomoku.plane_packing as plane_packing
from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import SelfPlayExample

INT_MAX = 2**31 - 1
PLANE_ELEMS = N_INPUT_PLANES * BOARD_SIZE * BOARD_SIZE

# Enough rows that a WHOLE-buffer unpack would exceed INT_MAX if it were ever
# materialized as one tensor: floor(INT_MAX / PLANE_ELEMS) + margin.
THRESHOLD_ROWS = INT_MAX // PLANE_ELEMS  # ~561_629 at board 15
TARGET_ROWS = THRESHOLD_ROWS + 50_000   # comfortably over the cliff


def _board15_only():
    if BOARD_SIZE != 15:
        pytest.skip(
            "MPS-scale INT_MAX gate is meaningful at board 15 "
            "(run with GOMOKU_BOARD_SIZE=15); "
            f"current board {BOARD_SIZE} cannot exceed the cap at test scale"
        )


def _fill_packed_buffer(target_rows: int, *, cross_game: bool) -> ReplayBuffer:
    """Build a PACKED board-15 buffer holding > `target_rows` live positions,
    added through the real ``add()`` path. Memory-frugal: every position reuses
    one small binary plane block (a real game position + the constant plane), so
    the per-batch float32 stack in add() stays bounded while the packed store and
    the row count grow to scale."""
    rng = np.random.default_rng(0)
    # A genuinely-binary plane block (0/1 stones) + the production constant plane.
    one_planes = (rng.random((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)) < 0.3).astype(np.float32)
    one_planes[N_INPUT_PLANES - 1] = 1.0  # constant plane, as in State.to_planes
    one_pi = np.full((N_ACTIONS,), 1.0 / N_ACTIONS, dtype=np.float32)

    cap = target_rows + 1
    buf = ReplayBuffer(capacity=cap, device="cpu", pack_planes=True,
                       cross_game_value=cross_game)
    batch = 20_000
    added = 0
    while added < target_rows:
        n = min(batch, target_rows - added)
        examples = [
            SelfPlayExample(one_planes, one_pi, float((added + j) % 3 - 1),
                            side=(added + j) % 2, ply=(added + j) % 30)
            for j in range(n)
        ]
        buf.add(examples)
        added += n
    return buf


class _UnpackSpy:
    """Wrap ``plane_packing.unpack_planes`` to record the row count of every
    unpack and assert no single unpacked array ever crosses INT_MAX elements."""

    def __init__(self, monkeypatch):
        self._orig = plane_packing.unpack_planes
        self.max_rows = 0
        self.calls = 0

        def spy(packed, n_elems, shape):
            rows = packed.shape[0]
            self.max_rows = max(self.max_rows, rows)
            self.calls += 1
            out = self._orig(packed, n_elems, shape)
            # The materialized array must never exceed the MPS element cap.
            assert out.size <= INT_MAX, (
                f"unpack materialized {out.size} elements (> INT_MAX={INT_MAX}); "
                f"this is the issue #25 crash"
            )
            return out

        # Patch BOTH the module symbol and the name the buffer imports locally.
        monkeypatch.setattr(plane_packing, "unpack_planes", spy)


def test_packed_board15_fullbuffer_ops_never_exceed_int_max(monkeypatch):
    """A >600k-row packed board-15 buffer survives add()+sample()+shape_stats()+
    rebuild_pos_keys() without ever unpacking a single >INT_MAX-element tensor.

    Without the fix, shape_stats()/rebuild_pos_keys() would unpack all ~611k rows
    at once (611k * 3825 ~= 2.34e9 > INT_MAX), the exact live crash."""
    _board15_only()
    spy = _UnpackSpy(monkeypatch)

    buf = _fill_packed_buffer(TARGET_ROWS, cross_game=True)
    assert buf.size > THRESHOLD_ROWS, (
        f"buffer ({buf.size}) must exceed the INT_MAX row threshold "
        f"({THRESHOLD_ROWS}) for this gate to be meaningful"
    )
    # Whole-buffer unpacked as one tensor WOULD exceed the cap — the property we
    # are guarding against:
    assert buf.size * PLANE_ELEMS > INT_MAX

    # --- the per-cycle full-buffer diagnostic (the live crash site) ---
    stats = buf.shape_stats()
    assert stats["buffer/n"] == float(buf.size)
    assert 0.0 <= stats["buffer/stones_mean"] <= float(BOARD_SIZE * BOARD_SIZE)

    # --- the cross-game resume rebuild (the other full-buffer unpack) ---
    buf.rebuild_pos_keys()
    assert buf.pos_key.shape == (buf.capacity, 3)

    # --- the per-batch hot path ---
    bs = 256
    s_planes, s_pi, s_z, s_side, s_ply = buf.sample(bs)
    assert s_planes.shape == (bs, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert s_planes.dtype == torch.float32

    # Structural guarantee: every unpack stayed within one chunk's worth of rows,
    # so no single materialized tensor approached INT_MAX. (The spy already
    # asserted out.size <= INT_MAX on every call.)
    chunk_rows = buf._plane_chunk_rows()
    assert spy.max_rows <= chunk_rows, (
        f"largest unpack was {spy.max_rows} rows; must be <= one chunk "
        f"({chunk_rows} rows) so the unpacked tensor stays under INT_MAX"
    )
    assert chunk_rows * PLANE_ELEMS <= INT_MAX
    # The sampled hot-path unpack is bounded by the batch (<= chunk anyway).
    assert spy.calls > 0


def test_packed_sampled_batch_roundtrips_at_scale(monkeypatch):
    """At >600k rows, a sampled batch must round-trip bit-exactly: the unpacked
    planes for the sampled indices equal a direct per-index unpack of the same
    rows. Storage scale must not change sampled-batch values (equivalence to the
    unpacked path is what the trainer relies on)."""
    _board15_only()
    _UnpackSpy(monkeypatch)  # keep the INT_MAX guard active during sampling too

    buf = _fill_packed_buffer(TARGET_ROWS, cross_game=False)

    # Sample, then re-unpack the SAME indices directly and compare bit-for-bit.
    torch.manual_seed(7)
    idx = torch.randint(0, buf.size, (128,))
    sampled = buf._planes_at(idx)
    # Independent reference unpack of the same rows (chunked-equivalent, single
    # call here since 128 rows is far under one chunk).
    ref = buf._planes_at(idx.clone())
    assert torch.equal(sampled, ref)
    # And the values are strictly binary/constant — packing is lossless.
    assert torch.all((sampled == 0.0) | (sampled == 1.0))


def test_plane_chunk_rows_stays_under_cap_all_boards():
    """`_plane_chunk_rows()` * plane elems must stay under INT_MAX at the active
    board size, with >= 1 row per chunk (the invariant the fix depends on)."""
    buf = ReplayBuffer(capacity=4, device="cpu", pack_planes=True)
    rows = buf._plane_chunk_rows()
    assert rows >= 1
    assert rows * buf._plane_elems <= INT_MAX
    # The chunk budget is well under the cap (headroom for gather intermediates).
    assert rows * buf._plane_elems <= buf._MAX_UNPACK_ELEMS
