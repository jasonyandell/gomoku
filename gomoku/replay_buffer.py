"""Ring buffer of self-play examples backed by preallocated tensors."""

from __future__ import annotations

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES
from gomoku.self_play import SelfPlayExample


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device | str = "cpu",
        *,
        aux_opponent_reply: bool = False,
        aux_ownership: bool = False,
        aux_vct: bool = False,
        cross_game_value: bool = False,
        pack_planes: bool = False,
    ):
        self.capacity = capacity
        self.device = torch.device(device)
        # Per-position plane geometry, fixed at the active board size.
        self._plane_shape = (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
        self._plane_elems = N_INPUT_PLANES * BOARD_SIZE * BOARD_SIZE
        # OPT-IN bit-packed plane storage (issue #25). When ON, the dominant
        # float32 planes tensor (4 * N_INPUT_PLANES * N^2 bytes/pos) is replaced
        # by a uint8 bit-packed array (ceil(N_INPUT_PLANES * N^2 / 8) bytes/pos,
        # ~32x smaller) kept on the CPU; sample() unpacks per-batch back to
        # float32 planes on `device`. When OFF (default) NOTHING below changes:
        # `self.planes` is the same float32 tensor on `self.device` as before, so
        # the off-case buffer (RAM, schema, sample math, checkpoints) is
        # byte-identical to the pre-#25 path. Production planes are strictly
        # binary occupancy (0/1) + a constant plane, so packing is lossless.
        self.pack_planes = bool(pack_planes)
        if self.pack_planes:
            from gomoku.plane_packing import packed_bytes_per_position
            self._packed_bytes = packed_bytes_per_position(self._plane_elems)
            # Packed store lives on the CPU as a torch uint8 tensor: MPS cannot
            # hold the wide buffers this lever targets (INT_MAX element cap), and
            # the unpack→to(device) per batch is the intended hot-path transfer.
            self.packed_planes = torch.zeros(
                (capacity, self._packed_bytes), dtype=torch.uint8, device="cpu",
            )
            self.planes = None
        else:
            self._packed_bytes = 0
            self.packed_planes = None
            self.planes = torch.zeros(
                (capacity, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE),
                dtype=torch.float32, device=self.device,
            )
        self.pi = torch.zeros((capacity, N_ACTIONS), dtype=torch.float32, device=self.device)
        self.z = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        # Weight-version tag: which "brain" generated the position in each slot.
        # Used for stratified sampling and shape diagnostics. Auto-overwritten
        # when the ring buffer reuses a slot — no separate deletion needed.
        self.weight_version = torch.zeros((capacity,), dtype=torch.int64, device=self.device)
        # Side-to-move (0=black, 1=white) and ply count at the position. Used by
        # the WL5 diagnostics layer for per-color and per-ply-bucket metrics.
        self.side = torch.zeros((capacity,), dtype=torch.int8, device=self.device)
        self.ply = torch.zeros((capacity,), dtype=torch.int16, device=self.device)
        # V3 aux opponent-reply target + validity mask. LAZY-ALLOCATED: when the
        # lever is off (default) these tensors are NEVER allocated, so off-case
        # buffer RAM is byte-identical to a buffer that never knew about aux.
        # `aux_pi` holds the opponent's next-ply policy (zeros where undefined);
        # `aux_mask` is True for rows that carry a valid aux target (last-ply
        # positions and aux-off positions are False → excluded from the aux loss).
        self.aux_enabled = bool(aux_opponent_reply)
        if self.aux_enabled:
            self.aux_pi = torch.zeros(
                (capacity, N_ACTIONS), dtype=torch.float32, device=self.device
            )
            self.aux_mask = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
        else:
            self.aux_pi = None
            self.aux_mask = None
        # V4 aux ownership target + validity mask. LAZY-ALLOCATED exactly like
        # the opponent-reply tensors: off (default) => never allocated, so the
        # off-case buffer RAM/schema is byte-identical. `ownership` holds the
        # per-cell final-control target in [-1, 1] (zeros where undefined);
        # `ownership_mask` is True for rows carrying a valid target (draws still
        # carry a valid all-zeros target → mask True; ownership-off positions
        # are False → excluded from the ownership loss).
        self.ownership_enabled = bool(aux_ownership)
        if self.ownership_enabled:
            self.ownership = torch.zeros(
                (capacity, N_ACTIONS), dtype=torch.float32, device=self.device
            )
            self.ownership_mask = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
        else:
            self.ownership = None
            self.ownership_mask = None
        # Moonshot VCT-defense target + validity mask. LAZY-ALLOCATED exactly like
        # the ownership tensors: off (default) => never allocated, so the off-case
        # buffer RAM/schema is byte-identical. `vct` holds the per-cell 0/1
        # "blunder map" (1.0 where the side to move playing there walks into a
        # forced VCT for the opponent; zeros where undefined); `vct_mask` is True
        # for rows carrying a valid target (vct-off / unlabeled positions are False
        # => excluded from the masked-BCE vct loss).
        self.vct_enabled = bool(aux_vct)
        if self.vct_enabled:
            self.vct = torch.zeros(
                (capacity, N_ACTIONS), dtype=torch.float32, device=self.device
            )
            self.vct_mask = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
        else:
            self.vct = None
            self.vct_mask = None
        # Cross-game value sidecar (Derby 'position-stats'). LAZY-ALLOCATED like
        # the aux tensors: off (default) => NEVER allocated, so the off-case
        # buffer RAM/schema is byte-identical. When on, each row carries its
        # canonical D4-invariant position key as a 3-column uint64 array
        # (hi, lo, top) computed ONCE at ingest. Kept on the CPU as numpy (the
        # relabel-on-sample read path is a CPU dict lookup) — ~36MB at 1.5M rows.
        self.cross_game_enabled = bool(cross_game_value)
        if self.cross_game_enabled:
            self.pos_key = np.zeros((capacity, 3), dtype=np.uint64)
        else:
            self.pos_key = None
        self.current_weight_version: int = 0
        self.head = 0
        self.size = 0
        # Curated sampling (Derby v7 'buffer-composition'). 0.0 = uniform (default,
        # byte-identical). >0 = draw that fraction of each batch from the most-recent
        # `_recency_window` positions (the ring's tail before `head`), rest uniform —
        # over-sampling fresh positions from the stronger net. Set via configure_curator().
        self._recency_frac = 0.0
        self._recency_window = 200_000

    def configure_curator(self, recency_frac: float = 0.0,
                          recency_window: int = 200_000) -> None:
        """Set the recency-mix curator. recency_frac in [0,1] (0 = uniform); the
        recent slice is the last `recency_window` added positions."""
        self._recency_frac = max(0.0, min(1.0, float(recency_frac)))
        self._recency_window = max(1, int(recency_window))

    def _planes_at(self, idx) -> torch.Tensor:
        """Return float32 planes for the given row indices on `self.device`.

        The single read path for planes regardless of storage mode: a plain
        gather of the float32 tensor when packing is OFF, or an unpack of the
        bit-packed CPU store when packing is ON. `idx` may be a 1-D LongTensor or
        anything torch indexing accepts.
        """
        if not self.pack_planes:
            return self.planes[idx]
        from gomoku.plane_packing import unpack_planes
        idx_cpu = idx.detach().cpu() if torch.is_tensor(idx) else idx
        packed = self.packed_planes[idx_cpu].numpy()
        planes_np = unpack_planes(packed, self._plane_elems, self._plane_shape)
        return torch.from_numpy(planes_np).to(self.device)

    # Max planes-tensor elements we will ever materialize in one unpack/reshape.
    # MPSGraph rejects tensors with > INT_MAX (2**31-1) elements; at board 15 a
    # whole-buffer unpack of a 1.5M-row packed store is ~5.7e9 elems, far past
    # that (the ~550k-row crash in issue #25 was exactly this: 550k * 17*15*15 >
    # INT_MAX). Diagnostic/rebuild read paths chunk to stay strictly under this,
    # so a full-buffer op never builds a single oversized tensor regardless of
    # buffer size or board. Budget << INT_MAX leaves headroom for the unpacked
    # path's own gather intermediates and keeps each GPU chunk small.
    _MAX_UNPACK_ELEMS = 1 << 28  # 268,435,456 — ~70 MB float32 per chunk

    def _plane_chunk_rows(self) -> int:
        """Rows per chunk such that one unpacked chunk stays under the MPS
        INT_MAX element cap. Always >= 1 (a single row is far under the cap)."""
        return max(1, self._MAX_UNPACK_ELEMS // max(1, self._plane_elems))

    def _iter_planes_chunks(self):
        """Yield (start, planes_chunk) over the first `self.size` live rows, each
        chunk a float32 tensor on `device` of at most `_plane_chunk_rows()` rows.

        The chunked replacement for the old whole-buffer `_planes_view()`: a
        full-buffer diagnostic (stone counts, key rebuild) consumes chunks and
        reduces/copies each immediately, so NO single tensor ever exceeds the MPS
        INT_MAX element cap regardless of buffer size or board. Under packing
        each chunk is a small per-chunk unpack; when packing is OFF it is a plain
        gather of `self.planes` (same values as the old `self.planes[:size]`)."""
        if self.size == 0:
            return
        step = self._plane_chunk_rows()
        for start in range(0, self.size, step):
            end = min(start + step, self.size)
            yield start, self._planes_at(torch.arange(start, end))

    def _sample_indices(self, batch_size: int) -> torch.Tensor:
        """Pick batch indices. Uniform unless the recency curator is on (and the
        buffer has grown past the window), in which case a recency_frac slice is
        drawn from the most-recent `recency_window` positions, the rest uniform."""
        if self._recency_frac <= 0.0 or self.size < self._recency_window:
            return torch.randint(0, self.size, (batch_size,), device=self.device)
        n_recent = int(round(batch_size * self._recency_frac))
        window = min(self._recency_window, self.size)
        off = torch.randint(0, window, (n_recent,), device=self.device)
        recent = (self.head - 1 - off) % self.size          # tail of the ring (most recent)
        unif = torch.randint(0, self.size, (batch_size - n_recent,), device=self.device)
        return torch.cat([recent, unif])

    def set_weight_version(self, version: int) -> None:
        """Set the weight-version tag applied to subsequent add() calls."""
        self.current_weight_version = int(version)

    def add(self, examples: list[SelfPlayExample]) -> None:
        if not examples:
            return
        n = len(examples)
        planes_np = np.stack([e.planes for e in examples])
        if self.pack_planes:
            # Bit-pack the binary planes once for the whole add; the ring loop
            # below copies uint8 rows. Packing here (CPU numpy) keeps ingest
            # cheap and the per-batch unpack on the sample side.
            from gomoku.plane_packing import pack_planes
            packed = torch.from_numpy(pack_planes(planes_np, self._plane_elems))
        else:
            planes = torch.from_numpy(planes_np)
        pi = torch.from_numpy(np.stack([e.pi for e in examples]))
        z = torch.from_numpy(np.array([e.z for e in examples], dtype=np.float32))
        side = torch.from_numpy(np.array([getattr(e, "side", 0) for e in examples], dtype=np.int8))
        ply = torch.from_numpy(np.array([getattr(e, "ply", 0) for e in examples], dtype=np.int16))
        if self.aux_enabled:
            # Build a dense aux-target tensor + validity mask for this add.
            # examples whose aux_pi is None (last ply / aux-undefined) get a
            # zero target and mask=False, so they never contribute aux gradient.
            aux_np = np.zeros((n, N_ACTIONS), dtype=np.float32)
            aux_mask_np = np.zeros((n,), dtype=bool)
            for j, e in enumerate(examples):
                ap = getattr(e, "aux_pi", None)
                if ap is not None:
                    aux_np[j] = ap
                    aux_mask_np[j] = True
            aux_pi_t = torch.from_numpy(aux_np)
            aux_mask_t = torch.from_numpy(aux_mask_np)
        if self.ownership_enabled:
            # Build a dense ownership-target tensor + validity mask for this add.
            # examples whose ownership is None (ownership-undefined) get a zero
            # target and mask=False, so they never contribute ownership gradient.
            own_np = np.zeros((n, N_ACTIONS), dtype=np.float32)
            own_mask_np = np.zeros((n,), dtype=bool)
            for j, e in enumerate(examples):
                ow = getattr(e, "ownership", None)
                if ow is not None:
                    own_np[j] = ow
                    own_mask_np[j] = True
            own_t = torch.from_numpy(own_np)
            own_mask_t = torch.from_numpy(own_mask_np)
        if self.vct_enabled:
            # Build a dense vct-defense-target tensor + validity mask for this add.
            # examples whose vct is None (vct-undefined / unlabeled) get a zero
            # target and mask=False, so they never contribute vct gradient.
            vct_np = np.zeros((n, N_ACTIONS), dtype=np.float32)
            vct_mask_np = np.zeros((n,), dtype=bool)
            for j, e in enumerate(examples):
                vc = getattr(e, "vct", None)
                if vc is not None:
                    vct_np[j] = vc
                    vct_mask_np[j] = True
            vct_t = torch.from_numpy(vct_np)
            vct_mask_t = torch.from_numpy(vct_mask_np)
        if self.cross_game_enabled:
            # Compute each example's canonical D4-invariant key ONCE, here at
            # ingest, from the row's current-frame occupancy.
            from gomoku.position_stats import canonical_key_from_planes
            keys_np = np.zeros((n, 3), dtype=np.uint64)
            for j, e in enumerate(examples):
                hi, lo, top = canonical_key_from_planes(e.planes)
                keys_np[j, 0] = np.uint64(hi)
                keys_np[j, 1] = np.uint64(lo)
                keys_np[j, 2] = np.uint64(top)
        ver = int(self.current_weight_version)
        i = 0
        while i < n:
            end = min(self.head + (n - i), self.capacity)
            chunk = end - self.head
            if self.pack_planes:
                self.packed_planes[self.head:end].copy_(packed[i:i + chunk])
            else:
                self.planes[self.head:end].copy_(planes[i:i + chunk].to(self.device))
            self.pi[self.head:end].copy_(pi[i:i + chunk].to(self.device))
            self.z[self.head:end].copy_(z[i:i + chunk].to(self.device))
            self.weight_version[self.head:end] = ver
            self.side[self.head:end].copy_(side[i:i + chunk].to(self.device))
            self.ply[self.head:end].copy_(ply[i:i + chunk].to(self.device))
            if self.cross_game_enabled:
                self.pos_key[self.head:end] = keys_np[i:i + chunk]
            if self.aux_enabled:
                self.aux_pi[self.head:end].copy_(aux_pi_t[i:i + chunk].to(self.device))
                self.aux_mask[self.head:end].copy_(aux_mask_t[i:i + chunk].to(self.device))
            if self.ownership_enabled:
                self.ownership[self.head:end].copy_(own_t[i:i + chunk].to(self.device))
                self.ownership_mask[self.head:end].copy_(own_mask_t[i:i + chunk].to(self.device))
            if self.vct_enabled:
                self.vct[self.head:end].copy_(vct_t[i:i + chunk].to(self.device))
                self.vct_mask[self.head:end].copy_(vct_mask_t[i:i + chunk].to(self.device))
            i += chunk
            self.head = end % self.capacity
            self.size = min(self.size + chunk, self.capacity)

    def sample(
        self,
        batch_size: int,
        *,
        return_aux: bool = False,
        return_ownership: bool = False,
        return_vct: bool = False,
        return_keys: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if self.size == 0:
            raise ValueError("empty replay buffer")
        idx = self._sample_indices(batch_size)
        # `_planes_at` is `self.planes[idx]` when packing is OFF (byte-identical)
        # and the unpack-on-sample path when ON; both land float32 on `device`.
        base = (self._planes_at(idx), self.pi[idx], self.z[idx], self.side[idx], self.ply[idx])
        if not return_aux and not return_ownership and not return_vct and not return_keys:
            # Default 5-tuple — byte-identical to the pre-aux signature so every
            # existing caller is untouched.
            return base
        out = base
        # Aux path: append (aux_pi, aux_mask). If the buffer was built without
        # the aux tensors (lever off), return a zero target + all-False mask so
        # the trainer's masked aux loss is a no-op and the call never errors.
        if return_aux:
            if self.aux_enabled:
                out = out + (self.aux_pi[idx], self.aux_mask[idx])
            else:
                zero_aux = torch.zeros(
                    (batch_size, N_ACTIONS), dtype=torch.float32, device=self.device
                )
                false_mask = torch.zeros(
                    (batch_size,), dtype=torch.bool, device=self.device
                )
                out = out + (zero_aux, false_mask)
        # Ownership path: append (ownership, ownership_mask) under the same
        # off-buffer fallback contract. Order is always [aux..., ownership...].
        if return_ownership:
            if self.ownership_enabled:
                out = out + (self.ownership[idx], self.ownership_mask[idx])
            else:
                zero_own = torch.zeros(
                    (batch_size, N_ACTIONS), dtype=torch.float32, device=self.device
                )
                false_mask = torch.zeros(
                    (batch_size,), dtype=torch.bool, device=self.device
                )
                out = out + (zero_own, false_mask)
        # VCT-defense path: append (vct, vct_mask) under the same off-buffer
        # fallback contract. Order is always [aux..., ownership..., vct...] and
        # BEFORE the cross-game keys slot below.
        if return_vct:
            if self.vct_enabled:
                out = out + (self.vct[idx], self.vct_mask[idx])
            else:
                zero_vct = torch.zeros(
                    (batch_size, N_ACTIONS), dtype=torch.float32, device=self.device
                )
                false_mask = torch.zeros(
                    (batch_size,), dtype=torch.bool, device=self.device
                )
                out = out + (zero_vct, false_mask)
        # Cross-game keys path: append the sampled rows' canonical (B, 3) uint64
        # key array (numpy, CPU — the relabel read path is a CPU dict lookup).
        # When the lever is off, append None so the caller can no-op cleanly.
        if return_keys:
            if self.cross_game_enabled:
                idx_cpu = idx.detach().cpu().numpy()
                out = out + (self.pos_key[idx_cpu],)
            else:
                out = out + (None,)
        return out

    def rebuild_pos_keys(self) -> None:
        """(Re)compute the canonical key column for ALL live rows from their
        stored planes. Used on resume when the buffer was restored from a
        latest.pt that predates this lever (no key column embedded) — the keys
        are deterministically recomputable, so the side-car aggregate can be
        rebuilt without any GPU work (~2 min for a 1.5M buffer). No-op when the
        lever is off."""
        if not self.cross_game_enabled or self.size == 0:
            return
        from gomoku.position_stats import canonical_key_from_planes
        # Chunked: unpack at most `_plane_chunk_rows()` rows at a time and reduce
        # each chunk to its per-row keys before moving on, so the whole-buffer
        # rebuild never materializes a single >INT_MAX-element tensor (issue #25).
        for start, chunk in self._iter_planes_chunks():
            planes_cpu = chunk.detach().cpu().numpy()
            for j in range(planes_cpu.shape[0]):
                r = start + j
                hi, lo, top = canonical_key_from_planes(planes_cpu[j])
                self.pos_key[r, 0] = np.uint64(hi)
                self.pos_key[r, 1] = np.uint64(lo)
                self.pos_key[r, 2] = np.uint64(top)

    def shape_stats(self, stone_buckets: tuple[int, ...] = (0, 5, 10, 15, 20, 30, 40, 60, 81)
                    ) -> dict[str, float]:
        """Snapshot of the buffer's *shape* — distribution over position types.

        Returns a dict of named scalars for wandb. Includes:

          buffer/n            : current buffer size
          buffer/stones_bucket_{lo}_{hi}: fraction of positions whose total
                                          stone count is in [lo, hi)
          buffer/z_wins       : fraction with z == +1
          buffer/z_losses     : fraction with z == -1
          buffer/z_draws      : fraction with z == 0

        n_stones is computed as the sum of planes 0 (current side) and the
        opponent-current-frame plane (index = N_INPUT_PLANES//2). That gives
        the total stones on the board for each position in the buffer.
        """
        if self.size == 0:
            return {}
        from gomoku.game import N_INPUT_PLANES
        # current-side me-plane = 0, current-side opp-plane = N_INPUT_PLANES // 2
        opp_idx = N_INPUT_PLANES // 2
        # Sum stones per position into a (size,) array, reducing each unpacked
        # chunk to its per-row counts immediately. Chunking keeps the per-cycle
        # diagnostic from materializing a single >INT_MAX-element planes tensor
        # on MPS (issue #25 — the ~550k-row crash was here, called every cycle).
        stones_cpu = np.empty((self.size,), dtype=np.float32)
        for start, chunk in self._iter_planes_chunks():
            s = (chunk[:, 0].sum(dim=(-2, -1)) + chunk[:, opp_idx].sum(dim=(-2, -1)))
            stones_cpu[start:start + s.shape[0]] = s.detach().cpu().numpy()
        z = self.z[:self.size].detach().cpu().numpy()
        out: dict[str, float] = {
            "buffer/n": float(self.size),
            "buffer/z_wins": float((z > 0).mean()),
            "buffer/z_losses": float((z < 0).mean()),
            "buffer/z_draws": float((z == 0).mean()),
            "buffer/stones_mean": float(stones_cpu.mean()),
        }
        # Bucket histogram
        edges = list(stone_buckets) + [10**9]
        for lo, hi in zip(edges[:-1], edges[1:]):
            label = f"buffer/stones_{lo:02d}_{min(hi, 81):02d}"
            mask = (stones_cpu >= lo) & (stones_cpu < hi)
            out[label] = float(mask.mean())
        # Weight-version freshness: how stale the buffer is relative to the
        # current "brain." 0 = all positions tagged with the latest version.
        if self.current_weight_version > 0:
            ver = self.weight_version[:self.size].detach().cpu().numpy()
            age = self.current_weight_version - ver
            out["buffer/age_mean"] = float(age.mean())
            out["buffer/age_p50"] = float(np.percentile(age, 50))
            out["buffer/age_p90"] = float(np.percentile(age, 90))
            out["buffer/frac_current"] = float((age == 0).mean())
        return out

    def _load_planes(self, sd: dict, n: int) -> None:
        """Load the first `n` rows of planes from a checkpoint into this buffer,
        cross-converting between the float32 ("planes") and bit-packed
        ("packed_planes") storage forms as needed. Any save loads into any mode:
          f32 ckpt → f32 buffer : direct copy (the pre-#25 byte-identical path).
          f32 ckpt → packed buf : bit-pack on load.
          packed ckpt → packed buf : direct copy.
          packed ckpt → f32 buf : unpack on load.
        Existing (pre-#25) checkpoints only ever carry "planes", so they restore
        unchanged into the default (OFF) buffer."""
        has_packed = "packed_planes" in sd
        if self.pack_planes:
            if has_packed:
                self.packed_planes[:n].copy_(sd["packed_planes"][:n])
            else:
                from gomoku.plane_packing import pack_planes
                planes_np = np.ascontiguousarray(sd["planes"][:n].cpu().numpy())
                packed = pack_planes(planes_np, self._plane_elems)
                self.packed_planes[:n].copy_(torch.from_numpy(packed))
        else:
            if has_packed:
                from gomoku.plane_packing import unpack_planes
                packed_np = np.ascontiguousarray(sd["packed_planes"][:n].cpu().numpy())
                planes_np = unpack_planes(packed_np, self._plane_elems, self._plane_shape)
                self.planes[:n].copy_(torch.from_numpy(planes_np).to(self.device))
            else:
                self.planes[:n].copy_(sd["planes"][:n].to(self.device))

    def state_dict(self) -> dict:
        sd = {
            "capacity": self.capacity,
            "head": self.head,
            "size": self.size,
            "pi": self.pi[:self.size].cpu(),
            "z": self.z[:self.size].cpu(),
            "weight_version": self.weight_version[:self.size].cpu(),
            "side": self.side[:self.size].cpu(),
            "ply": self.ply[:self.size].cpu(),
            "current_weight_version": int(self.current_weight_version),
        }
        # Planes: when packing is OFF emit exactly the pre-#25 "planes" float32
        # key (no marker, no extra keys) so an off-buffer checkpoint is
        # byte-identical to the old schema. When ON emit the compact uint8
        # "packed_planes" array plus a marker; load_state_dict cross-converts
        # either form into either buffer mode.
        if self.pack_planes:
            sd["packed_planes"] = self.packed_planes[:self.size].cpu()
            sd["pack_planes"] = True
        else:
            sd["planes"] = self.planes[:self.size].cpu()
        # Only emit aux tensors when the lever is on, so an off-buffer's
        # checkpoint is byte-identical to the pre-aux schema.
        if self.aux_enabled:
            sd["aux_pi"] = self.aux_pi[:self.size].cpu()
            sd["aux_mask"] = self.aux_mask[:self.size].cpu()
        if self.ownership_enabled:
            sd["ownership"] = self.ownership[:self.size].cpu()
            sd["ownership_mask"] = self.ownership_mask[:self.size].cpu()
        if self.vct_enabled:
            sd["vct"] = self.vct[:self.size].cpu()
            sd["vct_mask"] = self.vct_mask[:self.size].cpu()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        if sd["capacity"] != self.capacity:
            # Tolerate capacity change — just load what fits.
            pass
        n = min(int(sd["size"]), self.capacity)
        if n > 0:
            self._load_planes(sd, n)
            self.pi[:n].copy_(sd["pi"][:n].to(self.device))
            self.z[:n].copy_(sd["z"][:n].to(self.device))
            if "weight_version" in sd:
                self.weight_version[:n].copy_(sd["weight_version"][:n].to(self.device))
            if "side" in sd:
                self.side[:n].copy_(sd["side"][:n].to(self.device))
            else:
                self.side[:n].zero_()
                print(f"replay buffer: side tag missing from checkpoint, zero-filled {n} slots")
            if "ply" in sd:
                self.ply[:n].copy_(sd["ply"][:n].to(self.device))
            else:
                self.ply[:n].zero_()
                print(f"replay buffer: ply tag missing from checkpoint, zero-filled {n} slots")
            # Aux target tolerate-missing: when this buffer has the aux head on
            # but the checkpoint predates it (or was off), zero-fill the target
            # and set mask=False so old positions contribute no aux gradient
            # until they evict — mirrors the side/ply missing-tag handling above.
            if self.aux_enabled:
                if "aux_pi" in sd and "aux_mask" in sd:
                    self.aux_pi[:n].copy_(sd["aux_pi"][:n].to(self.device))
                    self.aux_mask[:n].copy_(sd["aux_mask"][:n].to(self.device))
                else:
                    self.aux_pi[:n].zero_()
                    self.aux_mask[:n].zero_()
                    print(f"replay buffer: aux target missing from checkpoint, "
                          f"zero-filled + masked-off {n} slots")
            # Ownership tolerate-missing: same contract as aux above — old/off
            # checkpoints contribute no ownership gradient until they evict.
            if self.ownership_enabled:
                if "ownership" in sd and "ownership_mask" in sd:
                    self.ownership[:n].copy_(sd["ownership"][:n].to(self.device))
                    self.ownership_mask[:n].copy_(sd["ownership_mask"][:n].to(self.device))
                else:
                    self.ownership[:n].zero_()
                    self.ownership_mask[:n].zero_()
                    print(f"replay buffer: ownership target missing from checkpoint, "
                          f"zero-filled + masked-off {n} slots")
            # VCT-defense tolerate-missing: same contract as ownership above — old/
            # off checkpoints contribute no vct gradient until they evict.
            if self.vct_enabled:
                if "vct" in sd and "vct_mask" in sd:
                    self.vct[:n].copy_(sd["vct"][:n].to(self.device))
                    self.vct_mask[:n].copy_(sd["vct_mask"][:n].to(self.device))
                else:
                    self.vct[:n].zero_()
                    self.vct_mask[:n].zero_()
                    print(f"replay buffer: vct target missing from checkpoint, "
                          f"zero-filled + masked-off {n} slots")
        if "current_weight_version" in sd:
            self.current_weight_version = int(sd["current_weight_version"])
        self.size = n
        self.head = n % self.capacity


class ChoiceBuffer:
    """Tiny ring buffer of swap2 NEGOTIATION-choice examples (v2a choice head).

    SEPARATE from :class:`ReplayBuffer` — that buffer is shaped for the policy
    width (N_ACTIONS) and the D4-augmented self-play stream, while choice
    examples are N_CHOICES-wide, NOT D4-augmented, and ~1-2 per swap2 game. The
    target is OUTCOME-driven (built in the trainer from `chosen` + `chooser_z`);
    this buffer only carries the raw fields. Constructed ONLY when the choice
    lever is on (--choice-head-weight > 0), so the off-case path never allocates
    or touches it.
    """

    def __init__(
        self,
        capacity: int = 20_000,
        device: torch.device | str = "cpu",
    ):
        from gomoku.swap2 import N_CHOICES

        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.n_choices = int(N_CHOICES)
        self._plane_shape = (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
        self.planes = torch.zeros(
            (self.capacity, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE),
            dtype=torch.float32, device=self.device,
        )
        self.legal_mask = torch.zeros(
            (self.capacity, self.n_choices), dtype=torch.bool, device=self.device
        )
        self.chosen = torch.zeros((self.capacity,), dtype=torch.int64, device=self.device)
        self.chooser_z = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.head = 0
        self.size = 0

    def add(self, examples: list) -> None:
        """Add a list of :class:`~gomoku.self_play.ChoiceExample` (empty → no-op)."""
        if not examples:
            return
        n = len(examples)
        planes = torch.from_numpy(np.stack([e.planes for e in examples])).to(torch.float32)
        legal = torch.from_numpy(
            np.stack([np.asarray(e.legal_mask, dtype=bool) for e in examples])
        )
        chosen = torch.from_numpy(np.array([int(e.chosen) for e in examples], dtype=np.int64))
        cz = torch.from_numpy(np.array([float(e.chooser_z) for e in examples], dtype=np.float32))
        i = 0
        while i < n:
            end = min(self.head + (n - i), self.capacity)
            chunk = end - self.head
            self.planes[self.head:end].copy_(planes[i:i + chunk].to(self.device))
            self.legal_mask[self.head:end].copy_(legal[i:i + chunk].to(self.device))
            self.chosen[self.head:end].copy_(chosen[i:i + chunk].to(self.device))
            self.chooser_z[self.head:end].copy_(cz[i:i + chunk].to(self.device))
            i += chunk
            self.head = end % self.capacity
            self.size = min(self.size + chunk, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Uniformly sample `(planes, legal_mask, chosen, chooser_z)` on `device`."""
        if self.size == 0:
            raise ValueError("empty choice buffer")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.planes[idx],
            self.legal_mask[idx],
            self.chosen[idx],
            self.chooser_z[idx],
        )
