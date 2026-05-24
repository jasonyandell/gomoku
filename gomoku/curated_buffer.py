"""Curated replay buffer: retain-all on-disk archive + a curated in-RAM slice.

This is "Layer 1" of wiki/topics/curated-buffer-and-curriculum-design.md — the
*mechanism* that decouples generation from training. See also
wiki/topics/perf-bench-vs-real-training-cost.md for the runaway it fixes (a
rolling-FIFO buffer + ``sgd_per_position`` makes SGD work chase generation
rate, with no fixed point).

Two pieces:

1.  ``PositionArchive`` — an append-only, *retain-everything* store on disk.
    Positions are packed with EXACTLY the tensor layout the rest of the repo
    already uses (planes / pi / z, see ``gomoku.replay_buffer.ReplayBuffer`` and
    the ``SelfPlayExample`` dataclass in ``gomoku.self_play``). Per-position
    metadata (``model_version, plies, z, color, source``) rides alongside in
    int tensors. Sharded so appends never rewrite the whole file. Cheap random
    indexed reads of a metadata-selected subset.

2.  ``CuratedBuffer`` — wraps an archive + a curator. The curator selects a slice
    of archive indices; the buffer materializes those positions into RAM tensors
    and serves ``sample(batch_size)`` exactly like ``ReplayBuffer`` (same
    5-tuple return). The slice is refilled at EPOCH / CYCLE boundaries via
    ``refill(cycle)`` — NOT per-minibatch — so the SGD hot path is unchanged.

The point: generation rate and SGD budget become orthogonal knobs. A frozen
archive can be replayed at train-only cost (~constant per epoch) because there
is no self-play in the loop and the per-epoch SGD step count is fixed by the
caller, not by inflow.

Curator registry (Layer 2 lives here too, kept minimal on purpose):
  - ``lru``               : most-recent-N positions by archive order (the boring
                            control — equivalent to a rolling FIFO of width N).
  - ``recency_weighted``  : sample weighted toward recent model_versions.

Exotic recipes (diversity, hard-example pinning, source-weighting, external
curricula) plug in via ``register_curator``; ``source`` is carried as a
metadata field so an external-curriculum ingest can tag positions without any
change here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES

# ---------------------------------------------------------------------------
# Source-tag vocabulary. `source` is an int8 in the archive (compact, sortable);
# this maps the int back to a human label for logging. The mechanism does not
# care about the labels — they exist so a future external-curriculum ingest can
# tag positions (e.g. a Gomocup parser would append with SOURCE_GOMOCUP) without
# touching this module. Layer-1 only ever writes SOURCE_SELFPLAY / SOURCE_ARCHIVE.
# ---------------------------------------------------------------------------
SOURCE_SELFPLAY = 0
SOURCE_ARCHIVE = 1          # WL5-style trouble/validation-archive positions
SOURCE_OPPONENT_ZOO = 2     # past-checkpoint / opponent-zoo games
SOURCE_GOMOCUP = 3          # external champion games (needs a parser — not built)
SOURCE_OPENING_BOOK = 4     # opening-book / progressive-opening curricula

SOURCE_LABELS = {
    SOURCE_SELFPLAY: "selfplay",
    SOURCE_ARCHIVE: "archive",
    SOURCE_OPPONENT_ZOO: "opponent_zoo",
    SOURCE_GOMOCUP: "gomocup",
    SOURCE_OPENING_BOOK: "opening_book",
}


def source_label(code: int) -> str:
    return SOURCE_LABELS.get(int(code), f"src{int(code)}")


# ===========================================================================
# On-disk RETAIN-ALL archive
# ===========================================================================

@dataclass
class _ShardRef:
    path: Path
    n: int          # positions in this shard
    start: int      # global index of the first position in this shard


class PositionArchive:
    """Append-only, retain-everything store of packed training positions.

    Layout on disk (a directory):
        <root>/
            manifest.json          # ordered list of shards + total count
            shard_000000.pt        # torch.save dict (see _SHARD_KEYS below)
            shard_000001.pt
            ...

    Each shard is a dict of stacked tensors, ALL sharing the repo's existing
    packing:
        planes        : (n, N_INPUT_PLANES, BOARD, BOARD) float32  (== SelfPlayExample.planes)
        pi            : (n, N_ACTIONS)                      float32  (== SelfPlayExample.pi)
        z             : (n,)                                float32  (== SelfPlayExample.z)
        model_version : (n,)                                int32    (which "brain" generated it)
        plies         : (n,)                                int16    (game length / ply at capture)
        color         : (n,)                                int8     (0=black to move, 1=white; == SelfPlayExample.side)
        source        : (n,)                                int8     (SOURCE_* code)

    `plies` here is the *per-position* ply-at-capture (mirrors ReplayBuffer.ply
    and SelfPlayExample.ply) so the curator can ply-balance; the game-level
    GameRecord.plies is folded into it at append time via the `plies` arg.

    Appends create a NEW shard (never rewrite) so the file is genuinely
    append-only and crash-safe via atomic rename + manifest rewrite.
    """

    _SHARD_KEYS = ("planes", "pi", "z", "model_version", "plies", "color", "source")

    def __init__(self, root: str | Path, *, readonly: bool = False):
        self.root = Path(root)
        self.readonly = readonly
        self._shards: list[_ShardRef] = []
        self._total = 0
        # Lazy LRU-ish cache of materialized shard tensors for indexed reads.
        self._cache: dict[int, dict[str, torch.Tensor]] = {}
        self._cache_order: list[int] = []
        self._cache_max = 4
        if (self.root / "manifest.json").exists():
            self._load_manifest()
        elif not readonly:
            self.root.mkdir(parents=True, exist_ok=True)
            self._write_manifest()

    # -- manifest -----------------------------------------------------------
    def _load_manifest(self) -> None:
        man = json.loads((self.root / "manifest.json").read_text())
        self._shards = []
        start = 0
        for entry in man["shards"]:
            ref = _ShardRef(path=self.root / entry["path"], n=int(entry["n"]), start=start)
            self._shards.append(ref)
            start += ref.n
        self._total = start
        # Trust the manifest's total, but the running `start` is authoritative.

    def _write_manifest(self) -> None:
        man = {
            "format": "gomoku-position-archive-v1",
            "n_input_planes": N_INPUT_PLANES,
            "board_size": BOARD_SIZE,
            "n_actions": N_ACTIONS,
            "total": self._total,
            "shards": [{"path": ref.path.name, "n": ref.n} for ref in self._shards],
        }
        tmp = self.root / "manifest.json.tmp"
        tmp.write_text(json.dumps(man, indent=2))
        tmp.replace(self.root / "manifest.json")

    # -- append -------------------------------------------------------------
    def append(
        self,
        planes: np.ndarray | torch.Tensor,
        pi: np.ndarray | torch.Tensor,
        z: np.ndarray | torch.Tensor,
        *,
        model_version: int | np.ndarray,
        plies: int | np.ndarray,
        color: int | np.ndarray,
        source: int | np.ndarray = SOURCE_SELFPLAY,
    ) -> int:
        """Append a batch of `n` packed positions as a new shard.

        Scalars for model_version/plies/color/source are broadcast to `n`.
        Returns the global start index of the appended block.
        """
        if self.readonly:
            raise RuntimeError("archive opened readonly")
        planes_t = _as_f32(planes)
        n = planes_t.shape[0]
        if n == 0:
            return self._total
        pi_t = _as_f32(pi)
        z_t = _as_f32(z).reshape(n)
        shard = {
            "planes": planes_t,
            "pi": pi_t,
            "z": z_t,
            "model_version": _broadcast_int(model_version, n, torch.int32),
            "plies": _broadcast_int(plies, n, torch.int16),
            "color": _broadcast_int(color, n, torch.int8),
            "source": _broadcast_int(source, n, torch.int8),
        }
        idx = len(self._shards)
        name = f"shard_{idx:06d}.pt"
        tmp = self.root / (name + ".tmp")
        torch.save(shard, tmp)
        tmp.replace(self.root / name)
        ref = _ShardRef(path=self.root / name, n=n, start=self._total)
        self._shards.append(ref)
        self._total += n
        self._write_manifest()
        return ref.start

    def append_examples(
        self,
        examples: list,
        *,
        model_version: int,
        plies: int | None = None,
        source: int = SOURCE_SELFPLAY,
    ) -> int:
        """Convenience append from a list of `SelfPlayExample`.

        Reuses the SAME field layout the ReplayBuffer.add() path uses: e.planes,
        e.pi, e.z, e.side (-> color), e.ply (-> per-position plies). `plies` (the
        game length) overrides the per-position ply when given — pass it to keep
        the game-level plies metadata; otherwise we fall back to e.ply.
        """
        if not examples:
            return self._total
        planes = np.stack([e.planes for e in examples]).astype(np.float32)
        pi = np.stack([e.pi for e in examples]).astype(np.float32)
        z = np.array([e.z for e in examples], dtype=np.float32)
        color = np.array([int(getattr(e, "side", 0)) for e in examples], dtype=np.int8)
        if plies is None:
            plies_arr = np.array([int(getattr(e, "ply", 0)) for e in examples], dtype=np.int16)
        else:
            plies_arr = np.full(len(examples), int(plies), dtype=np.int16)
        return self.append(
            planes, pi, z,
            model_version=int(model_version),
            plies=plies_arr,
            color=color,
            source=source,
        )

    # -- read ---------------------------------------------------------------
    def __len__(self) -> int:
        return self._total

    @property
    def total(self) -> int:
        return self._total

    def _shard_for(self, global_idx: int) -> tuple[int, int]:
        """Return (shard_index, local_index) for a global position index."""
        lo, hi = 0, len(self._shards)
        while lo < hi:
            mid = (lo + hi) // 2
            ref = self._shards[mid]
            if global_idx < ref.start:
                hi = mid
            elif global_idx >= ref.start + ref.n:
                lo = mid + 1
            else:
                return mid, global_idx - ref.start
        raise IndexError(global_idx)

    def _get_shard(self, shard_idx: int) -> dict[str, torch.Tensor]:
        cached = self._cache.get(shard_idx)
        if cached is not None:
            return cached
        data = torch.load(self._shards[shard_idx].path, map_location="cpu", weights_only=False)
        self._cache[shard_idx] = data
        self._cache_order.append(shard_idx)
        while len(self._cache_order) > self._cache_max:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return data

    def read_metadata(self) -> dict[str, np.ndarray]:
        """Load JUST the per-position metadata for the whole archive (no planes).

        Returns numpy arrays of length `total`: model_version, plies, color,
        source, z. This is what curators run on — it's small (a few int arrays)
        so a curator can scan the entire retain-all archive cheaply without
        touching the heavy planes tensors.
        """
        if self._total == 0:
            empty_i = np.zeros(0, dtype=np.int64)
            return {
                "model_version": empty_i, "plies": empty_i,
                "color": empty_i, "source": empty_i,
                "z": np.zeros(0, dtype=np.float32),
            }
        mv, pl, co, so, zz = [], [], [], [], []
        for shard_idx in range(len(self._shards)):
            d = self._get_shard(shard_idx)
            mv.append(d["model_version"].numpy())
            pl.append(d["plies"].numpy())
            co.append(d["color"].numpy())
            so.append(d["source"].numpy())
            zz.append(d["z"].numpy())
        return {
            "model_version": np.concatenate(mv).astype(np.int64),
            "plies": np.concatenate(pl).astype(np.int64),
            "color": np.concatenate(co).astype(np.int64),
            "source": np.concatenate(so).astype(np.int64),
            "z": np.concatenate(zz).astype(np.float32),
        }

    def gather(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        """Materialize the given global indices into RAM tensors.

        Returns a dict with planes/pi/z/color(side)/plies(ply) so the result is
        directly loadable by CuratedBuffer (and shaped like a ReplayBuffer
        sample). Groups indices by shard to minimize shard loads.
        """
        indices = np.asarray(indices, dtype=np.int64)
        n = len(indices)
        planes = torch.empty((n, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
        pi = torch.empty((n, N_ACTIONS), dtype=torch.float32)
        z = torch.empty((n,), dtype=torch.float32)
        color = torch.empty((n,), dtype=torch.int8)
        plies = torch.empty((n,), dtype=torch.int16)
        # Map each requested index to (shard, local), then process shard-by-shard.
        shard_of = np.empty(n, dtype=np.int64)
        local_of = np.empty(n, dtype=np.int64)
        for i, g in enumerate(indices):
            s, l = self._shard_for(int(g))
            shard_of[i] = s
            local_of[i] = l
        for s in np.unique(shard_of):
            d = self._get_shard(int(s))
            mask = shard_of == s
            out_pos = np.nonzero(mask)[0]
            loc = torch.from_numpy(local_of[mask])
            planes[out_pos] = d["planes"][loc]
            pi[out_pos] = d["pi"][loc]
            z[out_pos] = d["z"][loc]
            color[out_pos] = d["color"][loc]
            plies[out_pos] = d["plies"][loc]
        return {"planes": planes, "pi": pi, "z": z, "color": color, "plies": plies}

    def iter_metadata_shards(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate shards yielding metadata-only dicts (for streaming scans)."""
        for shard_idx in range(len(self._shards)):
            d = self._get_shard(shard_idx)
            yield {k: d[k] for k in ("model_version", "plies", "color", "source", "z")}


# ===========================================================================
# Curator registry
# ===========================================================================
# A curator is: curate(meta, slice_size, cycle, rng) -> np.ndarray of indices
# into the archive. `meta` is the dict from PositionArchive.read_metadata().
# Kept tiny on purpose; richer recipes register via register_curator().

CuratorFn = Callable[[dict, int, int, np.random.Generator], np.ndarray]
_CURATORS: dict[str, CuratorFn] = {}


def register_curator(name: str, fn: CuratorFn) -> None:
    _CURATORS[name] = fn


def get_curator(name: str) -> CuratorFn:
    if name not in _CURATORS:
        raise KeyError(f"unknown curator {name!r}; options: {sorted(_CURATORS)}")
    return _CURATORS[name]


def list_curators() -> list[str]:
    return sorted(_CURATORS)


def _curate_lru(meta: dict, slice_size: int, cycle: int, rng: np.random.Generator) -> np.ndarray:
    """Most-recent-N by archive order (== a rolling FIFO of width slice_size).

    The boring control. Archive append order is generation order, so the last
    `slice_size` indices are the freshest positions.
    """
    total = len(meta["model_version"])
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    n = min(slice_size, total)
    return np.arange(total - n, total, dtype=np.int64)


def _curate_recency_weighted(
    meta: dict, slice_size: int, cycle: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample `slice_size` indices weighted toward recent model_versions.

    Weight = (1 + version)^2 normalized over the whole archive (favoring recent
    brains without ever fully excluding old positions — that's the difference
    from lru, which hard-clips to the tail). Sampled WITHOUT replacement when
    the archive is large enough, with replacement otherwise.
    """
    versions = meta["model_version"]
    total = len(versions)
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    n = min(slice_size, total)
    vmin = versions.min()
    # Shift so weights are positive even with version 0; square to bias harder
    # toward recent than a linear ramp (so it visibly differs from lru's tail).
    w = (versions - vmin + 1).astype(np.float64) ** 2
    w = w / w.sum()
    replace = total < slice_size
    return rng.choice(total, size=n, replace=replace, p=w).astype(np.int64)


register_curator("lru", _curate_lru)
register_curator("recency_weighted", _curate_recency_weighted)


# ===========================================================================
# Curated in-RAM slice
# ===========================================================================

class CuratedBuffer:
    """A curated in-RAM sample pool drawn from a (frozen or growing) archive.

    Holds `slice_size` positions in RAM as packed tensors. ``sample(batch)``
    matches ReplayBuffer.sample()'s 5-tuple signature
    (planes, pi, z, side, ply) so the trainer's SGD hot path is unchanged.

    Refill cadence is the caller's job: call ``refill(cycle)`` at EPOCH / CYCLE
    boundaries (never per-minibatch). Between refills the slice is fixed, so
    sampling is a pure RAM gather — no disk I/O on the hot path.
    """

    def __init__(
        self,
        archive: PositionArchive,
        *,
        slice_size: int,
        curator: str | CuratorFn = "lru",
        device: torch.device | str = "cpu",
        seed: int = 0,
    ):
        self.archive = archive
        self.slice_size = int(slice_size)
        self.curator: CuratorFn = curator if callable(curator) else get_curator(curator)
        self.curator_name = curator if isinstance(curator, str) else getattr(curator, "__name__", "custom")
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        # Materialized slice tensors.
        self.planes: torch.Tensor | None = None
        self.pi: torch.Tensor | None = None
        self.z: torch.Tensor | None = None
        self.side: torch.Tensor | None = None
        self.ply: torch.Tensor | None = None
        # Bookkeeping for the last refill (used for composition logging).
        self.last_indices: np.ndarray = np.zeros(0, dtype=np.int64)
        self.last_meta: dict[str, np.ndarray] = {}

    @property
    def size(self) -> int:
        return 0 if self.planes is None else self.planes.shape[0]

    def refill(self, cycle: int) -> int:
        """Re-run the curator against the current archive and reload the slice.

        Returns the number of positions now in the slice. Called at epoch/cycle
        boundaries. Reads only metadata to pick indices, then gathers the heavy
        planes for just those indices.

        NOTE (live-gen hook): in a future lazy/pull-based design, this is where
        the curator would signal "need N more from source X" and block on the
        worker pool before gathering. For replay-from-frozen-archive it just
        reads what's already on disk. The interface (refill at cycle boundary)
        is the same either way.
        """
        meta = self.archive.read_metadata()
        idx = self.curator(meta, self.slice_size, cycle, self.rng)
        idx = np.asarray(idx, dtype=np.int64)
        gathered = self.archive.gather(idx)
        self.planes = gathered["planes"].to(self.device)
        self.pi = gathered["pi"].to(self.device)
        self.z = gathered["z"].to(self.device)
        self.side = gathered["color"].to(self.device)
        self.ply = gathered["plies"].to(self.device)
        self.last_indices = idx
        # Stash the metadata of the SELECTED positions for composition logging.
        self.last_meta = {k: meta[k][idx] for k in ("model_version", "plies", "color", "source", "z")}
        return self.size

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Draw a minibatch — identical contract to ReplayBuffer.sample()."""
        if self.size == 0:
            raise ValueError("empty curated slice; call refill() first")
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.planes[idx], self.pi[idx], self.z[idx], self.side[idx], self.ply[idx]

    # -- composition logging ------------------------------------------------
    def composition(self) -> dict[str, float]:
        """Histogram the CURRENT slice over (model_version, plies, z, color,
        source) as a flat dict of wandb-friendly scalars. Computed from the
        stashed selection metadata — no extra disk reads.
        """
        meta = self.last_meta
        if not meta or len(meta.get("model_version", [])) == 0:
            return {"curator/slice_n": 0.0}
        mv = meta["model_version"]
        pl = meta["plies"]
        co = meta["color"]
        so = meta["source"]
        zz = meta["z"]
        n = len(mv)
        out: dict[str, float] = {
            "curator/slice_n": float(n),
            "curator/version_mean": float(mv.mean()),
            "curator/version_min": float(mv.min()),
            "curator/version_max": float(mv.max()),
            "curator/version_p50": float(np.percentile(mv, 50)),
            "curator/plies_mean": float(pl.mean()),
            "curator/color_black_frac": float((co == 0).mean()),
            "curator/color_white_frac": float((co == 1).mean()),
            "curator/z_wins": float((zz > 0).mean()),
            "curator/z_losses": float((zz < 0).mean()),
            "curator/z_draws": float((zz == 0).mean()),
        }
        # Per-source fractions.
        for code in np.unique(so):
            label = source_label(int(code))
            out[f"curator/source_{label}_frac"] = float((so == code).mean())
        # Coarse ply-bucket histogram (mirrors the trainer's ply buckets).
        for lo, hi, label in ((0, 10, "ply_00_10"), (10, 25, "ply_10_25"), (25, 1000, "ply_25_up")):
            mask = (pl >= lo) & (pl < hi)
            out[f"curator/{label}_frac"] = float(mask.mean())
        # Version-spread bucket histogram relative to the freshest version in
        # the slice (so the difference between lru's tight tail and
        # recency_weighted's long tail is visible).
        latest = int(mv.max())
        age = latest - mv
        for lo, hi, label in ((0, 1, "age_0"), (1, 5, "age_1_5"), (5, 20, "age_5_20"), (20, 10**9, "age_20_up")):
            mask = (age >= lo) & (age < hi)
            out[f"curator/{label}_frac"] = float(mask.mean())
        return out

    def log_composition(self, cycle: int, *, wandb_run=None, tsv_path: str | Path | None = None) -> dict[str, float]:
        """Emit the slice composition to W&B (if a run is given) and/or a TSV.

        Returns the composition dict so the caller can fold it into its own log.
        The TSV is append-only: header on first write, one row per refill.
        """
        comp = self.composition()
        comp_logged = {"cycle": cycle, "curator": self.curator_name, **comp}
        if wandb_run is not None:
            wandb_run.log({f"composition/{k}" if not k.startswith("curator/") else k: v
                           for k, v in comp.items()} | {"cycle": cycle})
        if tsv_path is not None:
            tsv_path = Path(tsv_path)
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            keys = list(comp_logged.keys())
            write_header = not tsv_path.exists()
            with tsv_path.open("a") as f:
                if write_header:
                    f.write("\t".join(keys) + "\n")
                f.write("\t".join(str(comp_logged[k]) for k in keys) + "\n")
        return comp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_f32(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().contiguous()
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _broadcast_int(v: int | np.ndarray | torch.Tensor, n: int, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        t = v.to(dtype).cpu().reshape(-1)
    elif isinstance(v, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(v)).to(dtype).reshape(-1)
    else:
        return torch.full((n,), int(v), dtype=dtype)
    if t.numel() == 1:
        return t.expand(n).clone()
    if t.numel() != n:
        raise ValueError(f"metadata length {t.numel()} != n {n}")
    return t
