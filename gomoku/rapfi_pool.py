"""A warm pool of persistent Rapfi engine processes.

The babysit eval scripts spawn a fresh ``pbrain-rapfi`` subprocess for every
eval pass — a 15×15 NNUE engine pays a real start-up + weight-load cost each
time. When you eval on a *cadence* (every checkpoint) or *teach* (label
thousands of positions), that respawn tax dominates. This pool spawns ``size``
engines once and lends them out, so the engines stay warm across calls — the
10×+ speedup that makes always-on eval and Rapfi-as-teacher practical.

Why a pool is safe here: in BOARD mode (``incremental=False``, the classical
default for Rapfi) every :class:`ExternalEnginePlayer` call does a full
``RESTART`` + ``BOARD`` re-dump of the position, so an engine carries NO state
between calls or games. One instance can label arbitrary unrelated positions
back-to-back. The only rule is *one borrower per instance at a time* — each
engine owns a single stdin/stdout pipe and two threads must not interleave on
it. The lease queue enforces that.

Concurrency model: ``size`` engines == ``size`` OS processes computing in true
parallel (Rapfi is CPU NNUE; the GIL is released during the blocking pipe
read). Run this on CPU alongside an MPS trainer and it does not compete for the
GPU — the same reason the babysit cadence is safe to run during training.

Self-healing: a crashed / timed-out engine raises
:class:`~gomoku.external_engine.ExternalEngineError`; :meth:`pick` transparently
discards the corpse, spawns a replacement (keeping the pool at ``size``), and
retries once. A caller never sees a transient engine death.
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np

from gomoku.board_config import BOARD_SIZE
from gomoku.external_engine import (
    ExternalEngineConfig,
    ExternalEngineError,
    ExternalEnginePlayer,
)


class RapfiUnavailable(RuntimeError):
    """Raised when the Rapfi binary / config cannot be resolved."""


# --------------------------------------------------------------------------
# Artifact resolution — local build OR a pinned HuggingFace snapshot.
#
# The Rapfi binary + NNUE weights are ~40MB of gitignored local build artifacts,
# so a fresh worktree / machine / CI doesn't have them. Rather than the brittle
# GOMOKU_REPO -> ~/code/gomoku indirection back to the main checkout, we resolve
# them from a PINNED-revision HF repo into the machine-global hub cache (one
# fetch per machine, shared across every worktree/venv — the one store that is
# genuinely worktree- and venv-invariant). The local build stays the higher-
# precedence fast path, so a box that already built the engine never touches the
# network and behaviour is byte-identical to before.
#
# Pinned to an immutable commit SHA (NOT a mutable tag) for reproducibility, and
# the binary's sha256 is asserted after fetch (we chmod +x a file pulled from the
# network — verify it's exactly the build we pinned). Both constants are filled by
# scripts/publish_rapfi.py; while None, only the local path is used (legacy
# behaviour, byte-identical).
# --------------------------------------------------------------------------
RAPFI_HF_REPO = "jasonyandell/rapfi-arm64"
# Pinned to an immutable commit SHA (built from dhbloo/rapfi @ 6e0a132, arm64).
# Bump both via scripts/publish_rapfi.py after a new build.
RAPFI_HF_REVISION: str | None = "697aec1a50ab8b8b5b280749516fb37ea95435c5"
RAPFI_BINARY_SHA256: str | None = (
    "9f8efade631f8391b2763acb5f66038d2b0ee5ace2f12919e80f8b497fe3ded6"
)
_RAPFI_PATTERNS = ["pbrain-rapfi", "config.toml", "*.bin.lz4", "model210901.bin"]


def default_rapfi_repo(repo: str | None = None) -> str:
    """Resolve the gomoku checkout that *may* hold a local ``engines/rapfi/``
    build. Precedence: explicit arg → ``GOMOKU_REPO`` env → this checkout's root
    (the package's own repo) → ``~/code/gomoku``.
    """
    if repo:
        return repo
    env = os.environ.get("GOMOKU_REPO")
    if env:
        return env
    # The repo root two levels up from this file (…/<root>/gomoku/rapfi_pool.py).
    here_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(here_root, "engines", "rapfi", "pbrain-rapfi")):
        return here_root
    return os.path.expanduser("~/code/gomoku")


def _local_rapfi_dir(repo: str | None = None) -> str | None:
    """The local ``engines/rapfi`` dir iff it holds a runnable build, else None."""
    rdir = os.path.join(default_rapfi_repo(repo), "engines", "rapfi")
    if os.path.isfile(os.path.join(rdir, "pbrain-rapfi")) and os.path.isfile(
        os.path.join(rdir, "config.toml")
    ):
        return rdir
    return None


def _hf_importable() -> bool:
    try:
        import huggingface_hub  # noqa: F401

        return True
    except Exception:
        return False


def _is_arm64_mac() -> bool:
    """The published binary is an arm64 macOS Mach-O — only fetchable-and-runnable
    here. (Making the FILES appear on a Linux runner must not imply it can run.)"""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rapfi_artifacts(
    *, revision: str | None = None, allow_fetch: bool = True, repo: str | None = None
) -> str:
    """Return a directory containing ``pbrain-rapfi`` + ``config.toml`` + weights.

    Precedence: local ``engines/rapfi`` build (fast path) → the pinned HF snapshot
    in the machine-global hub cache (fetched on first use iff ``allow_fetch``).
    The binary is made executable and (when a hash is pinned) sha256-verified.
    Raises :class:`RapfiUnavailable` if neither resolves.
    """
    local = _local_rapfi_dir(repo)
    if local is not None:
        return local

    rev = revision or RAPFI_HF_REVISION
    if not rev:
        raise RapfiUnavailable(
            "no local engines/rapfi build and no pinned HF revision "
            "(RAPFI_HF_REVISION is None). Build via engines/rapfi/build_rapfi.sh, "
            "or publish once via `python scripts/publish_rapfi.py`."
        )
    if not _hf_importable():
        raise RapfiUnavailable(
            "huggingface_hub not installed; cannot fetch Rapfi "
            "(`uv pip install huggingface_hub`)."
        )
    from huggingface_hub import snapshot_download

    try:
        adir = snapshot_download(
            RAPFI_HF_REPO,
            revision=rev,
            allow_patterns=_RAPFI_PATTERNS,
            local_files_only=not allow_fetch,
        )
    except Exception as e:
        raise RapfiUnavailable(
            f"could not resolve Rapfi from HF {RAPFI_HF_REPO}@{rev[:12]} "
            f"(allow_fetch={allow_fetch}): {type(e).__name__}: {e}"
        ) from e

    binp = os.path.join(adir, "pbrain-rapfi")
    if not os.path.isfile(binp):
        raise RapfiUnavailable(f"HF snapshot {adir} is missing pbrain-rapfi")
    if not os.access(binp, os.X_OK):
        # Hub cache stores blobs 0644; the engine must be executable.
        os.chmod(binp, 0o755)
    if RAPFI_BINARY_SHA256:
        got = _sha256(binp)
        if got != RAPFI_BINARY_SHA256:
            raise RapfiUnavailable(
                f"Rapfi binary sha256 mismatch (refusing to run an unexpected "
                f"binary): expected {RAPFI_BINARY_SHA256[:12]}…, got {got[:12]}…"
            )
    return adir


def default_rapfi_cmd(repo: str | None = None) -> str:
    """The canonical ``pbrain-rapfi --config ... gomocup`` launch command,
    resolving the binary+config from the local build or the pinned HF snapshot
    (fetching on first use if needed)."""
    adir = rapfi_artifacts(repo=repo)
    binp = os.path.join(adir, "pbrain-rapfi")
    cfg = os.path.join(adir, "config.toml")
    return f"{binp} --config {cfg} gomocup"


def rapfi_available(repo: str | None = None) -> bool:
    """True iff Rapfi is resolvable WITHOUT a network fetch — a local build is
    present, or the pinned HF snapshot is already cached. Cheap and never
    downloads, so it is safe for test gating."""
    if _local_rapfi_dir(repo) is not None:
        return True
    if not RAPFI_HF_REVISION or not _hf_importable():
        return False
    try:
        rapfi_artifacts(allow_fetch=False, repo=repo)  # cache-only
        return True
    except RapfiUnavailable:
        return False


def rapfi_obtainable(repo: str | None = None) -> bool:
    """True iff Rapfi can be made available, possibly via a one-time HF fetch on
    first use (local present, already cached, or fetchable on this arch). Use this
    (not :func:`rapfi_available`) to decide whether to *offer* Rapfi."""
    if rapfi_available(repo):
        return True
    return bool(RAPFI_HF_REVISION) and _hf_importable() and _is_arm64_mac()


class RapfiPool:
    """A fixed-size pool of warm :class:`ExternalEnginePlayer` instances.

    Use as a context manager so the engines are always torn down::

        with RapfiPool(size=6, timeout_ms=1000, board_size=15) as pool:
            move = pool.pick(state)          # one labelled move
            moves = pool.label_states(states)  # many, in parallel
    """

    def __init__(
        self,
        *,
        size: int = 4,
        cmd: str | None = None,
        timeout_ms: int = 1000,
        board_size: int = BOARD_SIZE,
        rule: int = 0,
        label: str = "rapfi",
        get_timeout_s: float = 120.0,
    ) -> None:
        if size < 1:
            raise ValueError(f"pool size must be >= 1, got {size}")
        # default_rapfi_cmd() resolves the local build or the pinned HF snapshot
        # (fetching on first use); an explicit cmd is used as-is.
        resolved_cmd = cmd or default_rapfi_cmd()
        # Fail fast with a clear message rather than a cryptic FileNotFoundError
        # from deep inside subprocess.Popen (covers an explicit cmd= that points
        # at a missing binary).
        bin_path = resolved_cmd.split()[0]
        if not os.path.isfile(bin_path):
            raise RapfiUnavailable(
                f"Rapfi binary not found: {bin_path!r}. Pass a valid cmd=, build "
                f"engines/rapfi/build_rapfi.sh, or publish via "
                f"scripts/publish_rapfi.py (then the HF resolver fetches it)."
            )
        self._cfg_kwargs = dict(
            cmd=resolved_cmd,
            timeout_ms=timeout_ms,
            label=label,
            rule=rule,
            board_size=board_size,
            incremental=False,  # stateless BOARD mode — required for safe reuse
        )
        self.size = size
        self._get_timeout_s = get_timeout_s
        self._pool: queue.Queue[ExternalEnginePlayer] = queue.Queue()
        self._all: list[ExternalEnginePlayer] = []
        self._lock = threading.Lock()
        self._closed = False
        for _ in range(size):
            self._pool.put(self._spawn())

    # -- lifecycle -------------------------------------------------------

    def _spawn(self) -> ExternalEnginePlayer:
        eng = ExternalEnginePlayer(ExternalEngineConfig(**self._cfg_kwargs))
        with self._lock:
            closing = self._closed
            if not closing:
                self._all.append(eng)
        if closing:
            # Lost the race with close(): don't leak the just-spawned subprocess.
            try:
                eng.close()
            except Exception:
                pass
            raise RuntimeError("RapfiPool is closed")
        return eng

    def _return(self, eng: ExternalEnginePlayer) -> None:
        """Return a healthy engine to the pool, or close it if we're shutting
        down — never leave a closed/orphan engine parked in the queue."""
        with self._lock:
            closing = self._closed
            if not closing:
                self._pool.put(eng)
        if closing:
            try:
                eng.close()
            except Exception:
                pass

    def _retire(self, eng: ExternalEnginePlayer) -> None:
        """Close a (likely dead) engine and drop it from the roster."""
        with self._lock:
            try:
                self._all.remove(eng)
            except ValueError:
                pass
        try:
            eng.close()
        except Exception:
            pass

    @contextmanager
    def lease(self, timeout_s: float | None = None):
        """Borrow one engine for the duration of the ``with`` block.

        On a clean exit the engine returns to the pool. If the block raises
        :class:`ExternalEngineError` the engine is presumed dead: it is retired
        and a fresh one is spawned to keep the pool at ``size``.
        """
        if self._closed:
            raise RuntimeError("RapfiPool is closed")
        eng = self._pool.get(
            timeout=self._get_timeout_s if timeout_s is None else timeout_s
        )
        healthy = True
        try:
            yield eng
        except ExternalEngineError:
            healthy = False
            raise
        finally:
            if healthy:
                self._return(eng)
            else:
                self._retire(eng)
                # Replace the corpse so the pool keeps its capacity (a closing
                # pool makes _spawn raise, which we swallow — no re-queue).
                try:
                    self._return(self._spawn())
                except Exception:
                    # If respawn fails the pool shrinks by one but stays usable.
                    pass

    # -- the work --------------------------------------------------------

    def pick(self, state, rng: np.random.Generator | None = None, *, retries: int = 1) -> int:
        """Return Rapfi's chosen action for ``state`` (self-heals one death)."""
        if rng is None:
            rng = np.random.default_rng()
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with self.lease() as eng:
                    return eng(state, rng)
            except ExternalEngineError as e:
                last = e
                if attempt >= retries:
                    break
        assert last is not None
        raise last

    def analyze_state(
        self, state, *, max_node: int = 20000, retries: int = 1,
        max_pv: int | None = None,
    ) -> dict:
        """Return Rapfi's whole-board ``{action: winrate}`` map for ``state``.

        The SOFT-teacher signal: Rapfi scores every candidate root move with a
        side-to-move winrate in [0, 1] (pruned-away cells correctly absent ->
        ~0 mass). Self-heals one engine death, exactly like :meth:`pick`. The
        stateless BOARD drive (``incremental=False``, forced by the pool) makes
        an analysis carry no state between calls — safe to reuse the instance.
        """
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with self.lease() as eng:
                    return eng.analyze(state, max_node=max_node, max_pv=max_pv)
            except ExternalEngineError as e:
                last = e
                if attempt >= retries:
                    break
        assert last is not None
        raise last

    def analyze_states(
        self, states, *, max_node: int = 20000, max_workers: int | None = None,
        max_pv: int | None = None,
    ) -> list[dict]:
        """Analyze many positions with the whole pool in parallel.

        Returns one ``{action: winrate}`` map per input state, in input order.
        Concurrency is capped at the pool size (more threads just block on the
        lease queue). ``max_pv`` caps the scored support per position (see
        :meth:`ExternalEnginePlayer.analyze`).
        """
        states = list(states)
        if not states:
            return []
        workers = min(self.size, len(states)) if max_workers is None else max_workers
        if workers <= 1:
            return [self.analyze_state(s, max_node=max_node, max_pv=max_pv) for s in states]
        out: list[dict | None] = [None] * len(states)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(self.analyze_state, s, max_node=max_node, max_pv=max_pv): i
                for i, s in enumerate(states)
            }
            for fut, i in futs.items():
                out[i] = fut.result()
        return [m if m is not None else {} for m in out]

    def label_states(
        self, states, *, max_workers: int | None = None
    ) -> list[int]:
        """Label many positions with Rapfi, using the whole pool in parallel.

        Returns one action per input state, in input order. Concurrency is
        capped at the pool size (more threads would just block on the queue).
        """
        states = list(states)
        if not states:
            return []
        workers = min(self.size, len(states)) if max_workers is None else max_workers
        if workers <= 1:
            return [self.pick(s) for s in states]
        out: list[int | None] = [None] * len(states)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.pick, s): i for i, s in enumerate(states)}
            for fut, i in futs.items():
                out[i] = fut.result()
        return [int(a) for a in out]  # type: ignore[arg-type]

    def close(self) -> None:
        # Hold the lock across set-closed + snapshot + drain so no concurrent
        # _spawn/_return can slip an engine into _all or the queue after we've
        # taken the snapshot (the leak/closed-engine-reuse races). _all tracks
        # every live engine — parked or on loan — so closing it covers both.
        with self._lock:
            if self._closed:
                return
            self._closed = True
            engines = list(self._all)
            self._all.clear()
            while True:
                try:
                    self._pool.get_nowait()
                except queue.Empty:
                    break
        for eng in engines:
            try:
                eng.close()
            except Exception:
                pass

    def __enter__(self) -> "RapfiPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
