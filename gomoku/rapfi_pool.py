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
import queue
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
    """Raised when the Rapfi binary / config cannot be found on disk."""


def default_rapfi_repo(repo: str | None = None) -> str:
    """Resolve the gomoku checkout that holds ``engines/rapfi/``.

    The Rapfi binary + NNUE weights are local build artifacts (not version
    controlled), so they live in the user's MAIN checkout even when this code
    runs from a worktree. Precedence: explicit arg → ``GOMOKU_REPO`` env →
    ``~/code/gomoku`` (matches ``tests/test_rapfi_native.py``).
    """
    return repo or os.environ.get("GOMOKU_REPO") or os.path.expanduser("~/code/gomoku")


def default_rapfi_cmd(repo: str | None = None) -> str:
    """The canonical ``pbrain-rapfi --config ... gomocup`` launch command."""
    rdir = os.path.join(default_rapfi_repo(repo), "engines", "rapfi")
    binp = os.path.join(rdir, "pbrain-rapfi")
    cfg = os.path.join(rdir, "config.toml")
    return f"{binp} --config {cfg} gomocup"


def rapfi_available(repo: str | None = None) -> bool:
    """True iff the Rapfi binary and config exist (cheap, no spawn)."""
    rdir = os.path.join(default_rapfi_repo(repo), "engines", "rapfi")
    return os.path.isfile(os.path.join(rdir, "pbrain-rapfi")) and os.path.isfile(
        os.path.join(rdir, "config.toml")
    )


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
        resolved_cmd = cmd or default_rapfi_cmd()
        # Fail fast with a clear message rather than a cryptic FileNotFoundError
        # from deep inside subprocess.Popen.
        bin_path = resolved_cmd.split()[0]
        if not os.path.isfile(bin_path):
            raise RapfiUnavailable(
                f"Rapfi binary not found: {bin_path!r}. Set GOMOKU_REPO or pass "
                f"cmd=... (the binary is a local build artifact, not in git)."
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
