"""Optional native MCTS engine boundary.

The native engine owns the arena-backed MCTS tree, PUCT selection, child
creation, virtual loss, backup, and input-plane materialization. Python still
owns the PyTorch model callback at wave boundaries and the outer self-play loop.
"""

from __future__ import annotations

import os
from typing import Any

try:  # pragma: no cover - import path depends on local build artifacts.
    if os.environ.get("GOMOKU_DISABLE_NATIVE_MCTS"):
        raise ImportError("native MCTS disabled by environment")
    from gomoku import _mcts_native as _native
except ImportError:  # pragma: no cover - fallback path on source-only installs.
    _native = None

USING_NATIVE_MCTS = _native is not None
NativeMCTSGame = None if _native is None else _native.NativeMCTSGame


def search_batch(*args: Any, **kwargs: Any) -> None:
    if _native is None:
        raise RuntimeError("native MCTS extension is not available")
    _native.search_batch(*args, **kwargs)


def backend() -> Any | None:
    return _native
