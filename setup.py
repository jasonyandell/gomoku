from __future__ import annotations

from setuptools import Extension, setup

import numpy as np


setup(
    ext_modules=[
        # Default 9x9 builds (BOARD_SIZE defaults to 9 inside the C sources).
        Extension(
            "gomoku._state_ops_native",
            sources=["gomoku/_state_ops_native.c"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "gomoku._mcts_native",
            sources=["gomoku/_mcts_native.c"],
            include_dirs=[np.get_include()],
        ),
        # 15x15 builds: thin shims that #define BOARD_SIZE 15 and #include the
        # shared implementation. gomoku.state_ops / gomoku.native_mcts pick the
        # module matching the active board size at import time.
        Extension(
            "gomoku._state_ops_native15",
            sources=["gomoku/_state_ops_native15.c"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "gomoku._mcts_native15",
            sources=["gomoku/_mcts_native15.c"],
            include_dirs=[np.get_include()],
        ),
    ],
)
