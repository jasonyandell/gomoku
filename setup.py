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
        # Larger-board builds: thin shims that #define BOARD_SIZE N and #include
        # the shared implementation. gomoku.state_ops / gomoku.native_mcts pick
        # the module matching the active board size at import time. The 11x11 and
        # 13x13 rungs exist for the progressive board-size curriculum (the
        # 9->11->13->15 ladder); native at every rung is required -- pure-Python
        # is ~1.6x slower regardless of size and would violate the epoch budget.
        Extension(
            "gomoku._state_ops_native11",
            sources=["gomoku/_state_ops_native11.c"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "gomoku._mcts_native11",
            sources=["gomoku/_mcts_native11.c"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "gomoku._state_ops_native13",
            sources=["gomoku/_state_ops_native13.c"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "gomoku._mcts_native13",
            sources=["gomoku/_mcts_native13.c"],
            include_dirs=[np.get_include()],
        ),
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
