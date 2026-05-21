from __future__ import annotations

from setuptools import Extension, setup

import numpy as np


setup(
    ext_modules=[
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
    ],
)
