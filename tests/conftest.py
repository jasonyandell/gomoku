"""Pytest session config for the gomoku test suite.

The CPU vcf solver (``gomoku.vcf`` :: ``solve_vcf`` / ``solve_vct`` and the
``*_from_planes`` wrappers) is RETIRED as a runtime dependency: every public
entry point raises :class:`gomoku.vcf.CpuSolverRetired` unless the deliberate-use
override ``GOMOKU_ALLOW_CPU_SOLVER=1`` is set (see
``wiki/topics/mega-vct-solver.md``).

The CPU solver is, however, the project's kept oracle / reference spec, and the
``tests/`` suite is exactly the sanctioned place that exercises it as such
(``test_vcf`` / ``test_vct`` validate the solver itself; the teacher / overlay /
proven-prop tests validate the runtime integrations against it). So this conftest
sets the override session-wide. The gate itself is covered by
``tests/test_cpu_solver_gate.py`` (which clears the env and asserts the raise).
"""

from __future__ import annotations

import os

# Sanctioned override: the test suite IS the kept-oracle / deep-validation path.
os.environ.setdefault("GOMOKU_ALLOW_CPU_SOLVER", "1")
