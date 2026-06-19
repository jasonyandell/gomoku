"""Unit tests for the --jobs shard distribution (issue #52).

Pure logic only — no GPU/engine. End-to-end parallel execution is covered by a
manual smoke (champion vs native Rapfi); here we pin the invariant that matters:
sharding never loses or duplicates a pair, and shards stay balanced.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_vs_rapfi",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "eval_vs_rapfi.py"),
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)
_shard_pairs = _mod._shard_pairs


@pytest.mark.parametrize("total_pairs,jobs", [
    (1, 1), (1, 8), (12, 4), (12, 5), (20, 8), (3, 8), (16, 16), (15, 4), (7, 3),
])
def test_shard_pairs_conserves_total(total_pairs, jobs):
    shards = _shard_pairs(total_pairs, jobs)
    assert sum(shards) == total_pairs           # never lose/duplicate a pair
    assert all(c >= 1 for c in shards)           # no empty shards
    assert len(shards) == min(jobs, total_pairs)  # use up to `jobs` shards
    assert max(shards) - min(shards) <= 1         # balanced to within one pair


def test_shard_pairs_single_job_is_one_shard():
    assert _shard_pairs(10, 1) == [10]


def test_shard_pairs_more_jobs_than_pairs():
    # 3 pairs, 8 jobs -> 3 shards of 1 (never spawn idle workers with no games)
    assert _shard_pairs(3, 8) == [1, 1, 1]
