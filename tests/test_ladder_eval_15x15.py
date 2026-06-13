"""Unit tests for the 15x15 external-ladder eval driver and the shared
Rapfi-eval JSONL schema/append helpers.

These are pure / no-GPU / no-engine: they exercise the checkpoint resolver
(`ladder_eval_15x15.resolve_checkpoint`) and the record schema + append-only
JSONL contract (`eval_vs_rapfi.make_eval_record` / `append_jsonl`). The live
Rapfi game is validated separately (it touches the model + the binary).

The driver pins GOMOKU_BOARD_SIZE=15 at import time; importing it here is
harmless because the pin only touches os.environ and does not import gomoku.*
until run_rapfi_eval is actually called (which these tests never do).
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")


def _load_script(name: str):
    """Import a scripts/*.py module by path (scripts/ is not a package)."""
    path = os.path.join(_SCRIPTS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ladder(monkeypatch):
    """Load the 15x15 driver with GOMOKU_BOARD_SIZE pinned to 15.

    The pytest session imports gomoku.* (board_config writes
    GOMOKU_BOARD_SIZE=9 back to os.environ at import time), so the driver's
    import-time pin would hard-error. We set it to 15 here; the driver does NOT
    import gomoku.* at module load (the heavy path is inside run_rapfi_eval,
    which these tests never call), so this is safe and does not change the
    already-resolved process board size.
    """
    monkeypatch.setenv("GOMOKU_BOARD_SIZE", "15")
    return _load_script("ladder_eval_15x15")


# --- checkpoint resolver -------------------------------------------------


def test_resolve_existing_path_returned_as_abspath(ladder, tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"stub")
    assert ladder.resolve_checkpoint(str(ckpt)) == os.path.abspath(str(ckpt))


def test_resolve_missing_latest_falls_back_to_newest_epoch(ladder, tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    # Out-of-order creation; resolver must pick by epoch number, not mtime.
    (ckpt_dir / "epoch0050.pt").write_bytes(b"x")
    (ckpt_dir / "epoch0055.pt").write_bytes(b"x")
    (ckpt_dir / "epoch0045.pt").write_bytes(b"x")
    (ckpt_dir / "worker_weights.pt").write_bytes(b"x")
    resolved = ladder.resolve_checkpoint(str(ckpt_dir / "latest.pt"))
    assert os.path.basename(resolved) == "epoch0055.pt"


def test_resolve_falls_back_to_worker_weights(ladder, tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "worker_weights.pt").write_bytes(b"x")
    resolved = ladder.resolve_checkpoint(str(ckpt_dir / "latest.pt"))
    assert os.path.basename(resolved) == "worker_weights.pt"


def test_resolve_no_checkpoint_anywhere_raises(ladder, tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    # Point WARMSTART_SEED at a missing path so the final fallback can't fire.
    ladder.WARMSTART_SEED = str(tmp_path / "does_not_exist.pt")
    with pytest.raises(SystemExit):
        ladder.resolve_checkpoint(str(ckpt_dir / "latest.pt"))


def test_driver_requires_board_size_15(ladder):
    assert ladder._REQUIRED_BOARD_SIZE == 15
    assert os.environ.get("GOMOKU_BOARD_SIZE") == "15"


# --- record schema + append contract ------------------------------------

_REQUIRED_KEYS = {
    "timestamp", "checkpoint", "model_sims", "model_c_puct",
    "fpu_reduction_c", "reuse_tree", "proven_prop", "engine",
    "engine_source", "engine_build_ref", "timeout_ms", "board_size",
    "rule", "wrapper_version", "n_games", "wins", "losses", "draws",
    "win_rate", "wall_secs",
}


def _make_sample_record(rapfi, **overrides):
    kw = dict(
        checkpoint="some/model.pt",
        sims=100,
        c_puct=1.5,
        fpu_reduction_c=0.0,
        reuse_tree=False,
        proven_prop=False,
        timeout_ms=200,
        board_size=15,
        rule=0,
        wrapper_version="1",
        build_ref={"path": "/x/pbrain-rapfi"},
        n_games=10,
        wins=4,
        losses=5,
        draws=1,
        win_rate=0.45,
        wall_secs=12.345,
    )
    kw.update(overrides)
    return rapfi.make_eval_record(**kw)


def test_record_has_full_schema_and_strength_fields():
    rapfi = _load_script("eval_vs_rapfi")
    rec = _make_sample_record(rapfi)
    assert _REQUIRED_KEYS <= set(rec)
    assert rec["board_size"] == 15
    assert rec["engine"] == "rapfi200"
    # win_rate carries the strength signal; checkpoint is abspath provenance.
    assert rec["win_rate"] == 0.45
    assert os.path.isabs(rec["checkpoint"])
    # wall_secs is rounded to 2 decimal places.
    assert rec["wall_secs"] == round(12.345, 2)
    # JSON-serializable end to end.
    assert json.loads(json.dumps(rec))["wins"] == 4


def test_record_extra_fields_merged():
    rapfi = _load_script("eval_vs_rapfi")
    rec = _make_sample_record(rapfi, extra_fields={"ladder": "rapfi_15x15"})
    assert rec["ladder"] == "rapfi_15x15"


def test_append_jsonl_is_append_only_and_safe_to_repeat(tmp_path):
    rapfi = _load_script("eval_vs_rapfi")
    out = tmp_path / "nested" / "ladder.jsonl"  # parent dir created on demand
    r1 = _make_sample_record(rapfi, timeout_ms=200)
    r2 = _make_sample_record(rapfi, timeout_ms=1000)
    rapfi.append_jsonl(str(out), r1)
    rapfi.append_jsonl(str(out), r2)  # second call must NOT clobber the first
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["timeout_ms"] for p in parsed] == [200, 1000]
    assert all(_REQUIRED_KEYS <= set(p) for p in parsed)
