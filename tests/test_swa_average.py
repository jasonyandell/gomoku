"""Tests for scripts/swa_average.py — uniform-mean averaging of peak.pt
checkpoints, CPU-only, no GPU."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

# Load scripts/swa_average.py as a module without requiring `scripts/` to be a
# package (it isn't — many sibling scripts are loaded this way in the repo).
_SWA_PATH = Path(__file__).resolve().parents[1] / "scripts" / "swa_average.py"
_spec = importlib.util.spec_from_file_location("swa_average", _SWA_PATH)
assert _spec is not None and _spec.loader is not None
swa_average = importlib.util.module_from_spec(_spec)
sys.modules["swa_average"] = swa_average
_spec.loader.exec_module(swa_average)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_state_dict(scalar: float) -> dict:
    """Tiny state_dict where every tensor is filled with `scalar`."""
    return {
        "conv.weight": torch.full((4, 3, 3, 3), scalar, dtype=torch.float32),
        "conv.bias": torch.full((4,), scalar, dtype=torch.float32),
        "bn.running_mean": torch.full((4,), scalar, dtype=torch.float32),
        "bn.running_var": torch.full((4,), scalar, dtype=torch.float32),
        "bn.num_batches_tracked": torch.tensor(int(scalar * 10), dtype=torch.long),
    }


def _make_payload(scalar: float, epoch: int) -> dict:
    return {
        "model_state_dict": _make_state_dict(scalar),
        "model_config": {"size": "tiny", "marker": "fixture"},
        "epoch": epoch,
        "total_games": epoch * 1000,
        "wandb_run_id": "fixture-run",
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 0.01}]},
    }


def _write_peak(dirpath: Path, name: str, scalar: float, epoch: int) -> Path:
    path = dirpath / name
    torch.save(_make_payload(scalar, epoch), path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_uniform_mean_three_peaks(tmp_path: Path) -> None:
    """Element-wise mean of {1.0, 3.0, 5.0} fixtures is exactly 3.0 everywhere."""
    lane = tmp_path / "lane"
    lane.mkdir()
    _write_peak(lane, "peak_e0010.pt", 1.0, epoch=10)
    _write_peak(lane, "peak_e0020.pt", 3.0, epoch=20)
    _write_peak(lane, "peak_e0030.pt", 5.0, epoch=30)

    out = lane / "peak_swa.pt"
    rc = swa_average.main(
        ["--lane-dir", str(lane), "--k", "3", "--output", str(out)]
    )
    assert rc == 0
    assert out.exists()

    payload = torch.load(out, map_location="cpu", weights_only=False)
    sd = payload["model_state_dict"]
    expected_mean = (1.0 + 3.0 + 5.0) / 3.0  # = 3.0
    assert torch.equal(
        sd["conv.weight"],
        torch.full((4, 3, 3, 3), expected_mean, dtype=torch.float32),
    )
    assert torch.equal(
        sd["conv.bias"], torch.full((4,), expected_mean, dtype=torch.float32)
    )
    # BN running stats — also a uniform mean.
    assert torch.equal(
        sd["bn.running_mean"], torch.full((4,), expected_mean, dtype=torch.float32)
    )
    # Integer counter — should be taken from the LAST peak (scalar=5.0, => 50),
    # not averaged.
    assert sd["bn.num_batches_tracked"].item() == 50

    # SWA metadata present.
    assert "swa" in payload
    assert payload["swa"]["k"] == 3
    assert payload["swa"]["method"] == "uniform_mean"
    assert payload["swa"]["source_peaks"] == [
        "peak_e0010.pt", "peak_e0020.pt", "peak_e0030.pt",
    ]
    assert payload["swa"]["source_epochs"] == [10, 20, 30]

    # Non-weight metadata preserved from the LAST peak.
    assert payload["epoch"] == 30
    assert payload["total_games"] == 30_000
    assert payload["wandb_run_id"] == "fixture-run"
    assert payload["model_config"] == {"size": "tiny", "marker": "fixture"}
    # Optimizer state dropped (averaging it is meaningless).
    assert "optimizer_state_dict" not in payload


def test_k_larger_than_available(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """--k 10 with only 3 peaks: averages 3, warns, no crash."""
    lane = tmp_path / "lane"
    lane.mkdir()
    _write_peak(lane, "peak_e0001.pt", 2.0, epoch=1)
    _write_peak(lane, "peak_e0002.pt", 4.0, epoch=2)
    _write_peak(lane, "peak_e0003.pt", 6.0, epoch=3)

    out = lane / "peak_swa.pt"
    rc = swa_average.main(
        ["--lane-dir", str(lane), "--k", "10", "--output", str(out)]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "only 3 peak" in captured.err

    payload = torch.load(out, map_location="cpu", weights_only=False)
    sd = payload["model_state_dict"]
    assert torch.equal(
        sd["conv.weight"], torch.full((4, 3, 3, 3), 4.0, dtype=torch.float32)
    )
    assert payload["swa"]["k"] == 3


def test_shape_mismatch_refusal(tmp_path: Path) -> None:
    """A peak with a differently shaped tensor → loud error, no output."""
    lane = tmp_path / "lane"
    lane.mkdir()
    _write_peak(lane, "peak_e0001.pt", 1.0, epoch=1)
    _write_peak(lane, "peak_e0002.pt", 2.0, epoch=2)

    # Build a bad peak with a different conv shape.
    bad = _make_payload(3.0, epoch=3)
    bad["model_state_dict"]["conv.weight"] = torch.full(
        (8, 3, 3, 3), 3.0, dtype=torch.float32  # 8 filters, not 4
    )
    bad_path = lane / "peak_e0003.pt"
    torch.save(bad, bad_path)

    out = lane / "peak_swa.pt"
    with pytest.raises(ValueError, match="shape mismatch"):
        swa_average.main(
            ["--lane-dir", str(lane), "--k", "3", "--output", str(out)]
        )
    assert not out.exists(), "No output should be written on shape mismatch"


def test_round_trip_loads_via_trainer_loader(tmp_path: Path) -> None:
    """SWA output must load via the same loader the trainer/eval uses
    (gomoku.model.load_checkpoint).
    """
    from gomoku.model import build_model, save_checkpoint, load_checkpoint

    lane = tmp_path / "lane"
    lane.mkdir()

    # Two real tiny-model checkpoints with deterministic params.
    torch.manual_seed(0)
    m1 = build_model("tiny")
    save_checkpoint(
        str(lane / "peak_e0010.pt"), m1, epoch=10, total_games=10_000
    )
    torch.manual_seed(1)
    m2 = build_model("tiny")
    save_checkpoint(
        str(lane / "peak_e0020.pt"), m2, epoch=20, total_games=20_000
    )

    out = lane / "peak_swa.pt"
    rc = swa_average.main(
        ["--lane-dir", str(lane), "--k", "2", "--output", str(out)]
    )
    assert rc == 0

    # Loads via the standard loader without modification.
    loaded_model, payload = load_checkpoint(str(out), device="cpu")
    assert payload["swa"]["k"] == 2
    assert payload["epoch"] == 20  # metadata from last peak

    # And the loaded params are the elementwise mean of m1's & m2's params.
    m1_params = dict(m1.named_parameters())
    m2_params = dict(m2.named_parameters())
    for name, p in loaded_model.named_parameters():
        expected = (m1_params[name].detach() + m2_params[name].detach()) / 2.0
        assert torch.allclose(p.detach(), expected, atol=1e-6, rtol=1e-6), (
            f"param {name} mismatch with computed mean"
        )


def test_no_peaks_returns_error(tmp_path: Path) -> None:
    lane = tmp_path / "empty_lane"
    lane.mkdir()
    rc = swa_average.main(["--lane-dir", str(lane), "--k", "3"])
    assert rc == 1


def test_excludes_prior_swa_outputs(tmp_path: Path) -> None:
    """Running the tool twice in a row must not feed peak_swa.pt back in."""
    lane = tmp_path / "lane"
    lane.mkdir()
    _write_peak(lane, "peak_e0001.pt", 1.0, epoch=1)
    _write_peak(lane, "peak_e0002.pt", 3.0, epoch=2)

    out = lane / "peak_swa.pt"
    rc = swa_average.main(
        ["--lane-dir", str(lane), "--k", "2", "--output", str(out)]
    )
    assert rc == 0
    # Now there's a peak_swa.pt in the dir; re-running should still find
    # exactly the two original peaks, not three.
    peaks = swa_average.discover_peaks(lane)
    assert [p.name for p in peaks] == ["peak_e0001.pt", "peak_e0002.pt"]


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    lane = tmp_path / "lane"
    lane.mkdir()
    _write_peak(lane, "peak_e0001.pt", 1.0, epoch=1)
    _write_peak(lane, "peak_e0002.pt", 3.0, epoch=2)
    out = lane / "peak_swa.pt"
    rc = swa_average.main(
        ["--lane-dir", str(lane), "--k", "2", "--output", str(out), "--dry-run"]
    )
    assert rc == 0
    assert not out.exists()


def test_metadata_swa_field_serializable(tmp_path: Path) -> None:
    """The swa metadata block should be JSON-serializable (for logs/dashboards)."""
    lane = tmp_path / "lane"
    lane.mkdir()
    _write_peak(lane, "peak_e0001.pt", 1.0, epoch=1)
    _write_peak(lane, "peak_e0002.pt", 3.0, epoch=2)
    out = lane / "peak_swa.pt"
    rc = swa_average.main(
        ["--lane-dir", str(lane), "--k", "2", "--output", str(out)]
    )
    assert rc == 0
    payload = torch.load(out, map_location="cpu", weights_only=False)
    # Must round-trip through JSON (no tensors / non-JSON types).
    blob = json.dumps(payload["swa"])
    assert "uniform_mean" in blob
