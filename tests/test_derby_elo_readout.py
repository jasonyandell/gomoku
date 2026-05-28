"""Tests for the derby's authoritative-elo wandb readout (bug derby-i5j).

Root cause being guarded against: in Derby v9 the wandb dashboard is fed by the
trainer tailing the in-cell ``eval_results.jsonl`` stream. For a big net the CPU
eval_worker can't keep pace, so that stream FREEZES (e.g. ``large`` stuck at elo
788 @ epoch 45 while the trainer is at epoch 242+) — a "mirage". The per-chunk
TRUTH lives in ``derby_state.json`` (peak ~1455). The fix logs that authoritative
``model_elo`` straight into the SAME (resumed) wandb run under a ``derby/*``
namespace, and emits an "eval lagging by N epochs" warning when the eval stream
trails the trainer.

These tests are CPU-only and never hit wandb (the wandb module is monkeypatched
with a recording fake) or any GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.delo_derby import (
    EVAL_LAG_WARN_EPOCHS,
    DERBY_ELO_WANDB_KEY,
    DERBY_EPOCH_WANDB_KEY,
    DERBY_PEAK_WANDB_KEY,
    DERBY_EVAL_LAG_WANDB_KEY,
    compute_eval_lag,
    log_authoritative_elo_to_wandb,
    read_last_elo,
)
import scripts.delo_derby as derby


# ---------------------------------------------------------------------------
# A recording fake for `wandb` — proves what was logged without any network.
# ---------------------------------------------------------------------------

class _FakeRun:
    def __init__(self, recorder):
        self._recorder = recorder
        self.finished = False

    def log(self, payload):
        self._recorder["logs"].append(dict(payload))

    def finish(self):
        self.finished = True


class _FakeWandb:
    def __init__(self):
        self.recorder = {"init_kwargs": None, "logs": [], "runs": []}

    def init(self, **kwargs):
        self.recorder["init_kwargs"] = dict(kwargs)
        run = _FakeRun(self.recorder)
        self.recorder["runs"].append(run)
        return run


@pytest.fixture
def fake_wandb(monkeypatch):
    """Install a fake `wandb` module so `import wandb` inside the function under
    test returns our recorder instead of hitting the network."""
    fw = _FakeWandb()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fw)
    return fw


# ---------------------------------------------------------------------------
# Fixtures: a derby_state.json mirroring the live v9 "large" lane, and an eval
# stream that LAGS behind it (the bug shape).
# ---------------------------------------------------------------------------

@pytest.fixture
def board():
    # v2 board: global.wandb is None (the run_sweep CELLS own wandb), so the
    # authoritative push must key off the embedded wandb_run_id, not global.wandb.
    return {
        "global": {
            "engine": "run_sweep_wall_slice",
            "base_out_dir": "sweep_runs/derby_v9",
            "wandb": None,
        },
        "ideas": [
            {"name": "small", "cell": "derby-v9-small", "cell_name": "derby-v9-small"},
            {"name": "large", "cell": "derby-v9-large", "cell_name": "derby-v9-large"},
        ],
    }


@pytest.fixture
def lagging_eval_jsonl(tmp_path):
    """An eval_results.jsonl whose LAST line is the frozen mirage (e45 / 788),
    matching the bug: the eval stream stalled while the trainer raced ahead."""
    p = tmp_path / "eval_results.jsonl"
    rows = [
        {"ts": 1.0, "eval_worker/epoch_evaluated": 20, "eval/model_elo": 600.0},
        {"ts": 2.0, "eval_worker/epoch_evaluated": 45, "eval/model_elo": 788.0},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


# ---------------------------------------------------------------------------
# compute_eval_lag — pure/stateless lag math
# ---------------------------------------------------------------------------

def test_compute_eval_lag_basic():
    # Trainer at 242, eval stuck at 45 -> lag 197 (the live bug magnitude).
    assert compute_eval_lag(242, 45) == 197


def test_compute_eval_lag_clamps_negative_to_zero():
    # Eval "ahead" of a stale latest.pt is not a lag.
    assert compute_eval_lag(45, 50) == 0


def test_compute_eval_lag_none_when_unknown():
    assert compute_eval_lag(None, 45) is None
    assert compute_eval_lag(242, None) is None


# ---------------------------------------------------------------------------
# (b) Authoritative readout: the derby_state elo (NOT the lagging stream) is what
# gets logged to the resumed wandb run.
# ---------------------------------------------------------------------------

def test_authoritative_elo_logged_to_resumed_run(board, fake_wandb):
    # derby_state's authoritative per-chunk truth for the big net.
    authoritative_elo = 1454.8187
    authoritative_peak = 1454.8187
    eval_epoch = 242
    run_id = "z0i6qs0x"

    logged = log_authoritative_elo_to_wandb(
        board, "large",
        model_elo=authoritative_elo,
        eval_epoch=eval_epoch,
        peak_elo=authoritative_peak,
        wandb_run_id=run_id,
        trainer_epoch=242,  # in sync here; lag path tested separately
    )

    assert logged is True
    # Eval ELOs go to a SEPARATE "<run>-eval" run, NEVER the trainer's run id
    # (Jason 2026-05-27). Writing into the trainer's run concurrently with the next
    # slice resuming it was the "run ID <x> is in use" crash-loop; a derived eval run
    # the trainer never touches removes the collision entirely.
    init = fake_wandb.recorder["init_kwargs"]
    assert init["id"] == f"{run_id}-eval"
    assert init["id"] != run_id  # must NOT collide with the training run
    assert init["resume"] == "allow"
    assert init["project"] == "gomoku"
    # Exactly one log call carrying the AUTHORITATIVE elo (1454, not the 788 mirage).
    assert len(fake_wandb.recorder["logs"]) == 1
    payload = fake_wandb.recorder["logs"][0]
    assert payload[DERBY_ELO_WANDB_KEY] == pytest.approx(authoritative_elo)
    assert payload[DERBY_PEAK_WANDB_KEY] == pytest.approx(authoritative_peak)
    assert payload[DERBY_EPOCH_WANDB_KEY] == eval_epoch
    # And the run was finished (no dangling writer on the resumed id).
    assert fake_wandb.recorder["runs"][0].finished is True


def test_authoritative_elo_overrides_the_lagging_stream(board, fake_wandb, lagging_eval_jsonl):
    """End-to-end intent: read_last_elo would surface the FROZEN 788@e45 from the
    eval stream, but the value pushed to wandb is the authoritative derby_state
    elo — proving the dashboard reflects derby_state, not the mirage."""
    # The lagging stream's last line is the mirage:
    parsed = derby._read_elo_from_jsonl(lagging_eval_jsonl)
    assert parsed is not None
    stream_elo, stream_epoch = parsed
    assert stream_elo == 788.0 and stream_epoch == 45

    authoritative_elo = 1454.8187  # from derby_state.json
    log_authoritative_elo_to_wandb(
        board, "large",
        model_elo=authoritative_elo,
        eval_epoch=242,
        peak_elo=authoritative_elo,
        wandb_run_id="z0i6qs0x",
        trainer_epoch=242,
    )
    payload = fake_wandb.recorder["logs"][0]
    # The wandb value is the authoritative elo, NOT the frozen stream value.
    assert payload[DERBY_ELO_WANDB_KEY] == pytest.approx(authoritative_elo)
    assert payload[DERBY_ELO_WANDB_KEY] != stream_elo


def test_no_wandb_log_without_run_id(board, fake_wandb):
    logged = log_authoritative_elo_to_wandb(
        board, "large",
        model_elo=1454.0, eval_epoch=242, peak_elo=1454.0,
        wandb_run_id=None, trainer_epoch=242,
    )
    assert logged is False
    assert fake_wandb.recorder["init_kwargs"] is None  # never touched wandb


def test_escape_hatch_disables_wandb_push(board, fake_wandb):
    board["global"]["derby_wandb_elo"] = False
    logged = log_authoritative_elo_to_wandb(
        board, "large",
        model_elo=1454.0, eval_epoch=242, peak_elo=1454.0,
        wandb_run_id="z0i6qs0x", trainer_epoch=242,
    )
    assert logged is False
    assert fake_wandb.recorder["init_kwargs"] is None


def test_wandb_failure_is_swallowed(board, monkeypatch):
    """A wandb hiccup must NEVER kill the race — returns False, doesn't raise."""
    class _BoomWandb:
        def init(self, **kwargs):
            raise RuntimeError("network down")
    monkeypatch.setitem(__import__("sys").modules, "wandb", _BoomWandb())
    logged = log_authoritative_elo_to_wandb(
        board, "large",
        model_elo=1454.0, eval_epoch=242, peak_elo=1454.0,
        wandb_run_id="z0i6qs0x", trainer_epoch=242,
    )
    assert logged is False  # swallowed, no exception


# ---------------------------------------------------------------------------
# (c) Lag warning: fires when the eval epoch trails the trainer past threshold.
# ---------------------------------------------------------------------------

def test_lag_warning_fires_past_threshold(board, fake_wandb, tmp_path, capsys):
    milestones = tmp_path / "derby_milestones.log"
    # Eval stuck at 45 while trainer at 242 -> lag 197 >> EVAL_LAG_WARN_EPOCHS.
    log_authoritative_elo_to_wandb(
        board, "large",
        model_elo=1454.0, eval_epoch=45, peak_elo=1454.0,
        wandb_run_id="z0i6qs0x", trainer_epoch=242,
        milestones=milestones,
    )
    out = capsys.readouterr().out
    assert "EVAL-LAG" in out
    assert "MIRAGE" in out
    # Written to the milestones log too.
    assert milestones.exists()
    assert "EVAL-LAG large" in milestones.read_text()
    # The lag is also pushed to wandb as a metric so it's chart-visible.
    payload = fake_wandb.recorder["logs"][0]
    assert payload[DERBY_EVAL_LAG_WANDB_KEY] == 197


def test_no_lag_warning_when_in_sync(board, fake_wandb, tmp_path, capsys):
    milestones = tmp_path / "derby_milestones.log"
    # small/medium track fine: eval epoch ~= trainer epoch -> NO warning.
    log_authoritative_elo_to_wandb(
        board, "small",
        model_elo=1318.0, eval_epoch=453, peak_elo=1318.0,
        wandb_run_id="b713kz9l", trainer_epoch=453,
        milestones=milestones,
    )
    out = capsys.readouterr().out
    assert "EVAL-LAG" not in out
    assert not (milestones.exists() and "EVAL-LAG" in milestones.read_text())
    # Lag metric still logged (== 0) — a healthy zero, not a warning.
    payload = fake_wandb.recorder["logs"][0]
    assert payload[DERBY_EVAL_LAG_WANDB_KEY] == 0


def test_lag_just_below_threshold_does_not_warn(board, fake_wandb, capsys):
    # Boundary: lag == threshold-1 must NOT warn.
    log_authoritative_elo_to_wandb(
        board, "small",
        model_elo=1300.0, eval_epoch=100, peak_elo=1300.0,
        wandb_run_id="b713kz9l", trainer_epoch=100 + EVAL_LAG_WARN_EPOCHS - 1,
    )
    assert "EVAL-LAG" not in capsys.readouterr().out


def test_lag_at_threshold_warns(board, fake_wandb, capsys):
    # Boundary: lag == threshold must warn (>= comparison).
    log_authoritative_elo_to_wandb(
        board, "small",
        model_elo=1300.0, eval_epoch=100, peak_elo=1300.0,
        wandb_run_id="b713kz9l", trainer_epoch=100 + EVAL_LAG_WARN_EPOCHS,
    )
    assert "EVAL-LAG" in capsys.readouterr().out


def test_lag_warning_fires_even_without_run_id(board, fake_wandb, tmp_path, capsys):
    """The warning needs no wandb — a frozen readout is visible in logs even if the
    run id is missing (no wandb push then, but the operator still sees the lag)."""
    milestones = tmp_path / "derby_milestones.log"
    logged = log_authoritative_elo_to_wandb(
        board, "large",
        model_elo=1454.0, eval_epoch=45, peak_elo=1454.0,
        wandb_run_id=None, trainer_epoch=242,
        milestones=milestones,
    )
    assert logged is False  # no run id -> no wandb push
    assert "EVAL-LAG" in capsys.readouterr().out  # but the warning still fired
    assert "EVAL-LAG large" in milestones.read_text()


# ---------------------------------------------------------------------------
# Regression guard: read_last_elo still surfaces the stream's last elo (small/
# medium track fine — the readout path for healthy lanes is unchanged).
# ---------------------------------------------------------------------------

def test_read_last_elo_unchanged_for_healthy_stream(tmp_path, monkeypatch):
    # Build a healthy stream where the last line is the freshest, in-sync elo.
    ckpt = tmp_path / "sweep_runs" / "derby-v9-small" / "checkpoints"
    ckpt.mkdir(parents=True)
    rows = [
        {"ts": 1.0, "eval_worker/epoch_evaluated": 400, "eval/model_elo": 1280.0},
        {"ts": 2.0, "eval_worker/epoch_evaluated": 453, "eval/model_elo": 1318.0},
    ]
    (ckpt / "eval_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    # Point REPO_ROOT-relative resolution at our tmp tree.
    monkeypatch.setattr(derby, "REPO_ROOT", tmp_path)
    board = {
        "global": {"engine": "run_sweep_wall_slice", "base_out_dir": "sweep_runs/derby_v9"},
        "ideas": [{"name": "small", "cell": "derby-v9-small", "cell_name": "derby-v9-small"}],
    }
    parsed = read_last_elo(board, "small")
    assert parsed == (1318.0, 453)
