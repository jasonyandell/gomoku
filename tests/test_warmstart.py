"""Cross-board warm-start loader tests (scripts/warmstart_15x15.py).

Board size is a process-level constant, so these run in a 15x15 process:

    GOMOKU_BOARD_SIZE=15 PYTHONPATH=$PWD pytest tests/test_warmstart.py -q

A 9x9 *source* net is still constructible inside a 15x15 process (cfg drives
all head shapes — see test_board15.py), so we build + save a 9x9 small+global_pool
champion-shaped net, warm-start it into a fresh 15x15 net, and assert:
  * every conv-tower tensor transferred BYTE-IDENTICALLY from the source;
  * the three board-bound FCs (policy_fc.weight/bias, value_fc1.weight) were
    NOT taken from the source (fresh init) and have the correct 15x15 shapes;
  * the saved seed loads via load_checkpoint at 15x15 and forward() yields
    (policy[225], value) of the right shape;
  * the seed carries a board_size=15 config and NO buffer / NO optimizer /
    NO wandb_run_id (fresh, resumable timeline).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from gomoku import game
from gomoku.model import GomokuNet, ModelConfig, load_checkpoint, save_checkpoint

pytestmark = pytest.mark.skipif(
    game.BOARD_SIZE != 15,
    reason="warm-start tests need GOMOKU_BOARD_SIZE=15 (board size is per-process)",
)

N = game.BOARD_SIZE

# Import the script as a module (it lives in scripts/, not the package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "warmstart_15x15.py"
_spec = importlib.util.spec_from_file_location("warmstart_15x15", _SCRIPT)
warmstart_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(warmstart_mod)

# The exact source-net architecture of the v8 'mate-discount' champion:
# small (64ch x 4blk, value_hidden 64), stem_padding=1, global_pool=True,
# scalar value head, relu. Constructed at board_size=9 inside this 15x15 process.
_CHAMPION_CFG_9 = ModelConfig(
    board_size=9,
    n_filters=64,
    n_blocks=4,
    value_hidden=64,
    stem_padding=1,
    global_pool=True,
    value_head="scalar",
    activation="relu",
)

# Tensors whose shape is board-bound and must be re-initialised at 15x15.
_BOARD_BOUND = {"policy_fc.weight", "policy_fc.bias", "value_fc1.weight"}


def _make_source_checkpoint(path: str) -> dict:
    torch.manual_seed(0)
    src = GomokuNet(_CHAMPION_CFG_9)
    save_checkpoint(path, src, epoch=1764, total_games=28632,
                    wandb_run_id="c1x7urjm")
    return src.state_dict()


def test_warmstart_transfers_tower_and_reinits_fcs(tmp_path):
    src_path = str(tmp_path / "champ9.pt")
    out_path = str(tmp_path / "g15_seed.pt")
    src_state = _make_source_checkpoint(src_path)

    summary = warmstart_mod.warmstart(
        src_path, out_path, size="small", board_size=15
    )

    # --- transfer accounting ----------------------------------------------
    assert summary["target_board"] == 15
    assert summary["src_board"] == 9
    # Exactly the three board-bound FCs are skipped; nothing else.
    assert set(summary["skipped_shape"]) == _BOARD_BOUND
    assert summary["skipped_missing"] == []
    assert summary["n_transferred"] == summary["n_src_keys"] - len(_BOARD_BOUND)
    # The champion's FC layers are small relative to its conv tower, so the
    # overwhelming majority of params transfer.
    assert summary["pct_params_transferred"] > 90.0

    # --- saved seed exists and loads at 15x15 -----------------------------
    payload = torch.load(out_path, map_location="cpu", weights_only=False)
    assert payload["model_config"]["board_size"] == 15
    # Fresh, resumable timeline: no buffer, no optimizer, no wandb id.
    assert "replay_buffer" not in payload
    assert "optimizer_state_dict" not in payload
    assert "wandb_run_id" not in payload
    assert payload["epoch"] == 0
    assert payload["total_games"] == 0

    seed_state = payload["model_state_dict"]

    # --- conv-tower tensors are BYTE-IDENTICAL to the source --------------
    tower_prefixes = ("stem.", "tower.", "policy_conv", "policy_bn",
                      "value_conv", "value_bn")
    n_tower_checked = 0
    for k, src_t in src_state.items():
        if k in _BOARD_BOUND:
            continue
        # Same-shaped non-board-bound keys must equal the source exactly.
        assert k in seed_state, f"missing transferred key {k}"
        assert torch.equal(seed_state[k], src_t), f"tensor {k} was not byte-identical"
        if k.startswith(tower_prefixes):
            n_tower_checked += 1
    # Sanity: we actually verified a non-trivial tower (stem + 4 blocks incl.
    # 2 global-pool blocks + policy/value convs+bns).
    assert n_tower_checked > 40

    # value_fc1.bias and value_fc2.* are NOT board-bound -> transferred too.
    for k in ("value_fc1.bias", "value_fc2.weight", "value_fc2.bias"):
        assert torch.equal(seed_state[k], src_state[k])

    # --- the three FCs are fresh init + correctly 15x15-shaped ------------
    spatial = 15 + 2 * 1 - 3 + 1  # stem_padding=1, kernel 3 -> 15
    assert tuple(seed_state["policy_fc.weight"].shape) == (225, 2 * spatial * spatial)
    assert tuple(seed_state["policy_fc.bias"].shape) == (225,)
    assert tuple(seed_state["value_fc1.weight"].shape) == (64, 1 * spatial * spatial)
    # Fresh init must differ from a (shape-incompatible) source anyway; assert it
    # is NOT all-zero / NOT a broadcast of the 9x9 tensor by checking it has the
    # 15x15 fan-in and is not trivially constant.
    assert seed_state["policy_fc.weight"].abs().sum() > 0
    assert seed_state["value_fc1.weight"].abs().sum() > 0


def test_warmstart_seed_loads_and_forward_passes(tmp_path):
    src_path = str(tmp_path / "champ9.pt")
    out_path = str(tmp_path / "g15_seed.pt")
    _make_source_checkpoint(src_path)
    warmstart_mod.warmstart(src_path, out_path, size="small", board_size=15)

    # load_checkpoint enforces the 15x15 board-size contract (no expect override).
    model, payload = load_checkpoint(out_path)  # device defaults to cpu
    assert model.cfg.board_size == 15
    assert model.cfg.n_filters == 64
    assert model.cfg.n_blocks == 4
    assert model.cfg.global_pool is True
    assert model.cfg.stem_padding == 1
    assert model.cfg.value_head == "scalar"

    model.eval()
    x = torch.zeros(3, game.N_INPUT_PLANES, N, N)
    with torch.no_grad():
        p, v = model(x)
    assert p.shape == (3, 225)
    assert v.shape == (3,)
    assert torch.all(v.abs() <= 1.0)


def test_warmstart_detects_arch_mismatch(tmp_path):
    """If the source tower can't map onto the target tower (e.g. a different
    n_filters), the loader must refuse rather than write a misleading seed."""
    src_path = str(tmp_path / "wrong_arch9.pt")
    out_path = str(tmp_path / "should_not_exist.pt")
    # A 9x9 net with a DIFFERENT channel count -> the conv tower shapes diverge,
    # so transferring into a small (64ch) target leaves tower tensors unfilled.
    cfg = ModelConfig(board_size=9, n_filters=32, n_blocks=4, value_hidden=64,
                      stem_padding=1, global_pool=True)
    save_checkpoint(src_path, GomokuNet(cfg))
    with pytest.raises(SystemExit, match="ARCH MISMATCH"):
        warmstart_mod.warmstart(src_path, out_path, size="small", board_size=15)
    assert not Path(out_path).exists()
