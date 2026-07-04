"""Warm-start layering of the moonshot aux VCT-defense head onto a champion
that predates it (the Bruce/idx-2 pivot).

The pivot resumes a 15x15 champion whose saved config has aux_vct absent/False
and layers the VCT-defense head on at the wound. load_checkpoint(...,
force_aux_vct=True) must build the model WITH the head, load the core strict,
and splice the fresh vct_* params in. With force_aux_vct=False (default) the
load path is byte-identical to before the head existed.
"""

import torch

from gomoku.game import GameState
from gomoku.model import build_model, load_checkpoint


def _input(batch: int = 2) -> torch.Tensor:
    s = GameState.initial()
    x = torch.from_numpy(s.to_planes()).unsqueeze(0)
    return x.repeat(batch, 1, 1, 1)


def _save_champion_without_vct(tmp_path):
    # Simulate the champion: a model built WITHOUT the aux VCT head, whose saved
    # config also lacks aux_vct (a pre-moonshot checkpoint like Bruce).
    m = build_model("tiny")
    assert m.cfg.aux_vct is False
    assert not any(k.startswith("vct_") for k in m.state_dict())
    payload = {
        "model_state_dict": m.state_dict(),
        "model_config": {**m.cfg.__dict__},
        "epoch": 605,
        "total_games": 35432,
    }
    payload["model_config"].pop("aux_vct", None)
    p = tmp_path / "champion.pt"
    torch.save(payload, str(p))
    return p, m


def test_force_aux_vct_layers_head_on_pre_vct_checkpoint(tmp_path):
    p, champ = _save_champion_without_vct(tmp_path)

    m2, _ = load_checkpoint(str(p), force_aux_vct=True)
    # Head is now present.
    assert m2.cfg.aux_vct is True
    vct_keys = [k for k in m2.state_dict() if k.startswith("vct_")]
    assert vct_keys, "expected fresh vct_* keys after force_aux_vct load"
    # Core loaded STRICT (byte-identical to the champion's weights).
    champ_sd = champ.state_dict()
    for k, t in champ_sd.items():
        assert torch.equal(m2.state_dict()[k], t), f"core weight {k} not loaded"
    # Head is usable.
    m2.eval()
    with torch.no_grad():
        out = m2(_input(2), return_vct=True)
    assert len(out) == 3
    assert out[-1].shape[0] == 2  # (batch, N_ACTIONS) vct map


def test_force_aux_vct_default_off_is_byte_identical(tmp_path):
    p, _ = _save_champion_without_vct(tmp_path)
    m2, _ = load_checkpoint(str(p))  # default force_aux_vct=False
    assert m2.cfg.aux_vct is False
    assert not any(k.startswith("vct_") for k in m2.state_dict())


def test_force_aux_vct_still_raises_on_missing_core_weight(tmp_path):
    # Layering the head must not mask a genuinely-missing CORE weight.
    m = build_model("tiny")
    sd = m.state_dict()
    del sd["policy_fc.weight"]
    payload = {"model_state_dict": sd, "model_config": {**m.cfg.__dict__}}
    p = tmp_path / "broken.pt"
    torch.save(payload, str(p))
    try:
        load_checkpoint(str(p), force_aux_vct=True)
    except (RuntimeError, KeyError):
        pass
    else:
        raise AssertionError("expected a missing core weight to raise on load")
