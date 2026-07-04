"""Line-potential input planes (issue #107, the sound-world moonshot).

Validates the vectorized shifted-sum implementation against a naive
per-cell/per-window reference, checks the hand-readable threat semantics
(win-in-one == 1.0, dead line == 0, double threat == two channels hot at one
cell), and asserts the model-side contract: cfg.line_planes off is
byte-identical (stem in_channels unchanged), on widens the stem by 8 while the
EXTERNAL input stays 17 planes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gomoku.features import _DIRS, N_LINE_PLANES, WIN_LEN, line_potential_planes
from gomoku.game import HISTORY_PLY, N_INPUT_PLANES


def naive_line_planes(me: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Reference: per-cell max over live 5-windows, straight from the spec."""
    n = me.shape[0]
    out = np.zeros((N_LINE_PLANES, n, n), dtype=np.float32)
    for p_idx, (stones, blockers) in enumerate(((me, opp), (opp, me))):
        for d_idx, (dr, dc) in enumerate(_DIRS):
            ch = p_idx * len(_DIRS) + d_idx
            for r in range(n):
                for c in range(n):
                    best = 0
                    for k in range(WIN_LEN):
                        ar, ac = r - k * dr, c - k * dc
                        er, ec = ar + (WIN_LEN - 1) * dr, ac + (WIN_LEN - 1) * dc
                        if not (0 <= ar < n and 0 <= ac < n
                                and 0 <= er < n and 0 <= ec < n):
                            continue
                        cells = [(ar + j * dr, ac + j * dc) for j in range(WIN_LEN)]
                        if any(blockers[x, y] for x, y in cells):
                            continue
                        best = max(best, sum(int(stones[x, y]) for x, y in cells))
                    out[ch, r, c] = min(best, 4) / 4.0
    return out


@pytest.mark.parametrize("n", [9, 11])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_naive_reference_on_random_boards(n, seed):
    rng = np.random.default_rng(seed)
    # Random non-overlapping stone sets, roughly a midgame density.
    cells = rng.permutation(n * n)
    n_me, n_opp = n * n // 6, n * n // 6
    me = np.zeros((n, n), dtype=bool)
    opp = np.zeros((n, n), dtype=bool)
    me.reshape(-1)[cells[:n_me]] = True
    opp.reshape(-1)[cells[n_me:n_me + n_opp]] = True

    got = line_potential_planes(
        torch.from_numpy(me.astype(np.float32))[None],
        torch.from_numpy(opp.astype(np.float32))[None],
    )[0].numpy()
    want = naive_line_planes(me, opp)
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_empty_board_is_all_zero():
    z = torch.zeros((1, 9, 9))
    assert line_potential_planes(z, z).abs().sum() == 0.0


def test_win_in_one_reads_full_potential():
    # my stones at (4, 1..4): both flanking empties complete five -> me_H == 1.0
    me = torch.zeros((1, 9, 9))
    me[0, 4, 1:5] = 1.0
    opp = torch.zeros((1, 9, 9))
    feat = line_potential_planes(me, opp)[0]
    assert feat[0, 4, 0] == 1.0 and feat[0, 4, 5] == 1.0    # me_H channel
    assert feat[4].max() == 0.0                              # opp channels silent


def test_blocked_line_is_dead():
    # my (4, 0..3), opp at (4, 4): the ONLY H 5-window through them is blocked.
    me = torch.zeros((1, 9, 9))
    me[0, 4, 0:4] = 1.0
    opp = torch.zeros((1, 9, 9))
    opp[0, 4, 4] = 1.0
    feat = line_potential_planes(me, opp)[0]
    assert feat[0, 4, 0:5].max() == 0.0                      # me_H dead on the line


def test_double_threat_is_two_channels_hot_at_one_cell():
    # Open three on the row and open three on the column, crossing at (4, 5):
    # the claw-style conjunction is a LOCAL read of two direction channels.
    me = torch.zeros((1, 9, 9))
    me[0, 4, 1:4] = 1.0        # (4,1),(4,2),(4,3) horizontal
    me[0, 1:4, 5] = 1.0        # (1,5),(2,5),(3,5) vertical
    feat = line_potential_planes(me, torch.zeros((1, 9, 9)))[0]
    assert feat[0, 4, 5] == 0.75                             # me_H: 3 stones / 4
    assert feat[1, 4, 5] == 0.75                             # me_V: 3 stones / 4


def test_model_stem_width_and_forward():
    from gomoku.model import GomokuNet, ModelConfig

    off = GomokuNet(ModelConfig(n_filters=16, n_blocks=1, value_hidden=16))
    on = GomokuNet(ModelConfig(n_filters=16, n_blocks=1, value_hidden=16,
                               line_planes=True))
    assert off.stem[0].in_channels == N_INPUT_PLANES
    assert on.stem[0].in_channels == N_INPUT_PLANES + N_LINE_PLANES
    # Both consume the SAME external 17-plane input; expansion is in-forward.
    x = torch.rand(2, N_INPUT_PLANES, off.cfg.board_size, off.cfg.board_size)
    x[:, 0] = (x[:, 0] > 0.9).float()
    x[:, HISTORY_PLY] = ((x[:, HISTORY_PLY] > 0.9) & (x[:, 0] == 0)).float()
    for m in (off, on):
        m.eval()
        with torch.no_grad():
            p, v = m(x)
        assert p.shape == (2, m.cfg.board_size ** 2) and v.shape == (2,)
    # Off-model state_dict has no line-planes trace (pure config lever).
    assert "line_planes" not in str(list(off.state_dict().keys()))


def test_checkpoint_roundtrip_carries_line_planes(tmp_path):
    from gomoku.model import GomokuNet, ModelConfig, load_checkpoint, save_checkpoint

    m = GomokuNet(ModelConfig(n_filters=16, n_blocks=1, value_hidden=16,
                              line_planes=True))
    path = str(tmp_path / "lp.pt")
    save_checkpoint(path, m)
    loaded, _ = load_checkpoint(path)
    assert loaded.cfg.line_planes is True
    assert loaded.stem[0].in_channels == N_INPUT_PLANES + N_LINE_PLANES
