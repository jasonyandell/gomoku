"""Line-potential input feature planes (issue #107, the sound-world moonshot).

Gomoku is a game of lines: every win is five-in-a-row along one of 4
directions, and every threat is a live 5-window filling up. The raw
stone/history planes force the conv tower to rediscover that geometry through
stacked 3x3 kernels — and the claw result (wiki/topics/the-claw.md) shows
cross-line conjunctions (double threats) are effectively invisible to it.
These planes hand the tower the geometry so it only has to learn the
chemistry.

For each cell c, direction d in {H, V, diag, anti-diag} and player
P in {side-to-move, opponent}:

    feat[P, d, c] = max over the 5-windows w along d that contain c of
                      (# P stones in w) / 4   if w is fully on-board and
                                              contains no opposing stones
                      0                       otherwise (dead or off-board)

Semantics worth knowing:
  * An empty cell with feat[me, d] == 1.0 completes five-in-a-row there.
  * An empty cell with feat[opp, d] == 1.0 is an opponent win-in-one (block!).
  * A double threat reads as TWO direction channels hot at the SAME cell — a
    purely local conjunction a 1x1 conv can see, instead of a long-range
    relation between two 5-cell spans through an intersection.

The planes are a pure function of the CURRENT stone planes (0 and
HISTORY_PLY), so they are computed inside GomokuNet.forward()
(cfg.line_planes) rather than stored: records, the replay buffer, D4
augmentation, and the native paths all keep the 17-plane external contract,
and augmentation consistency is automatic (the potentials are derived from
already-augmented stones — rotating the board correctly re-routes H stones
into the V channel with no explicit channel permutation anywhere).
"""
from __future__ import annotations

import torch

# Same 4 line directions as gomoku.game._DIRS (the other 4 are negatives).
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
WIN_LEN = 5
N_LINE_PLANES = 2 * len(_DIRS)          # {me, opp} x 4 directions = 8


def _shift2d(x: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    """Translate (B, N, N) by (dr, dc) with zero fill: out[r, c] = x[r-dr, c-dc]."""
    B, H, W = x.shape
    if abs(dr) >= H or abs(dc) >= W:
        return x.new_zeros(x.shape)
    out = x.new_zeros(x.shape)
    src_r = slice(max(0, -dr), H - max(0, dr))
    src_c = slice(max(0, -dc), W - max(0, dc))
    dst_r = slice(max(0, dr), H - max(0, -dr))
    dst_c = slice(max(0, dc), W - max(0, -dc))
    out[:, dst_r, dst_c] = x[:, src_r, src_c]
    return out


def line_potential_planes(
    stones_me: torch.Tensor, stones_opp: torch.Tensor
) -> torch.Tensor:
    """Compute the 8 line-potential planes from the two current stone planes.

    Args are (B, N, N) float tensors of 0/1 stone indicators (input planes 0
    and HISTORY_PLY). Returns (B, 8, N, N) float in [0, 1], channel order
    [me_H, me_V, me_diag, me_anti, opp_H, opp_V, opp_diag, opp_anti].

    Pure shifted-sum implementation: for a direction d, the 5-window anchored
    at cell a covers a, a+d, ..., a+4d, so
      window_sum(a)   = sum_k shift(stones, -k*d)(a)
      on_board(a)     = sum_k shift(ones,   -k*d)(a) == 5
      pot(a)          = window_sum(a) if live (no blockers, on-board) else 0
      feat(c)         = max_k shift(pot, +k*d)(c)     (max over anchors thru c)
    All ops are plain tensor shifts/sums/maxes — batched, differentiability
    irrelevant (inputs are data), MPS/torch.compile friendly.
    """
    ones = stones_me.new_ones((1,) + stones_me.shape[1:])
    feats = []
    for stones, blockers in ((stones_me, stones_opp), (stones_opp, stones_me)):
        for dr, dc in _DIRS:
            s_sum = torch.zeros_like(stones)
            b_sum = torch.zeros_like(stones)
            v_sum = torch.zeros_like(ones)
            for k in range(WIN_LEN):
                s_sum = s_sum + _shift2d(stones, -k * dr, -k * dc)
                b_sum = b_sum + _shift2d(blockers, -k * dr, -k * dc)
                v_sum = v_sum + _shift2d(ones, -k * dr, -k * dc)
            live = (b_sum == 0) & (v_sum == WIN_LEN)   # broadcast (1,N,N)->(B,N,N)
            pot = s_sum * live.to(stones.dtype)
            feat = pot
            for k in range(1, WIN_LEN):
                feat = torch.maximum(feat, _shift2d(pot, k * dr, k * dc))
            feats.append(feat)
    # A live window holds at most 4 stones mid-game (5 = already won; such
    # boards are never evaluated, but clamp for safety on synthetic inputs).
    return torch.stack(feats, dim=1).clamp_(max=4.0).div_(4.0)


def expand_input_planes(x: torch.Tensor, history_ply: int) -> torch.Tensor:
    """Append the 8 line-potential planes to a (B, P, N, N) input-plane stack.

    Plane 0 = side-to-move stones, plane `history_ply` = opponent stones (the
    to_planes() contract). Returns (B, P + N_LINE_PLANES, N, N).
    """
    extra = line_potential_planes(x[:, 0], x[:, history_ply])
    return torch.cat([x, extra], dim=1)
