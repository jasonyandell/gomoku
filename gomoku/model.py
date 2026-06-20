"""Small AlphaZero-style ResNet for gomoku (board size from cfg.board_size).

Input:  (B, N_INPUT_PLANES, board, board) float32 — see GameState.to_planes()
Output: policy logits (B, board*board), value (B,) in [-1, 1] after tanh.

Board size: every spatial/head dimension derives from ``cfg.board_size``
(default = the process-level ``gomoku.board_config.BOARD_SIZE``). Checkpoints
embed the board size in their model_config; ``load_checkpoint`` refuses a
checkpoint whose board size differs from the active process config, and
pre-board-size checkpoints (which were all 9x9) load as board_size=9.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES
from gomoku.swap2 import N_CHOICES


@dataclass
class ModelConfig:
    n_input_planes: int = N_INPUT_PLANES
    # Board side length. The default is the PROCESS-LEVEL board size resolved
    # by gomoku.board_config at import time (9 unless GOMOKU_BOARD_SIZE /
    # --board-size says otherwise), so existing 9x9 construction paths are
    # unchanged. All head/output sizing inside GomokuNet derives from this
    # field, never from the module-level constant, so a config fully describes
    # the net it builds.
    board_size: int = BOARD_SIZE
    n_filters: int = 64
    n_blocks: int = 4
    policy_filters: int = 2
    value_filters: int = 1
    value_hidden: int = 64
    # Stem-conv padding. michaelnny/alpha_zero uses padding=3 with a 3x3 kernel
    # for gomoku ("agent fails to block on edge cases" fix). That expands the
    # feature map from BOARD_SIZE to BOARD_SIZE+4 after the stem, giving the
    # network a "virtual padding zone" around the real board.
    stem_padding: int = 3
    # V3 auxiliary opponent-reply policy head (KataGo-style). When False
    # (default), the head modules are NOT constructed, so state_dict(), the
    # param count, and the inference graph are byte-identical to a model
    # without this field. When True, a structural clone of the policy head
    # taps the same shared tower; it is a TRAINING-ONLY appendage, never run
    # by forward() unless the trainer explicitly passes return_aux=True, and
    # never touched by fuse_model_for_inference. Gated on by the trainer flag
    # --aux-opponent-reply-weight > 0.
    aux_opponent_reply: bool = False
    # V4 auxiliary ownership head (KataGo's biggest sample-efficiency win,
    # adapted to gomoku). When False (default), NOT constructed -> byte-identical
    # state_dict / param count / inference graph. When True, a dense head off the
    # SAME shared tower predicts per-cell final control (81-way). TRAINING-ONLY,
    # dropped at inference, untouched by fuse_model_for_inference. Gated on by the
    # trainer flag --aux-ownership-weight > 0.
    aux_ownership: bool = False
    # KataGo-style global pooling (Derby v4 "Whole-board" lever). When False
    # (default), the residual tower is the exact current arch and the
    # state_dict is byte-identical. When True, the LATTER HALF of residual
    # blocks become GlobalPoolResBlocks that inject a board-global (mean+max
    # over spatial dims -> FC -> per-channel bias) signal into every cell. An
    # int instead of a bool sets how many of the trailing blocks get pooling.
    # Rationale: a tiny 3x3-conv tower's receptive field barely spans 9x9, so
    # it cannot cheaply represent board-global facts ("is there a live-four
    # ANYWHERE?"); global pooling injects that context directly.
    global_pool: bool | int = False
    # Derby 'x-wdl' value-representation lever (bead derby-cgf). Selects how the
    # value head represents the position value:
    #   "scalar" (default) = today's path EXACTLY: value_fc2 -> Linear(hidden,1)
    #     -> tanh, forward() returns a scalar v in [-1,1]. BYTE-IDENTICAL: the WDL
    #     FC is NOT constructed, so the state_dict keys / param count / inference
    #     graph are unchanged from a model without this field.
    #   "wdl" = a categorical {win,draw,loss} head: value_wdl_fc -> Linear(hidden,3)
    #     emits 3 logits; the scalar value consumers see the DERIVED scalar
    #     v = P(win) - P(loss) (softmax over the 3 logits), so MCTS leaf eval and
    #     the anchor-ladder eval are untouched. The 3 logits are exposed only when
    #     the trainer asks (forward(return_value_logits=True)) for the WDL
    #     cross-entropy loss; self-play/eval never request them (zero extra cost).
    #   "hlgauss" = HL-Gauss distributional value (Farebrother+2024, arxiv 2403.03950):
    #     value_hlgauss_fc -> Linear(hidden, N) emits N logits over bin centers in
    #     [-1, 1] (default N=51, paper-standard); trained with cross-entropy against
    #     a Gaussian-smoothed target N(z, sigma^2) discretized onto the bin centers.
    #     The scalar consumers see the DERIVED scalar v = sum(softmax_i * bin_center_i)
    #     (same WDL discipline, generalized to N bins). Generalizes WDL: WDL has 3
    #     hard-coded bins at {-1,0,+1}; HL-Gauss has N evenly-spaced bins and a
    #     Gaussian-smoothed target, giving finer resolution in drawish positions.
    #     The N logits are exposed only when the trainer asks
    #     (forward(return_value_logits=True)) for the HL-Gauss cross-entropy loss;
    #     self-play/eval never request them (zero extra cost on the gen hot path).
    # Mirrors the aux-head gating discipline: a single config field selects which
    # head module is constructed; the unselected one is never created.
    value_head: str = "scalar"
    # HL-Gauss bin count (N) and Gaussian-smoothed-target sigma. Only used when
    # value_head == "hlgauss"; for "scalar"/"wdl" these fields are inert and the
    # constructed modules are byte-identical (the bin-centers buffer / N-bin FC are
    # NOT built unless hlgauss is selected). Defaults: N=51 (Farebrother+2024's
    # paper-standard) and sigma=0.05 (~1.25 bin widths for N=51 over [-1,1]; a
    # sensible default that smooths each target over ~3 adjacent bins so the head
    # has a smooth gradient at every z, without collapsing to a near-delta or
    # spreading mass over the whole range).
    value_hlgauss_bins: int = 51
    value_hlgauss_sigma: float = 0.05
    # Derby 'x-mish' activation-function lever (bead derby-sib). Selects the
    # nonlinearity used for EVERY residual-tower (stem + ResBlock /
    # GlobalPoolResBlock) nonlinearity:
    #   "relu" (default) = today's path EXACTLY: nn.ReLU, byte-identical to a
    #     model predating this field (ReLU has no params, so no state_dict key /
    #     param-count change, and the forward graph is the same math as the old
    #     F.relu calls).
    #   "mish" = nn.Mish, the smooth self-gated nonlinearity Mish(x) =
    #     x*tanh(softplus(x)) (KataGo's newer default). ZERO added params and the
    #     SAME state_dict keys as relu (Mish is parameter-free), so a checkpoint
    #     of matching shape loads either way; only the activation math changes.
    # The value/policy HEAD nonlinearities are deliberately untouched (they stay
    # F.relu) so this lever is exactly "the tower activation" — one axis.
    activation: str = "relu"
    # Swap2 choice head (the swap2 opening protocol's only non-spatial decisions:
    # the responder's STAY/SWAP/PLACE2 and the opener's pick-color). A width-3
    # head (gomoku.swap2.N_CHOICES) emits a prior over those choice slots, used
    # ONLY at choice nodes; the board policy head and value head are unchanged.
    # The head ALWAYS exists (default True) because swap2 is the protocol this
    # net is for; it is small and shares the trunk via the value head's
    # penultimate hidden layer. It is reached only through forward_with_choice()
    # — never through forward() — so the (policy, value) hot path is byte-
    # identical and every existing `policy, value = model(x)` caller is unchanged.
    # When False (e.g. an ablation), the head module is NOT constructed, so the
    # state_dict / param count / inference graph match a model predating it. The
    # warm-start load path tolerates an OLD checkpoint that lacks the choice-head
    # weights (the champion predates this head): the core loads strict and the
    # fresh choice head is left at its init (see load_checkpoint).
    choice_head: bool = True


SIZE_PRESETS: dict[str, ModelConfig] = {
    "tiny":   ModelConfig(n_filters=32, n_blocks=2, value_hidden=32),
    "small":  ModelConfig(n_filters=64, n_blocks=4, value_hidden=64),
    "medium": ModelConfig(n_filters=96, n_blocks=6, value_hidden=128),
    # 96ch x 8blk — the planned first "serious" 15x15 net (wiki:
    # 15x15-era-feasibility-and-plan.md §2: 2.32x the 9x9 champ eval cost at
    # the training wave size). Constructible at any board size.
    "96x8":   ModelConfig(n_filters=96, n_blocks=8, value_hidden=128),
    "large":  ModelConfig(n_filters=128, n_blocks=10, value_hidden=256),
}


def make_activation(name: str) -> nn.Module:
    """Activation factory for the residual tower (bead derby-sib).

    Returns a FRESH (parameter-free) activation module per call:
      "relu" -> nn.ReLU(inplace=True)  (today's exact tower nonlinearity)
      "mish" -> nn.Mish()              (smooth self-gated x*tanh(softplus(x)))
    Both are parameter-free, so swapping the activation never changes the
    state_dict keys or the parameter count. The "relu" path is byte-identical
    to the old F.relu(inplace) calls it replaces.
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "mish":
        return nn.Mish()
    raise ValueError(f"unknown activation {name!r}; options: relu, mish")


class ResBlock(nn.Module):
    def __init__(self, channels: int, activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Two tower nonlinearities, each a parameter-free activation module so
        # the relu default is byte-identical to the old F.relu calls.
        self.act1 = make_activation(activation)
        self.act2 = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act2(x + h)


class GlobalPoolResBlock(nn.Module):
    """ResBlock with a KataGo-style global-pooling bias.

    Structure (mirrors KataGo's "global pooling bias" in its residual blocks):
      x -> conv1 -> bn1 -> relu = h          (local features)
      g = [mean_HW(h) ; max_HW(h)]           (2*C global summary, per sample)
      b = pool_fc(g)                         (C per-channel biases)
      h = h + b[:, :, None, None]            (broadcast bias to every cell)
      h -> conv2 -> bn2
      out = relu(x + h)

    The bias is the same for every spatial location of a sample but is
    computed from the WHOLE board, so each cell sees a board-global signal
    (e.g. "a live-four exists somewhere") that a 3x3-conv stack of this depth
    cannot otherwise represent. Params added: pool_fc only = 2*C*C + C.

    Note: the first two ops (conv1, bn1) keep the same module names as
    ResBlock, so fuse_model_for_inference's conv1/bn1/conv2/bn2 fusion still
    applies. The pool_fc has no BatchNorm and is untouched by fusion.
    """

    def __init__(self, channels: int, activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        # mean + max pooled -> 2*C inputs; one bias per channel out.
        self.pool_fc = nn.Linear(2 * channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Two tower nonlinearities, parameter-free activation modules (relu
        # default byte-identical to the old F.relu calls).
        self.act1 = make_activation(activation)
        self.act2 = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.bn1(self.conv1(x)))
        mean = h.mean(dim=(2, 3))                 # (B, C)
        mx = h.amax(dim=(2, 3))                    # (B, C)
        bias = self.pool_fc(torch.cat([mean, mx], dim=1))  # (B, C)
        h = h + bias[:, :, None, None]             # broadcast over H, W
        h = self.bn2(self.conv2(h))
        return self.act2(x + h)


def _global_pool_block_flags(cfg: ModelConfig) -> list[bool]:
    """Return, per residual block, whether it uses global pooling.

    global_pool == False -> all False (byte-identical to current arch).
    global_pool == True  -> the latter half of blocks (n_blocks // 2 trailing).
    global_pool == int k -> the trailing k blocks (clamped to [0, n_blocks]).
    """
    n = cfg.n_blocks
    gp = cfg.global_pool
    if gp is False or gp == 0:
        return [False] * n
    if gp is True:
        n_pool = n // 2
    else:
        n_pool = max(0, min(int(gp), n))
    return [i >= n - n_pool for i in range(n)]


class GomokuNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.n_filters
        n_actions = cfg.board_size * cfg.board_size

        # 3x3 conv with padding=cfg.stem_padding. Output H/W after the stem is
        # board_size + 2*stem_padding - 2. The residual tower preserves that shape.
        kernel = 3
        spatial = cfg.board_size + 2 * cfg.stem_padding - kernel + 1

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.n_input_planes, c, kernel, padding=cfg.stem_padding, bias=False),
            nn.BatchNorm2d(c),
            make_activation(cfg.activation),
        )
        gp_flags = _global_pool_block_flags(cfg)
        self.tower = nn.Sequential(*[
            GlobalPoolResBlock(c, cfg.activation) if use_gp else ResBlock(c, cfg.activation)
            for use_gp in gp_flags
        ])

        # Policy head: 1x1 conv -> flatten -> linear to board_size^2.
        # Note the FC input uses the post-stem spatial size, not board_size.
        self.policy_conv = nn.Conv2d(c, cfg.policy_filters, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(cfg.policy_filters)
        self.policy_fc = nn.Linear(cfg.policy_filters * spatial * spatial, n_actions)

        # Value head: 1x1 conv -> flatten -> linear -> head-specific final FC.
        # The shared trunk (conv/bn/fc1) is identical for both representations;
        # only the final FC differs and exactly ONE of the two is constructed,
        # so the scalar default's state_dict / param count are byte-identical to
        # a model predating the value_head field.
        self.value_conv = nn.Conv2d(c, cfg.value_filters, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(cfg.value_filters)
        self.value_fc1 = nn.Linear(cfg.value_filters * spatial * spatial, cfg.value_hidden)
        if cfg.value_head == "wdl":
            # 3 logits over {win, draw, loss}; softmax in forward().
            self.value_wdl_fc = nn.Linear(cfg.value_hidden, 3)
        elif cfg.value_head == "hlgauss":
            # N logits over evenly-spaced bin centers in [-1, 1]; softmax in
            # forward() and the scalar value is the expectation (sum of
            # prob_i * bin_center_i). Register bin centers as a non-trainable
            # buffer so they ride state_dict / device moves alongside the model.
            if cfg.value_hlgauss_bins < 2:
                raise ValueError(
                    f"value_hlgauss_bins must be >= 2; got {cfg.value_hlgauss_bins}"
                )
            if cfg.value_hlgauss_sigma <= 0.0:
                raise ValueError(
                    f"value_hlgauss_sigma must be > 0; got {cfg.value_hlgauss_sigma}"
                )
            self.value_hlgauss_fc = nn.Linear(cfg.value_hidden, cfg.value_hlgauss_bins)
            centers = torch.linspace(-1.0, 1.0, cfg.value_hlgauss_bins)
            self.register_buffer("value_hlgauss_bin_centers", centers, persistent=False)
        else:
            # scalar tanh head — the exact pre-WDL module/key.
            self.value_fc2 = nn.Linear(cfg.value_hidden, 1)

        # V3 auxiliary opponent-reply policy head — a structural clone of the
        # policy head off the SAME shared tower output `h`. Constructed ONLY
        # when cfg.aux_opponent_reply is set, so the off-case state_dict and
        # param count are byte-identical to a head-less model. Predicts the
        # opponent's MCTS policy at the next ply (training-only target; dropped
        # at inference — see forward(return_aux=...) and fuse_model_for_inference).
        if cfg.aux_opponent_reply:
            self.aux_policy_conv = nn.Conv2d(c, cfg.policy_filters, 1, bias=False)
            self.aux_policy_bn = nn.BatchNorm2d(cfg.policy_filters)
            self.aux_policy_fc = nn.Linear(cfg.policy_filters * spatial * spatial, n_actions)

        # V4 auxiliary ownership head — a dense head off the SAME shared tower
        # output `h`. Predicts per-cell final control (81-way, one scalar per
        # board cell, no tanh — the loss is a masked MSE against a target in
        # [-1, 1]). Structural twin of the policy head (1x1 conv -> bn -> relu ->
        # flatten -> linear to N_ACTIONS). Constructed ONLY when
        # cfg.aux_ownership is set, so the off-case state_dict / param count /
        # inference graph are byte-identical to a model without it. TRAINING-
        # ONLY: never run by forward() unless return_ownership=True, never
        # touched by fuse_model_for_inference. Independent of the opponent-reply
        # head — either, both, or neither may be present.
        if cfg.aux_ownership:
            self.ownership_conv = nn.Conv2d(c, cfg.value_filters, 1, bias=False)
            self.ownership_bn = nn.BatchNorm2d(cfg.value_filters)
            self.ownership_fc = nn.Linear(cfg.value_filters * spatial * spatial, n_actions)

        # Swap2 choice head — a single Linear off the value head's penultimate
        # hidden activation (value_fc1's `value_hidden`-wide output), emitting
        # N_CHOICES (=3) logits over the choice slots. It taps the SAME shared
        # trunk as every other head and reuses the value-head's already-computed
        # `value_hidden` features, so it is tiny (value_hidden*3 + 3 params).
        # Constructed ONLY when cfg.choice_head is set (default True); when False
        # the state_dict / param count / inference graph are byte-identical to a
        # model predating this head. Reached only via forward_with_choice(), never
        # forward(), so the (policy, value) hot path is untouched.
        if cfg.choice_head:
            self.choice_fc = nn.Linear(cfg.value_hidden, N_CHOICES)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
        return_ownership: bool = False,
        return_value_logits: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass. Returns (policy, value) by default.

        `value` is ALWAYS the scalar in [-1, 1] that the rest of the codebase
        (MCTS leaf backup, anchor-ladder eval) consumes:
          * scalar head: tanh(value_fc2) — today's path, byte-identical.
          * wdl head: P(win) - P(loss) derived from softmax over the 3 logits.
          * hlgauss head: sum(softmax_i * bin_center_i) derived from softmax over
            the N logits (bin centers evenly spaced over [-1, 1]).
        So every scalar consumer is untouched regardless of representation.

        The auxiliary heads / value logits are independent and gated separately:
          * return_aux=True (requires cfg.aux_opponent_reply) appends the
            opponent-reply logits;
          * return_ownership=True (requires cfg.aux_ownership) appends the
            per-cell ownership logits;
          * return_value_logits=True (requires cfg.value_head in {"wdl","hlgauss"})
            appends the raw (B, 3) {win,draw,loss} OR (B, N) HL-Gauss bin logits
            for the value cross-entropy loss.
        The output tuple is ordered (policy, value, [aux_policy], [ownership],
        [value_logits]) with each optional tensor present iff its flag is set:
          none -> (p, v)                          [byte-identical default]
          aux only -> (p, v, aux_policy)          [unchanged 3-tuple contract]
        Only the trainer passes these flags; self-play and eval call forward(x)
        and never run either aux head or request WDL logits (zero extra FLOPs in
        the generation-bound path). Requesting a head/logits that were not
        constructed raises."""
        h = self.tower(self.stem(x))

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        value_logits = None
        if self.cfg.value_head == "wdl":
            value_logits = self.value_wdl_fc(v)            # (B, 3) = {win, draw, loss}
            wdl = F.softmax(value_logits, dim=-1)
            v = (wdl[:, 0] - wdl[:, 2])                    # derived scalar P(win)-P(loss)
        elif self.cfg.value_head == "hlgauss":
            value_logits = self.value_hlgauss_fc(v)        # (B, N)
            probs = F.softmax(value_logits, dim=-1)
            # Derived scalar = expectation under the categorical distribution:
            # v = sum(prob_i * bin_center_i). For a hard one-hot on a center
            # this exactly equals the center; for symmetric mass it equals 0.
            v = (probs * self.value_hlgauss_bin_centers).sum(dim=-1)
        else:
            v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        out: tuple[torch.Tensor, ...] = (p, v)
        if return_aux:
            if not self.cfg.aux_opponent_reply:
                raise RuntimeError(
                    "forward(return_aux=True) requires cfg.aux_opponent_reply=True "
                    "(aux head was not constructed)"
                )
            a = F.relu(self.aux_policy_bn(self.aux_policy_conv(h)))
            a = self.aux_policy_fc(a.flatten(1))
            out = out + (a,)
        if return_ownership:
            if not self.cfg.aux_ownership:
                raise RuntimeError(
                    "forward(return_ownership=True) requires cfg.aux_ownership=True "
                    "(ownership head was not constructed)"
                )
            o = F.relu(self.ownership_bn(self.ownership_conv(h)))
            o = self.ownership_fc(o.flatten(1))
            out = out + (o,)
        if return_value_logits:
            if self.cfg.value_head not in ("wdl", "hlgauss"):
                raise RuntimeError(
                    "forward(return_value_logits=True) requires cfg.value_head "
                    "in {'wdl','hlgauss'} (the value-logits FC was not constructed)"
                )
            out = out + (value_logits,)
        return out

    def forward_with_choice(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for swap2 CHOICE nodes: (policy, value, choice_logits).

        Returns the same (policy, value) as ``forward(x)`` PLUS the choice head's
        ``choice_logits`` of shape ``(B, N_CHOICES)`` — a prior over the swap2
        choice slots (responder STAY/SWAP/PLACE2 or opener pick-color), masked to
        the legal slots by the caller (see OpeningState.legal_choice_mask). This
        is the ONLY entry point that runs the choice head; ``forward`` and every
        ``policy, value = model(x)`` caller are byte-identical and never touch it.

        Choice nodes are ≤2 per game, so recomputing the trunk here (rather than
        threading a third return through the hot path) costs nothing meaningful.
        The choice head taps the value head's penultimate hidden activation, so
        the shared value-trunk compute is reused inline.
        """
        if not self.cfg.choice_head:
            raise RuntimeError(
                "forward_with_choice requires cfg.choice_head=True "
                "(the choice head was not constructed)"
            )
        h = self.tower(self.stem(x))

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = self.policy_fc(p.flatten(1))

        vh = F.relu(self.value_bn(self.value_conv(h)))
        vh = F.relu(self.value_fc1(vh.flatten(1)))   # penultimate value hidden
        choice_logits = self.choice_fc(vh)           # (B, N_CHOICES)

        if self.cfg.value_head == "wdl":
            wdl = F.softmax(self.value_wdl_fc(vh), dim=-1)
            v = wdl[:, 0] - wdl[:, 2]
        elif self.cfg.value_head == "hlgauss":
            probs = F.softmax(self.value_hlgauss_fc(vh), dim=-1)
            v = (probs * self.value_hlgauss_bin_centers).sum(dim=-1)
        else:
            v = torch.tanh(self.value_fc2(vh)).squeeze(-1)

        return p, v, choice_logits


def build_model(
    size: str = "small",
    *,
    stem_padding: int | None = None,
    aux_opponent_reply: bool = False,
    aux_ownership: bool = False,
    global_pool: bool | int | None = None,
    value_head: str | None = None,
    value_hlgauss_bins: int | None = None,
    value_hlgauss_sigma: float | None = None,
    activation: str | None = None,
) -> GomokuNet:
    if size not in SIZE_PRESETS:
        raise ValueError(f"unknown size {size!r}; options: {list(SIZE_PRESETS)}")
    cfg = SIZE_PRESETS[size]
    overrides: dict = {}
    if stem_padding is not None:
        overrides["stem_padding"] = stem_padding
    # Only override aux flags when explicitly enabled, so the default build path
    # is byte-identical to before these fields existed.
    if aux_opponent_reply:
        overrides["aux_opponent_reply"] = True
    if aux_ownership:
        overrides["aux_ownership"] = True
    if global_pool is not None:
        overrides["global_pool"] = global_pool
    # Only override value_head when an explicit non-default is requested, so the
    # scalar default build path stays byte-identical to a pre-WDL model.
    if value_head is not None and value_head != "scalar":
        if value_head not in ("wdl", "hlgauss"):
            raise ValueError(
                f"unknown value_head {value_head!r}; options: scalar, wdl, hlgauss"
            )
        overrides["value_head"] = value_head
    # HL-Gauss-specific knobs: only override when the user supplied a value AND
    # they differ from the ModelConfig defaults, so a scalar/wdl model never gets
    # a non-default bins/sigma in its config (keeps the off-path config dict
    # byte-identical for scalar/wdl checkpoints).
    if value_hlgauss_bins is not None and value_hlgauss_bins != ModelConfig.value_hlgauss_bins:
        overrides["value_hlgauss_bins"] = int(value_hlgauss_bins)
    if value_hlgauss_sigma is not None and value_hlgauss_sigma != ModelConfig.value_hlgauss_sigma:
        overrides["value_hlgauss_sigma"] = float(value_hlgauss_sigma)
    # Only override activation when an explicit non-default is requested, so the
    # relu default build path stays byte-identical to a pre-activation-field model.
    if activation is not None and activation != "relu":
        if activation != "mish":
            raise ValueError(f"unknown activation {activation!r}; options: relu, mish")
        overrides["activation"] = activation
    if overrides:
        from dataclasses import replace
        cfg = replace(cfg, **overrides)
    return GomokuNet(cfg)


def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.Module) -> tuple[nn.Conv2d, nn.Module]:
    if isinstance(bn, nn.BatchNorm2d):
        return fuse_conv_bn_eval(conv, bn), nn.Identity()
    return conv, bn


def fuse_model_for_inference(model: GomokuNet) -> GomokuNet:
    """Fuse Conv+BatchNorm pairs for eval-only inference.

    This mutates and returns `model`. Call only after loading checkpoint weights
    into a model that will not be trained or saved back as a normal checkpoint.
    """
    model.eval()
    model.stem[0], model.stem[1] = _fuse_conv_bn_pair(model.stem[0], model.stem[1])
    for block in model.tower:
        block.conv1, block.bn1 = _fuse_conv_bn_pair(block.conv1, block.bn1)
        block.conv2, block.bn2 = _fuse_conv_bn_pair(block.conv2, block.bn2)
    model.policy_conv, model.policy_bn = _fuse_conv_bn_pair(
        model.policy_conv, model.policy_bn
    )
    model.value_conv, model.value_bn = _fuse_conv_bn_pair(
        model.value_conv, model.value_bn
    )
    return model


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    path: str,
    model: GomokuNet,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    epoch: int = 0,
    total_games: int = 0,
    wandb_run_id: str | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.cfg),
        "epoch": epoch,
        "total_games": total_games,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if wandb_run_id is not None:
        payload["wandb_run_id"] = wandb_run_id
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    *,
    expect_board_size: int | None = BOARD_SIZE,
) -> tuple[GomokuNet, dict]:
    """Load a checkpoint into a freshly built model.

    Board-size contract: checkpoints embed ``board_size`` in their
    model_config (older checkpoints predate the field and were all 9x9, so
    they load as board_size=9). By default the checkpoint's board size must
    match the active process board size (``gomoku.board_config.BOARD_SIZE``);
    a mismatch raises ValueError instead of producing a model whose heads
    disagree with every other shape in the process. Pass
    ``expect_board_size=None`` to skip the check (offline inspection only).
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_cfg = dict(payload["model_config"])
    # Pre-AZ-recipe checkpoints predate stem_padding; default to 1 so they load.
    saved_cfg.setdefault("stem_padding", 1)
    # Pre-HL-Gauss checkpoints predate value_hlgauss_{bins,sigma}; fall back to
    # the ModelConfig defaults so a scalar/wdl checkpoint loads unchanged.
    saved_cfg.setdefault("value_hlgauss_bins", ModelConfig.value_hlgauss_bins)
    saved_cfg.setdefault("value_hlgauss_sigma", ModelConfig.value_hlgauss_sigma)
    # Pre-15x15-era checkpoints predate board_size; they were all 9x9.
    saved_cfg.setdefault("board_size", 9)
    # Pre-swap2 checkpoints predate choice_head. The champion's state_dict has no
    # choice-head weights, but the swap2 net WANTS the head, so we build it (the
    # config default is True) and warm-start the core strict + fall back the
    # choice head to its fresh init below — rather than disabling the head.
    saved_cfg.setdefault("choice_head", ModelConfig.choice_head)
    if expect_board_size is not None and saved_cfg["board_size"] != expect_board_size:
        raise ValueError(
            f"checkpoint {path!r} was trained at board size "
            f"{saved_cfg['board_size']}, but this process is configured for "
            f"board size {expect_board_size} (set GOMOKU_BOARD_SIZE="
            f"{saved_cfg['board_size']} or pass --board-size "
            f"{saved_cfg['board_size']} to use this checkpoint)"
        )
    cfg = ModelConfig(**saved_cfg)
    model = GomokuNet(cfg).to(device)

    # Warm-start tolerance for the swap2 choice head ONLY. The champion predates
    # this head, so its state_dict lacks `choice_*` keys. We still want a STRICT
    # load for the core (a genuinely-missing core weight must error, not be
    # silently skipped), so we splice the freshly-initialized choice-head params
    # into the saved dict for any choice key the checkpoint is missing, then load
    # strict. A checkpoint that already carries the choice head loads it exactly
    # (no fallback triggered) and round-trips bit-for-bit.
    state = dict(payload["model_state_dict"])
    if cfg.choice_head:
        model_sd = model.state_dict()
        for key, tensor in model_sd.items():
            if key.startswith("choice_") and key not in state:
                state[key] = tensor
    model.load_state_dict(state)
    return model, payload
