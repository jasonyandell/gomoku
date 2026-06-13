#!/usr/bin/env python3
"""Function-preserving Net2Net grow: widen + deepen a trained GomokuNet.

Grows a strong SMALL net into a BIGGER net (more channels and/or more residual
blocks) whose forward function is (almost) IDENTICAL to the source, so the
grown net STARTS at the source's strength and can then train into its extra
capacity — no cold-start retraining, no fast-attack collapse.

Method (Chen, Goodfellow, Shlens 2016, "Net2Net"):

  WIDEN (Net2WiderNet).  To grow a layer's output from m units to q (>= m)
  function-preservingly, pick a mapping g: {0..q-1} -> {0..m-1} with g(j)=j for
  j<m (originals kept) and g(j) a *replicated* original for j>=m. Copy output
  unit g(j) into target unit j. Then EVERY consumer of those units divides its
  incoming weight for target-input j by the replication count
  count(g(j)) = |{k : g(k)=g(j)}|, so the duplicated units' contributions sum
  back to the original. BatchNorm is per-channel and is replicated by g (gamma,
  beta, running_mean, running_var all carried channel-wise); since BN is an
  affine reparam of the channel it composes exactly with the conv replication.

  DEEPEN (Net2DeeperNet).  Insert extra residual blocks initialised to compute
  IDENTITY. For our post-activation ResBlock
      out = act2( x + bn2(conv2(act1(bn1(conv1(x))))) )
  zeroing bn2.weight (gamma) AND bn2.bias (beta) forces the residual branch to 0,
  so out = act2(x). With ReLU and x already >= 0 (every tower input is the output
  of a preceding ReLU — the stem and every block end in act2=ReLU), act2(x)=x:
  exact identity. (For Mish this trick is NOT exact; we refuse non-relu deepen.)
  GlobalPoolResBlock is deepened the same way (zero bn2) — the pool bias feeds
  conv2->bn2, which is zeroed, so the whole residual branch is still 0.

What is preserved here, concretely (small 64x4 -> 96x8, both global_pool):
  * Tower width c: 64 -> 96 via ONE channel map g_c. Flows through the stem
    output, every block (the residual add uses the SAME g_c on identity + conv
    paths, so it stays consistent), and into the policy_conv / value_conv INPUT
    channels (which is where g_c is "consumed" and divided out).
  * value_hidden: 64 -> 128 via a SECOND map g_h on value_fc1's OUTPUT, divided
    out at value_fc2 (scalar) / value_wdl_fc / value_hlgauss_fc input.
  * policy_filters / value_filters are UNCHANGED across the presets, and
    board_size is preserved, so policy_fc and value_fc1's INPUT (flattened
    spatial) are shape-identical and copy verbatim — no FC remap needed.
  * extra blocks (4 -> 8) inserted as zero-residual identity blocks.

CPU-only by construction (no training, only a forward sanity check). Run under
GOMOKU_DEVICE=cpu to stay off a busy GPU.

Usage:
    GOMOKU_DEVICE=cpu python scripts/net2net_grow.py \
        --from checkpoints15/latest.pt \
        --out  /tmp/g15_96x8_net2net_seed.pt \
        [--target-size 96x8]

The output is in the trainer's --resume format (embedded ModelConfig for the
TARGET size, epoch/total_games=0, no optimizer/buffer/wandb_run_id), exactly
like scripts/warmstart_15x15.py, so it launches warm via
    GOMOKU_BOARD_SIZE=15 python -m gomoku.train --resume <out> --size 96x8 ...
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_widen_map(m: int, q: int, seed: int) -> tuple[list[int], list[int]]:
    """Net2WiderNet channel map g and per-source replication counts.

    Returns (g, counts) where g has length q with g[j] in [0, m): g[j]=j for
    j<m (originals preserved in place) and g[j] a random original for j>=m.
    counts has length m: counts[s] = |{j : g[j]=s}| (>=1 for every s, so every
    consumer divisor is well defined). q >= m is required.
    """
    if q < m:
        raise ValueError(f"target width {q} < source width {m} (widen only)")
    import random  # noqa: PLC0415

    rng = random.Random(seed)
    g = list(range(m))  # originals map to themselves
    # Extra target units replicate randomly-chosen original units.
    g += [rng.randrange(m) for _ in range(q - m)]
    counts = [0] * m
    for s in g:
        counts[s] += 1
    return g, counts


def grow(
    src_path: str,
    out_path: str,
    *,
    target_size: str,
    seed: int = 0,
) -> dict:
    """Net2Net-grow the source checkpoint into `target_size`, save the seed.

    Returns a summary dict (param counts + the measured max output deviation).
    """
    # 1) Raw payload load — DO NOT go through load_checkpoint (it board-size
    #    guards). Mirror warmstart_15x15.py: read state_dict + config, backfill
    #    the same defaults, and preserve the source board_size in the target.
    import torch  # noqa: PLC0415

    payload = torch.load(src_path, map_location="cpu", weights_only=False)
    src_state = payload["model_state_dict"]
    src_cfg = dict(payload["model_config"])
    src_cfg.setdefault("stem_padding", 1)
    src_board = int(src_cfg.get("board_size", 9))

    # Lock the process board size to the SOURCE board (so the rebuilt source net
    # and the freshly built target net both size their heads for that board)
    # BEFORE importing anything that derives module-level constants.
    os.environ["GOMOKU_BOARD_SIZE"] = str(src_board)
    import gomoku.board_config as bc  # noqa: PLC0415

    if bc.BOARD_SIZE != src_board:
        raise SystemExit(
            f"GOMOKU_BOARD_SIZE did not resolve to the source board {src_board} "
            f"(got {bc.BOARD_SIZE}); set GOMOKU_BOARD_SIZE={src_board} before "
            f"launching (some earlier import pinned a different size)."
        )

    from gomoku.model import (  # noqa: PLC0415
        SIZE_PRESETS,
        GomokuNet,
        ModelConfig,
        _global_pool_block_flags,
        n_params,
        save_checkpoint,
    )

    src_cfg.setdefault("value_hlgauss_bins", ModelConfig.value_hlgauss_bins)
    src_cfg.setdefault("value_hlgauss_sigma", ModelConfig.value_hlgauss_sigma)

    if target_size not in SIZE_PRESETS:
        raise SystemExit(
            f"unknown --target-size {target_size!r}; options: {list(SIZE_PRESETS)}"
        )

    # 2) Rebuild the SOURCE net exactly (so we can forward it for the equivalence
    #    check) and BUILD the TARGET net: same board_size + same head/stem/
    #    global_pool/value-head/activation config, only n_filters / n_blocks /
    #    value_hidden grow per the preset.
    from dataclasses import replace  # noqa: PLC0415

    src_model_cfg = ModelConfig(**src_cfg)
    src_net = GomokuNet(src_model_cfg)
    src_net.load_state_dict(src_state)
    src_net.eval()

    preset = SIZE_PRESETS[target_size]
    # Carry every architecture-defining field from the source EXCEPT the three
    # the preset grows. board_size, stem_padding, policy/value_filters, the
    # value-head representation, global_pool, activation, aux flags all come
    # from the source so the grown net is structurally the same net, wider+deeper.
    tgt_cfg = replace(
        src_model_cfg,
        n_filters=preset.n_filters,
        n_blocks=preset.n_blocks,
        value_hidden=preset.value_hidden,
    )
    tgt_net = GomokuNet(tgt_cfg)
    tgt_net.eval()

    activation = src_cfg.get("activation", "relu")
    if activation != "relu" and tgt_cfg.n_blocks > src_model_cfg.n_blocks:
        raise SystemExit(
            f"deepen requires relu activation for exact identity blocks; source "
            f"activation is {activation!r}. Widen-only is safe; refusing to "
            f"insert non-identity blocks under {activation!r}."
        )

    c_src = src_model_cfg.n_filters
    c_tgt = tgt_cfg.n_filters
    h_src = src_model_cfg.value_hidden
    h_tgt = tgt_cfg.value_hidden

    # Channel maps. g_c: tower width (stem out, every block, conv-head inputs).
    # g_h: value-hidden width (value_fc1 out -> value_fc2/wdl/hlgauss in).
    g_c, counts_c = _build_widen_map(c_src, c_tgt, seed)
    g_h, counts_h = _build_widen_map(h_src, h_tgt, seed + 1)

    tdiv_c = torch.tensor([counts_c[g_c[j]] for j in range(c_tgt)], dtype=torch.float32)
    tdiv_h = torch.tensor([counts_h[g_h[j]] for j in range(h_tgt)], dtype=torch.float32)
    idx_c = torch.tensor(g_c, dtype=torch.long)
    idx_h = torch.tensor(g_h, dtype=torch.long)

    # ---- channel-mapping primitives ------------------------------------------
    def widen_out_conv(w: torch.Tensor) -> torch.Tensor:
        """Replicate a conv's OUT channels (dim 0) by g_c: (O,I,k,k)->(c_tgt,I,k,k)."""
        return w[idx_c].clone()

    def widen_in_conv(w: torch.Tensor) -> torch.Tensor:
        """Replicate+divide a conv's IN channels (dim 1) by g_c, preserving the
        composed function: (O,c_src,k,k)->(O,c_tgt,k,k), each replicated input
        column divided by its replication count."""
        w2 = w[:, idx_c].clone()
        w2 = w2 / tdiv_c.view(1, c_tgt, 1, 1)
        return w2

    def widen_bn(prefix_src: str, prefix_tgt: str, sd: dict) -> None:
        """Replicate a BatchNorm's per-channel tensors by g_c (no division: BN is
        a per-channel affine reparam, so the duplicated channels are identical
        and the DOWNSTREAM conv's in-channel division handles the sum)."""
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            sd[f"{prefix_tgt}.{suffix}"] = src_state[f"{prefix_src}.{suffix}"][idx_c].clone()
        # num_batches_tracked is a scalar counter, copy verbatim.
        k = f"{prefix_src}.num_batches_tracked"
        if k in src_state:
            sd[f"{prefix_tgt}.num_batches_tracked"] = src_state[k].clone()

    new_sd: dict = {}

    # ---- stem: conv out-channels widen, BN widen -----------------------------
    # stem.0 is the conv (in = n_input_planes, fixed; out = c). stem.1 is BN(c).
    new_sd["stem.0.weight"] = widen_out_conv(src_state["stem.0.weight"])
    widen_bn("stem.1", "stem.1", new_sd)

    # ---- residual tower ------------------------------------------------------
    # Strategy (see the gp-type-aware scheduler below): each source block is
    # WIDENED into a target block of the SAME gp type, in order; the extra target
    # slots become zero-residual IDENTITY blocks (conv1/bn1/pool_fc kept at fresh
    # init, bn2.{weight,bias} zeroed -> residual branch is 0 -> block = act2(x) =
    # x for relu with x>=0, which holds everywhere in the post-ReLU tower). This
    # both preserves the function and keeps the plain/global-pool block counts of
    # the target preset.
    src_gp_flags = _global_pool_block_flags(src_model_cfg)
    tgt_gp_flags = _global_pool_block_flags(tgt_cfg)
    n_src_blocks = src_model_cfg.n_blocks
    n_tgt_blocks = tgt_cfg.n_blocks

    def widen_block(si: int, ti: int, sd: dict) -> None:
        """Widen source block `si` into target block `ti` (both width c).

        Residual block:  conv1 (c->c) then bn1, optionally pool_fc, conv2 (c->c)
        then bn2. Both convs sit BETWEEN the same channel map g_c on input AND
        output, so each gets in-channel division AND out-channel replication.
        That keeps the residual add x + h consistent: the identity path uses g_c
        (carried from the previous block's output) and the conv path reproduces
        the same widened channels, each scaled so the post-bn2 sum matches the
        source per the standard Net2Wider divide-the-consumer rule.
        """
        sp, tp = f"tower.{si}", f"tower.{ti}"
        # conv1: in g_c (divide), out g_c (replicate)
        sd[f"{tp}.conv1.weight"] = widen_out_conv(widen_in_conv(src_state[f"{sp}.conv1.weight"]))
        widen_bn(f"{sp}.bn1", f"{tp}.bn1", sd)
        # conv2: in g_c (divide), out g_c (replicate)
        sd[f"{tp}.conv2.weight"] = widen_out_conv(widen_in_conv(src_state[f"{sp}.conv2.weight"]))
        widen_bn(f"{sp}.bn2", f"{tp}.bn2", sd)
        # GlobalPool block: pool_fc is Linear(2c -> c).
        #   input  = cat([mean(h), max(h)]) over the c (widened) channels of h;
        #   output = a per-channel bias added to h (which has c channels).
        # h's channels are g_c-widened (each replicated channel is identical to
        # its source channel). For the OUTPUT bias to match per channel, replicate
        # pool_fc OUT rows by g_c. For the INPUT (2c): the two halves [mean|max]
        # are each indexed by g_c; replicated input columns must be DIVIDED by the
        # replication count so the summed contribution matches the source.
        if f"{sp}.pool_fc.weight" in src_state:
            assert src_gp_flags[si] and tgt_gp_flags[ti], (
                f"global-pool flag mismatch mapping src block {si} -> tgt block {ti}"
            )
            w = src_state[f"{sp}.pool_fc.weight"]      # (c_src, 2*c_src)
            b = src_state[f"{sp}.pool_fc.bias"]        # (c_src,)
            mean_half = w[:, :c_src]                   # (c_src, c_src)
            max_half = w[:, c_src:]                    # (c_src, c_src)
            # widen INPUT cols of each half by g_c with divide
            mean_w = (mean_half[:, idx_c] / tdiv_c.view(1, c_tgt)).contiguous()
            max_w = (max_half[:, idx_c] / tdiv_c.view(1, c_tgt)).contiguous()
            w2 = torch.cat([mean_w, max_w], dim=1)     # (c_src, 2*c_tgt)
            # widen OUTPUT rows by g_c (replicate; the bias is per-channel, added
            # to the matching widened channel of h, so NO divide on the rows)
            w2 = w2[idx_c].clone()                      # (c_tgt, 2*c_tgt)
            b2 = b[idx_c].clone()                       # (c_tgt,)
            sd[f"{tp}.pool_fc.weight"] = w2
            sd[f"{tp}.pool_fc.bias"] = b2

    # gp-type-aware block scheduling. The global-pool blocks are the TRAILING
    # blocks of each tower (gp split = latter half), so both flag lists look like
    # [plain..plain, gp..gp]. We map each source block to a target block OF THE
    # SAME gp TYPE (widen), and fill the remaining target slots with IDENTITY
    # blocks (which preserve the function regardless of where they sit, since
    # zero-residual identity holds for both ResBlock and GlobalPoolResBlock under
    # relu). Walking target blocks left-to-right with a per-type source pointer
    # keeps the source blocks in order and places the inserted identity blocks
    # within the matching type group (so a deepened global_pool net keeps the
    # right plain/gp counts AND stays function-preserving).
    src_plain = [i for i in range(n_src_blocks) if not src_gp_flags[i]]
    src_gp = [i for i in range(n_src_blocks) if src_gp_flags[i]]
    tgt_plain = sum(1 for i in range(n_tgt_blocks) if not tgt_gp_flags[i])
    tgt_gp = sum(1 for i in range(n_tgt_blocks) if tgt_gp_flags[i])
    if len(src_plain) > tgt_plain or len(src_gp) > tgt_gp:
        raise SystemExit(
            f"target has fewer blocks of a type than the source: source "
            f"plain={len(src_plain)} gp={len(src_gp)}, target plain={tgt_plain} "
            f"gp={tgt_gp}. net2net widen+deepen requires the target to have at "
            f"least as many blocks of each (plain / global-pool) type."
        )

    tgt_fresh = tgt_net.state_dict()
    p_ptr = g_ptr = 0
    inserted = 0
    block_plan: list[tuple[int, int | None]] = []  # (tgt_idx, src_idx or None)
    for ti in range(n_tgt_blocks):
        if tgt_gp_flags[ti]:
            si = src_gp[g_ptr] if g_ptr < len(src_gp) else None
            if si is not None:
                g_ptr += 1
        else:
            si = src_plain[p_ptr] if p_ptr < len(src_plain) else None
            if si is not None:
                p_ptr += 1
        block_plan.append((ti, si))
        if si is not None:
            widen_block(si, ti, new_sd)
        else:
            # Inserted IDENTITY block: copy the target's FRESH-init weights for
            # the whole block, then zero bn2 (gamma + beta) so the residual
            # branch is exactly 0 -> block computes act2(x) = x (relu, x>=0).
            tp = f"tower.{ti}"
            for k in tgt_fresh:
                if k.startswith(tp + "."):
                    new_sd[k] = tgt_fresh[k].clone()
            new_sd[f"{tp}.bn2.weight"] = torch.zeros_like(new_sd[f"{tp}.bn2.weight"])
            new_sd[f"{tp}.bn2.bias"] = torch.zeros_like(new_sd[f"{tp}.bn2.bias"])
            inserted += 1

    # ---- policy head ---------------------------------------------------------
    # policy_conv: 1x1 conv (c -> policy_filters). IN channels widen by g_c with
    # divide; OUT channels (policy_filters) are UNCHANGED.
    new_sd["policy_conv.weight"] = widen_in_conv(src_state["policy_conv.weight"])
    # policy_bn is over policy_filters (unchanged) -> copy verbatim.
    for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
        k = f"policy_bn.{suffix}"
        if k in src_state:
            new_sd[k] = src_state[k].clone()
    # policy_fc: Linear(policy_filters*spatial^2 -> n_actions). Neither dim moved
    # (policy_filters fixed, board_size preserved) -> copy verbatim.
    new_sd["policy_fc.weight"] = src_state["policy_fc.weight"].clone()
    new_sd["policy_fc.bias"] = src_state["policy_fc.bias"].clone()

    # ---- value head ----------------------------------------------------------
    # value_conv: 1x1 conv (c -> value_filters). IN widen by g_c with divide; OUT
    # (value_filters) unchanged.
    new_sd["value_conv.weight"] = widen_in_conv(src_state["value_conv.weight"])
    for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
        k = f"value_bn.{suffix}"
        if k in src_state:
            new_sd[k] = src_state[k].clone()
    # value_fc1: Linear(value_filters*spatial^2 -> value_hidden). INPUT dim fixed
    # (value_filters + board_size preserved) -> copy input columns verbatim;
    # OUTPUT widens value_hidden by g_h (replicate rows; the consumer FC divides).
    vfc1_w = src_state["value_fc1.weight"]   # (h_src, value_filters*spatial^2)
    vfc1_b = src_state["value_fc1.bias"]     # (h_src,)
    new_sd["value_fc1.weight"] = vfc1_w[idx_h].clone()
    new_sd["value_fc1.bias"] = vfc1_b[idx_h].clone()
    # final value FC consumes value_hidden -> divide its input columns by g_h's
    # replication counts (Net2Wider consumer rule). Exactly one of these exists.
    for fc_name in ("value_fc2", "value_wdl_fc", "value_hlgauss_fc"):
        wk = f"{fc_name}.weight"
        if wk in src_state:
            w = src_state[wk]                # (out, h_src)
            new_sd[wk] = (w[:, idx_h] / tdiv_h.view(1, h_tgt)).contiguous()
            new_sd[f"{fc_name}.bias"] = src_state[f"{fc_name}.bias"].clone()

    # ---- aux heads (training-only; widen identically if present) -------------
    if src_model_cfg.aux_opponent_reply:
        new_sd["aux_policy_conv.weight"] = widen_in_conv(src_state["aux_policy_conv.weight"])
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            k = f"aux_policy_bn.{suffix}"
            if k in src_state:
                new_sd[k] = src_state[k].clone()
        new_sd["aux_policy_fc.weight"] = src_state["aux_policy_fc.weight"].clone()
        new_sd["aux_policy_fc.bias"] = src_state["aux_policy_fc.bias"].clone()
    if src_model_cfg.aux_ownership:
        new_sd["ownership_conv.weight"] = widen_in_conv(src_state["ownership_conv.weight"])
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            k = f"ownership_bn.{suffix}"
            if k in src_state:
                new_sd[k] = src_state[k].clone()
        new_sd["ownership_fc.weight"] = src_state["ownership_fc.weight"].clone()
        new_sd["ownership_fc.bias"] = src_state["ownership_fc.bias"].clone()

    # 3) Load into the target net. We then ASSERT every target param was set
    #    (the inserted-block conv1/bn1/pool_fc came from tgt_fresh, so all keys
    #    are present), catching any unmapped tensor loudly. strict=False just lets
    #    us collect the missing/unexpected lists ourselves for a clear error.
    missing, unexpected = tgt_net.load_state_dict(new_sd, strict=False)
    # We populated every target key (mapped tensors + inserted-block keys from
    # tgt_fresh), so `missing` should be empty. Any leftover is a real
    # unsupported-divergence bug -> fail loudly rather than write a partial seed.
    if missing:
        leftover = sorted(missing)
        raise SystemExit(
            f"net2net: target keys not populated by the grow map: {leftover}. "
            f"This is an unsupported architecture divergence; refusing to write "
            f"a partial seed.\n  source cfg: {src_cfg}\n  target cfg: {vars(tgt_cfg)}"
        )
    if unexpected:
        raise SystemExit(f"net2net: unexpected keys produced: {sorted(unexpected)}")

    # 4) EQUIVALENCE CHECK (the proof). Forward both nets in eval() on a batch of
    #    random board-shaped inputs; report the max abs deviation of policy + value.
    tgt_net.eval()
    torch.manual_seed(1234)
    n_planes = src_model_cfg.n_input_planes
    x = torch.randn(8, n_planes, src_board, src_board)
    with torch.no_grad():
        p_src, v_src = src_net(x)
        p_tgt, v_tgt = tgt_net(x)
    max_dev_policy = float((p_src - p_tgt).abs().max())
    max_dev_value = float((v_src - v_tgt).abs().max())
    max_dev = max(max_dev_policy, max_dev_value)

    # 5) Save in the trainer's --resume format (fresh timeline; no optimizer /
    #    buffer / wandb_run_id / ema). board_size preserved via target.cfg.
    save_checkpoint(
        out_path,
        tgt_net,
        optimizer=None,
        epoch=0,
        total_games=0,
        wandb_run_id=None,
        extra={
            "net2net_from": os.path.abspath(src_path),
            "net2net_source_size": f"{c_src}x{n_src_blocks}",
            "net2net_target_size": target_size,
            "net2net_board_size": src_board,
            "net2net_max_output_deviation": max_dev,
        },
    )

    return {
        "src_path": src_path,
        "out_path": out_path,
        "board_size": src_board,
        "target_size": target_size,
        "src_cfg": src_cfg,
        "tgt_cfg": dict(vars(tgt_cfg)),
        "src_params": n_params(src_net),
        "tgt_params": n_params(tgt_net),
        "n_src_blocks": n_src_blocks,
        "n_tgt_blocks": n_tgt_blocks,
        "c_src": c_src,
        "c_tgt": c_tgt,
        "h_src": h_src,
        "h_tgt": h_tgt,
        "n_inserted_blocks": inserted,
        "block_plan": block_plan,
        "max_dev_policy": max_dev_policy,
        "max_dev_value": max_dev_value,
        "max_dev": max_dev,
    }


def _print_summary(s: dict) -> None:
    print("=" * 70)
    print("NET2NET GROW SUMMARY")
    print("=" * 70)
    b = s["board_size"]
    print(f"  source : {s['src_path']}  (board {b}x{b})")
    print(f"  target : {s['out_path']}  (size={s['target_size']}, board {b}x{b})")
    print(f"  widen  : {s['c_src']}ch -> {s['c_tgt']}ch  (tower)")
    print(f"           {s['h_src']} -> {s['h_tgt']}  (value_hidden)")
    print(f"  deepen : {s['n_src_blocks']} -> {s['n_tgt_blocks']} blocks "
          f"(+{s['n_inserted_blocks']} identity blocks, placed by gp-type)")
    print(f"  params : {s['src_params']:,} -> {s['tgt_params']:,}  "
          f"(x{s['tgt_params']/max(1,s['src_params']):.2f})")
    print("-" * 70)
    print(f"  EQUIVALENCE (max |source - grown| over a random batch):")
    print(f"    policy logits : {s['max_dev_policy']:.3e}")
    print(f"    value         : {s['max_dev_value']:.3e}")
    print(f"    overall max   : {s['max_dev']:.3e}")
    print("=" * 70)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--from", dest="src", required=True,
                   help="source checkpoint to grow FROM (e.g. the live "
                        "15x15 latest.pt)")
    p.add_argument("--out", dest="out", required=True,
                   help="output grown seed checkpoint (trainer --resume format)")
    p.add_argument("--target-size", default="96x8",
                   help="target architecture preset to grow into (default: 96x8)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the channel-replication maps (default: 0)")
    args = p.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f"source checkpoint not found: {args.src}")

    summary = grow(args.src, args.out, target_size=args.target_size, seed=args.seed)
    _print_summary(summary)
    sys.exit(0)


if __name__ == "__main__":
    main()
