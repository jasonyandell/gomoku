#!/usr/bin/env python3
"""Cross-board warm-start: seed a 15x15 net from a 9x9 champion's conv tower.

A 15x15 GomokuNet shares the entire board-size-INDEPENDENT convolutional tower
with a 9x9 net of the same architecture (stem, residual blocks incl. global-pool
blocks, policy_conv/policy_bn, value_conv/value_bn, value_fc1.bias, value_fc2).
Only three tensors are board-bound and differ in shape:

  * policy_fc.weight   (n_actions, policy_filters * spatial^2)
  * policy_fc.bias     (n_actions,)
  * value_fc1.weight   (value_hidden, value_filters * spatial^2)

where n_actions = board_size^2 and spatial = board_size + 2*stem_padding - 2.
This script builds a FRESH 15x15 net of the matching architecture and copies
every source tensor whose name AND shape match, leaving those three FCs at their
fresh init. The result is saved in the trainer's checkpoint format so it can be
launched cold-but-warm via `GOMOKU_BOARD_SIZE=15 python -m gomoku.train --resume`.

The transfer is driven purely by name+shape agreement (a partial, non-strict
load), NOT a hard-coded skip list, so an unexpected arch divergence shows up as
an extra skip. We then ASSERT that the entire conv tower (every conv / BN /
global-pool tensor) transferred, failing loudly if it did not.

CPU-only by construction (it never trains and only does a forward sanity check);
run it under GOMOKU_DEVICE=cpu to stay off a busy GPU.

Usage:
    python scripts/warmstart_15x15.py \
        --from sweep_runs/derby_v8/_peaks/mate-discount/peak.pt \
        --out  /tmp/g15_warmstart_seed.pt \
        [--size small] [--board-size 15]
"""

from __future__ import annotations

import argparse
import os
import sys


# A state_dict key is "board-bound" iff its shape depends on the board size,
# i.e. on n_actions (= board^2) or on the post-stem spatial flattening. For the
# scalar/relu small+global_pool arch these are exactly the policy FC and the
# value_fc1 weight. We do NOT use this set to decide what to copy (shape-matching
# does that); it is used only to recognise an EXPECTED skip vs a surprise.
_EXPECTED_BOARD_BOUND_KEYS = frozenset({
    "policy_fc.weight",
    "policy_fc.bias",
    "value_fc1.weight",
})


def _resolve_board_size(arg_board_size: int) -> int:
    """Make sure the process board size matches the requested target.

    gomoku.board_config resolves BOARD_SIZE at import time from
    GOMOKU_BOARD_SIZE, so the target net must be built in a process whose
    GOMOKU_BOARD_SIZE equals --board-size. We set the env var BEFORE importing
    anything from gomoku, then assert it stuck.
    """
    os.environ["GOMOKU_BOARD_SIZE"] = str(arg_board_size)
    import gomoku.board_config as bc  # noqa: PLC0415 — must follow the env set

    if bc.BOARD_SIZE != arg_board_size:
        raise SystemExit(
            f"GOMOKU_BOARD_SIZE did not resolve to {arg_board_size} "
            f"(got {bc.BOARD_SIZE}); set GOMOKU_BOARD_SIZE={arg_board_size} in the "
            f"environment before launching, or some import pinned it earlier."
        )
    return bc.BOARD_SIZE


# The value head's final FC is the only head whose KEY (not just shape) depends
# on the value representation: "scalar" -> value_fc2, "wdl" -> value_wdl_fc,
# "hlgauss" -> value_hlgauss_fc. When --target-value-head changes the
# representation vs the source, the source's value-head FC has no name match in
# the target and is an INTENDED fresh-head skip (the conv tower + value trunk
# still transfer). We recognise these as expected skips so the arch-mismatch
# assertion does not fire on a deliberate head swap.
_VALUE_HEAD_FC_PREFIXES = frozenset({"value_fc2", "value_wdl_fc", "value_hlgauss_fc"})


def _is_value_head_fc_key(name: str) -> bool:
    return any(name.startswith(pfx) for pfx in _VALUE_HEAD_FC_PREFIXES)


def warmstart(
    src_path: str,
    out_path: str,
    *,
    size: str,
    board_size: int,
    target_value_head: str | None = None,
) -> dict:
    """Build a fresh `board_size` net, partial-load the source conv tower, save.

    ``target_value_head`` (None = inherit the source's value_head) lets the
    target use a DIFFERENT value representation than the source — e.g. warm-start
    a WDL net's conv tower from a scalar 9x9 champion. The value head's final FC
    is then a FRESH init (its key differs: value_fc2 vs value_wdl_fc), exactly
    the intended behavior for the G15-wdl cell ("warm tower + fresh wdl head").

    Returns a summary dict (counts + percentages + key lists) for the caller to
    print / assert on.
    """
    target_board = _resolve_board_size(board_size)

    # Imports AFTER the board-size env is locked, so GomokuNet sizes its heads
    # for the target board.
    import torch  # noqa: PLC0415

    from gomoku.model import (  # noqa: PLC0415
        GomokuNet,
        ModelConfig,
        build_model,
        save_checkpoint,
    )

    # 1) Load the source checkpoint's state_dict + config WITHOUT going through
    #    load_checkpoint (which hard-errors on a board-size mismatch). We only
    #    need the raw payload here.
    payload = torch.load(src_path, map_location="cpu", weights_only=False)
    src_state = payload["model_state_dict"]
    src_cfg = dict(payload["model_config"])
    # Backfill the same defaults load_checkpoint uses, so a pre-field 9x9
    # checkpoint still describes a buildable config.
    src_cfg.setdefault("stem_padding", 1)
    src_cfg.setdefault("value_hlgauss_bins", ModelConfig.value_hlgauss_bins)
    src_cfg.setdefault("value_hlgauss_sigma", ModelConfig.value_hlgauss_sigma)
    src_cfg.setdefault("board_size", 9)
    src_board = int(src_cfg["board_size"])

    # 2) Build a FRESH target net at the target board size + matching arch.
    #    We carry the source's architecture-defining fields so the conv tower is
    #    structurally identical; only board_size changes. build_model only
    #    overrides non-default knobs, which keeps the off-path byte-identical.
    gp = src_cfg.get("global_pool", False)
    src_value_head = src_cfg.get("value_head", "scalar")
    # The target value head is the source's unless an override is requested
    # (e.g. "wdl" warm-started from a "scalar" champion -> fresh WDL head).
    tgt_value_head = target_value_head or src_value_head
    head_swapped = tgt_value_head != src_value_head
    target = build_model(
        size,
        stem_padding=int(src_cfg.get("stem_padding", 1)),
        global_pool=(None if gp is False or gp == 0 else gp),
        value_head=tgt_value_head,
        value_hlgauss_bins=src_cfg.get("value_hlgauss_bins"),
        value_hlgauss_sigma=src_cfg.get("value_hlgauss_sigma"),
        activation=src_cfg.get("activation", "relu"),
        aux_opponent_reply=bool(src_cfg.get("aux_opponent_reply", False)),
        aux_ownership=bool(src_cfg.get("aux_ownership", False)),
    )
    assert isinstance(target, GomokuNet)
    if target.cfg.board_size != target_board:
        raise SystemExit(
            f"target net built at board_size={target.cfg.board_size}, expected "
            f"{target_board}; check GOMOKU_BOARD_SIZE."
        )

    tgt_state = target.state_dict()

    # 3) Partial load: copy every tensor whose name+shape matches; record skips.
    transferred: list[str] = []
    skipped_shape: list[str] = []   # present in both, shape differs (board-bound)
    skipped_missing: list[str] = []  # in source but not in target (arch divergence)

    to_load: dict = {}
    for name, src_t in src_state.items():
        if name not in tgt_state:
            skipped_missing.append(name)
            continue
        if tuple(src_t.shape) == tuple(tgt_state[name].shape):
            to_load[name] = src_t.clone()
            transferred.append(name)
        else:
            skipped_shape.append(name)

    # Apply (strict=False also tolerates target-only keys; we report them).
    missing, unexpected = target.load_state_dict(to_load, strict=False)
    # `missing` = target keys we did NOT supply (the fresh-init FCs + any
    # source-missing keys); `unexpected` should be empty since we filtered to
    # target keys above.
    if unexpected:
        raise SystemExit(
            f"unexpected keys when loading warm-start weights: {sorted(unexpected)}"
        )

    # 4) ASSERT the entire conv tower transferred. A board-bound FC skip is
    #    expected; anything else (a conv / BN / pool_fc tensor that did NOT
    #    transfer) means the architectures diverged and the warm-start is unsafe.
    #    When the value head is deliberately swapped (--target-value-head), the
    #    source's value-head FC (e.g. scalar value_fc2 -> WDL value_wdl_fc) is an
    #    INTENDED fresh-head skip, not an arch mismatch, so it is tolerated too.
    tolerated = set(_EXPECTED_BOARD_BOUND_KEYS)
    if head_swapped:
        tolerated |= {
            k for k in (skipped_shape + skipped_missing) if _is_value_head_fc_key(k)
        }
    surprises = sorted(set(skipped_shape + skipped_missing) - tolerated)
    if surprises:
        raise SystemExit(
            "ARCH MISMATCH: these source tensors did NOT transfer and are not "
            "the expected board-bound FCs (policy_fc.weight/bias, "
            f"value_fc1.weight): {surprises}. Source and target architectures "
            "disagree — refusing to write a misleading warm-start seed.\n"
            f"  source cfg: {src_cfg}\n"
            f"  target cfg: {vars(target.cfg)}"
        )

    # Also assert: every conv/BN/global-pool tower tensor in the TARGET that has
    # a same-named source key was actually filled (belt-and-suspenders against a
    # silent shape divergence inside the tower).
    tower_prefixes = ("stem.", "tower.", "policy_conv", "policy_bn",
                      "value_conv", "value_bn")
    tower_keys = [
        k for k in tgt_state
        if k.startswith(tower_prefixes) and k in src_state
    ]
    not_filled = sorted(set(tower_keys) - set(transferred))
    if not_filled:
        raise SystemExit(
            f"ARCH MISMATCH: conv-tower tensors present in both nets but NOT "
            f"transferred (shape divergence inside the tower): {not_filled}"
        )

    # 5) Param-transfer accounting (over the SOURCE net's parameters).
    src_total_params = sum(int(t.numel()) for t in src_state.values())
    transferred_params = sum(int(src_state[k].numel()) for k in transferred)
    skipped_params = src_total_params - transferred_params
    pct = 100.0 * transferred_params / src_total_params if src_total_params else 0.0

    # 6) Save in the trainer's --resume format. Fresh timeline:
    #    - embedded ModelConfig with board_size = target (via save_checkpoint,
    #      which serialises target.cfg).
    #    - epoch / total_games = 0 (cold timeline).
    #    - NO optimizer_state_dict  -> trainer builds a fresh AdamW.
    #    - NO replay_buffer         -> trainer starts with an empty buffer.
    #    - NO wandb_run_id          -> trainer starts a fresh wandb run.
    save_checkpoint(
        out_path,
        target,
        optimizer=None,
        epoch=0,
        total_games=0,
        wandb_run_id=None,
        extra={
            "warmstart_from": os.path.abspath(src_path),
            "warmstart_source_board_size": src_board,
            "warmstart_target_board_size": target_board,
        },
    )

    return {
        "src_path": src_path,
        "out_path": out_path,
        "src_board": src_board,
        "target_board": target_board,
        "size": size,
        "src_cfg": src_cfg,
        "target_cfg": dict(vars(target.cfg)),
        "value_head_swapped": head_swapped,
        "n_src_keys": len(src_state),
        "n_transferred": len(transferred),
        "n_skipped": len(skipped_shape) + len(skipped_missing),
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
        "src_total_params": src_total_params,
        "transferred_params": transferred_params,
        "skipped_params": skipped_params,
        "pct_params_transferred": pct,
        "fresh_init_keys": sorted(missing),
    }


def _print_summary(s: dict) -> None:
    print("=" * 68)
    print("CROSS-BOARD WARM-START SUMMARY")
    print("=" * 68)
    print(f"  source : {s['src_path']}  (board {s['src_board']}x{s['src_board']})")
    print(f"  target : {s['out_path']}  (board {s['target_board']}x{s['target_board']}, size={s['size']})")
    print(f"  source arch: n_filters={s['src_cfg'].get('n_filters')} "
          f"n_blocks={s['src_cfg'].get('n_blocks')} "
          f"stem_padding={s['src_cfg'].get('stem_padding')} "
          f"global_pool={s['src_cfg'].get('global_pool')} "
          f"value_head={s['src_cfg'].get('value_head')} "
          f"activation={s['src_cfg'].get('activation')}")
    print(f"  target arch: n_filters={s['target_cfg'].get('n_filters')} "
          f"n_blocks={s['target_cfg'].get('n_blocks')} "
          f"stem_padding={s['target_cfg'].get('stem_padding')} "
          f"global_pool={s['target_cfg'].get('global_pool')} "
          f"value_head={s['target_cfg'].get('value_head')} "
          f"activation={s['target_cfg'].get('activation')}")
    print("-" * 68)
    print(f"  tensors transferred : {s['n_transferred']} / {s['n_src_keys']} source keys")
    print(f"  tensors skipped     : {s['n_skipped']} "
          f"(board-bound FCs kept fresh-init)")
    if s["skipped_shape"]:
        print(f"    shape-mismatch skips : {s['skipped_shape']}")
    if s["skipped_missing"]:
        print(f"    source-only skips    : {s['skipped_missing']}")
    print(f"  params transferred  : {s['transferred_params']:,} / "
          f"{s['src_total_params']:,}  "
          f"({s['pct_params_transferred']:.1f}%)")
    print(f"  params re-initialised: {s['skipped_params']:,}  "
          f"(fresh init at {s['target_board']}x{s['target_board']})")
    print(f"  fresh-init target keys: {s['fresh_init_keys']}")
    print("=" * 68)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="src", required=True,
                   help="9x9 source checkpoint to warm-start FROM (e.g. the "
                        "v8 champion peak.pt)")
    p.add_argument("--out", dest="out", required=True,
                   help="output 15x15 seed checkpoint to write (trainer "
                        "--resume format)")
    p.add_argument("--size", default="small",
                   help="target architecture size preset (default: small)")
    p.add_argument("--board-size", type=int, default=15,
                   help="target board size (default: 15)")
    p.add_argument("--target-value-head", default=None,
                   choices=["scalar", "wdl", "hlgauss"],
                   help="override the target's value representation (default: "
                        "inherit the source's). Use 'wdl' to warm-start a WDL "
                        "net's conv tower from a scalar champion — the value "
                        "head's final FC is then a FRESH init (the G15-wdl "
                        "warm-tower + fresh-wdl-head recipe).")
    args = p.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit(f"source checkpoint not found: {args.src}")

    summary = warmstart(
        args.src,
        args.out,
        size=args.size,
        board_size=args.board_size,
        target_value_head=args.target_value_head,
    )
    _print_summary(summary)
    sys.exit(0)


if __name__ == "__main__":
    main()
