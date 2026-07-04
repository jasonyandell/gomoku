"""Smoke test for the moonshot VCT-defense auxiliary head.

Exercises all six seams end-to-end WITHOUT launching real training:
  1. no-op invariant  — aux_vct off state_dict identical; on adds ONLY vct_* keys
  2. forward          — return_vct=True shape (B, N_ACTIONS); off-model raises
  3. loss path        — masked BCE term finite + logged; weight 0.0 byte-identical
  4. labeler          — _vct_defense_solve returns per-cell map+mask, prints fired
  5. tiny end-to-end  — generate_games(record_vct=True) -> buffer -> one SGD step

Run:  uv run python scripts/smoke_vct_head.py
NEVER SIGKILL this process mid-run: the labeler/terminus use the MLX Metal
solver, and killing it mid-compile wedges the Metal compiler service.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES, HISTORY_PLY
from gomoku.model import build_model
from gomoku.train import train_step
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import (
    _vct_defense_solve,
    configure_vct_terminus,
    generate_games,
)
from gomoku.mcts import make_torch_evaluator


_FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _FAILS.append(name)


def _rand_planes(b: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE, generator=g)


# --------------------------------------------------------------------------- #
# Check 1 — no-op invariant
# --------------------------------------------------------------------------- #
def check_1_invariant() -> None:
    print("\n=== Check 1: no-op invariant (state_dict keys) ===")
    off = build_model("small", aux_vct=False)
    on = build_model("small", aux_vct=True)
    keys_off = set(off.state_dict().keys())
    keys_on = set(on.state_dict().keys())

    check("aux_vct=False adds NO vct_* keys",
          not any(k.startswith("vct_") for k in keys_off),
          f"{sum(k.startswith('vct_') for k in keys_off)} vct_ keys off")

    new = keys_on - keys_off
    removed = keys_off - keys_on
    check("aux_vct=True removes no keys", not removed, f"removed={sorted(removed)}")
    check("aux_vct=True's ONLY new keys are vct_*",
          bool(new) and all(k.startswith("vct_") for k in new),
          f"new={sorted(new)}")


# --------------------------------------------------------------------------- #
# Check 2 — forward
# --------------------------------------------------------------------------- #
def check_2_forward() -> None:
    print("\n=== Check 2: forward(return_vct=True) ===")
    b = 4
    x = _rand_planes(b, seed=1)

    on = build_model("small", aux_vct=True).eval()
    with torch.no_grad():
        out = on(x, return_vct=True)
    vlog = out[-1]
    check("return_vct appends a tensor of shape (B, N_ACTIONS)",
          tuple(vlog.shape) == (b, N_ACTIONS), f"shape={tuple(vlog.shape)}")

    off = build_model("small", aux_vct=False).eval()
    raised = False
    try:
        with torch.no_grad():
            off(x, return_vct=True)
    except RuntimeError:
        raised = True
    check("aux_vct=False model raises RuntimeError on return_vct=True", raised)


# --------------------------------------------------------------------------- #
# Check 3 — loss path
# --------------------------------------------------------------------------- #
def check_3_loss() -> None:
    print("\n=== Check 3: masked-BCE vct loss path ===")
    b = 8
    rng = np.random.default_rng(3)
    planes = torch.from_numpy(rng.random((b, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)).astype(np.float32))
    pi_np = rng.random((b, N_ACTIONS)).astype(np.float32)
    pi = torch.from_numpy(pi_np / pi_np.sum(1, keepdims=True))
    z = torch.from_numpy((rng.random(b).astype(np.float32) * 2 - 1))
    vct_t = torch.from_numpy((rng.random((b, N_ACTIONS)) < 0.3).astype(np.float32))
    vct_mask = torch.from_numpy(np.array([i % 2 == 0 for i in range(b)], dtype=bool))

    model = build_model("small", aux_vct=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    m = train_step(model, opt, planes, pi, z,
                   vct=vct_t, vct_mask=vct_mask, vct_weight=0.1,
                   do_optimizer_step=False)
    check("loss/total finite with vct_weight=0.1",
          np.isfinite(m["loss/total"]), f"loss={m['loss/total']:.4f}")
    check("train/vct_loss present and finite",
          "train/vct_loss" in m and np.isfinite(m["train/vct_loss"]),
          f"vct_loss={m.get('train/vct_loss')}")
    check("train/vct_mask_frac present",
          "train/vct_mask_frac" in m, f"mask_frac={m.get('train/vct_mask_frac')}")

    # Byte-identical: vct_weight=0.0 (use_vct False) == a call with no vct args,
    # on two identical models (compare the loss computed on the shared weights).
    a = build_model("small", aux_vct=True)
    bmodel = copy.deepcopy(a)
    oa = torch.optim.AdamW(a.parameters(), lr=1e-3)
    ob = torch.optim.AdamW(bmodel.parameters(), lr=1e-3)
    ma = train_step(a, oa, planes, pi, z,
                    vct=vct_t, vct_mask=vct_mask, vct_weight=0.0,
                    do_optimizer_step=False)
    mb = train_step(bmodel, ob, planes, pi, z, do_optimizer_step=False)
    check("vct_weight=0.0 loss byte-identical to a no-vct call",
          ma["loss/total"] == mb["loss/total"],
          f"{ma['loss/total']!r} vs {mb['loss/total']!r}")
    check("vct_weight=0.0 emits NO train/vct_loss key",
          "train/vct_loss" not in ma)


# --------------------------------------------------------------------------- #
# Check 4 — labeler
# --------------------------------------------------------------------------- #
def _planes_with_opp_four() -> np.ndarray:
    """Position (side-to-move = me, plane 0) where the OPPONENT (plane HISTORY_PLY)
    has four-in-a-row on row 4 cols 0..3 with the open end at (4,4). It is my move.
    For almost any cell m I play, the opponent then plays (4,4) for five => a
    forced VCT; only playing (4,4) myself blocks it."""
    p = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for c in range(4):
        p[HISTORY_PLY, 4, c] = 1.0          # opponent four
    p[0, 0, 0] = 1.0                        # one of my stones somewhere harmless
    return p


def check_4_labeler() -> None:
    print("\n=== Check 4: _vct_defense_solve labeler ===")
    planes_list = [_planes_with_opp_four()]
    maps, masks = _vct_defense_solve(planes_list, max_cands=0)
    mp, mk = maps[0], masks[0]

    check("map shape/dtype (N_ACTIONS,) float32",
          mp.shape == (N_ACTIONS,) and mp.dtype == np.float32, f"{mp.shape} {mp.dtype}")
    check("mask shape/dtype (N_ACTIONS,) bool",
          mk.shape == (N_ACTIONS,) and mk.dtype == np.bool_, f"{mk.shape} {mk.dtype}")

    occupied = (planes_list[0][0].astype(bool) | planes_list[0][HISTORY_PLY].astype(bool))
    n_empty = int((~occupied.reshape(-1)).sum())
    check("mask covers exactly the legal empty cells (max_cands=0)",
          int(mk.sum()) == n_empty, f"mask={int(mk.sum())} empties={n_empty}")

    fired = int((mp > 0.5).sum())
    saving = 4 * BOARD_SIZE + 4          # flat index of (4,4), the blocking move
    print(f"    cells fired (blunders)= {fired} / {n_empty} empties; "
          f"map[block(4,4)]={mp[saving]:.0f}")
    check("at least one blunder cell fired", fired > 0, f"fired={fired}")
    check("the blocking move (4,4) is NOT a blunder", mp[saving] == 0.0,
          f"map[block]={mp[saving]}")


# --------------------------------------------------------------------------- #
# Check 5 — tiny end-to-end
# --------------------------------------------------------------------------- #
def check_5_end_to_end() -> None:
    print("\n=== Check 5: tiny end-to-end (gen -> label -> buffer -> SGD) ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # A live vctsci-terminus run shares the GPU; keep the labeler breadth capped so
    # this smoke's per-ply bulk solve stays tiny (check 4 covers max_cands=0).
    configure_vct_terminus(enabled=True, budget=50, defense_max_cands=6)

    model = build_model("small", aux_vct=True).to(device)
    evaluator = make_torch_evaluator(model, device)

    t0 = time.perf_counter()
    records = generate_games(
        2, evaluator,
        n_simulations=8,
        max_plies=12,
        temperature_moves=30,
        record_vct=True,
    )
    dt = time.perf_counter() - t0
    gps = len(records) / dt if dt > 0 else float("nan")

    examples = [e for r in records for e in r.examples]
    labeled = [e for e in examples if getattr(e, "vct", None) is not None]
    fired = int(sum(int((e.vct > 0.5).sum()) for e in labeled))
    print(f"    generated {len(records)} games in {dt:.1f}s ({gps:.2f} games/s), "
          f"{len(examples)} examples, {len(labeled)} vct-labeled, {fired} cells fired total")
    check("generate_games(record_vct=True) produced games", len(records) >= 1)
    check("some recorded examples carry a vct target", len(labeled) > 0,
          f"{len(labeled)}/{len(examples)}")
    # D4 sanity: an example's vct map is a per-cell (N_ACTIONS,) float32 board map.
    if labeled:
        e0 = labeled[0]
        check("vct example field shape/dtype (N_ACTIONS,) float32",
              e0.vct.shape == (N_ACTIONS,) and e0.vct.dtype == np.float32,
              f"{e0.vct.shape} {e0.vct.dtype}")

    # Buffer round-trip + one SGD step through the real sample() plumbing.
    buf = ReplayBuffer(200_000, device=device, aux_vct=True)
    buf.add(examples)
    bs = min(64, buf.size)
    planes, pi, z, side, ply, vct, vct_mask = buf.sample(bs, return_vct=True)
    print(f"    buffer size={buf.size}, sampled batch={bs}, "
          f"vct_mask_frac={float(vct_mask.float().mean()):.3f}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    m = train_step(model, opt, planes, pi, z, side=side, ply=ply,
                   vct=vct, vct_mask=vct_mask, vct_weight=0.1,
                   do_optimizer_step=True)
    check("SGD step on a vct batch produced finite loss",
          np.isfinite(m["loss/total"]), f"loss={m['loss/total']:.4f}")
    check("train/vct_loss logged in end-to-end SGD step",
          "train/vct_loss" in m and np.isfinite(m["train/vct_loss"]),
          f"vct_loss={m.get('train/vct_loss')}")


def main() -> int:
    torch.manual_seed(0)
    check_1_invariant()
    check_2_forward()
    check_3_loss()
    check_4_labeler()
    check_5_end_to_end()
    print("\n" + "=" * 60)
    if _FAILS:
        print(f"RESULT: {len(_FAILS)} FAIL(s): {_FAILS}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
