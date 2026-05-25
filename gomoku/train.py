"""AlphaZero-style training loop: self-play -> replay buffer -> SGD -> repeat."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gomoku.eval import mcts_picker, play_match_pickers
from gomoku.match import build_player, parse_spec
from gomoku.mcts import make_torch_evaluator
from gomoku.model import build_model, load_checkpoint, n_params, save_checkpoint
from gomoku.replay_buffer import ReplayBuffer
from gomoku.self_play import generate_games, generate_games_vs_baseline
from gomoku.util import load_wandb_key_from_keychain, pick_device


class _ShutdownRequested(Exception):
    """Raised from inside a blocking worker-ingest wait when SIGTERM/SIGINT
    has set the stop flag. Lets a wait that would otherwise never return
    (e.g. no workers are alive to deliver the next wave) unwind to the
    epoch-boundary clean-stop path instead of hanging until SIGKILL."""


def policy_loss(logits: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft targets. pi may have zeros on illegal moves; those contribute 0."""
    logp = F.log_softmax(logits, dim=-1)
    return -(pi * logp).sum(dim=-1).mean()


def value_loss(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v, z)


def train_step(
    model,
    optimizer,
    planes: torch.Tensor,
    pi: torch.Tensor,
    z: torch.Tensor,
    *,
    value_weight: float = 1.0,
    l2_weight: float = 1e-4,
    accum_scale: float = 1.0,
    do_optimizer_step: bool = True,
    zero_grad: bool = True,
    side: torch.Tensor | None = None,
    ply: torch.Tensor | None = None,
    aux_pi: torch.Tensor | None = None,
    aux_mask: torch.Tensor | None = None,
    aux_weight: float = 0.0,
    ownership: torch.Tensor | None = None,
    ownership_mask: torch.Tensor | None = None,
    ownership_weight: float = 0.0,
) -> dict[str, float]:
    """One forward + backward. By default does optimizer.step() + zero_grad
    (legacy behavior). For WL2 gradient accumulation, the caller controls
    those: pass accum_scale=1/N to scale loss before backward, zero_grad=True
    only on the first microbatch of an accumulation window, and
    do_optimizer_step=True only on the last microbatch.

    V3 aux opponent-reply head: when aux_weight > 0 (and the model has the aux
    head + aux_pi/aux_mask are provided), a second forward output predicts the
    opponent's next-ply policy; its masked soft-target CE is added as
    aux_weight * aux_ce.

    V4 aux ownership head: when ownership_weight > 0 (and the model has the
    ownership head + ownership/ownership_mask are provided), a dense head
    predicts per-cell final control; its masked MSE against the [-1,1] target is
    added as ownership_weight * own_mse. The two aux heads are independent.

    With BOTH weights == 0.0 (default) the forward, loss, and SGD graph are
    byte-identical to the pre-aux path — neither aux head is even run."""
    model.train()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    use_aux = aux_weight > 0.0 and aux_pi is not None and aux_mask is not None
    use_ownership = (
        ownership_weight > 0.0 and ownership is not None and ownership_mask is not None
    )
    aux_logits = None
    own_logits = None
    if use_aux or use_ownership:
        outputs = model(planes, return_aux=use_aux, return_ownership=use_ownership)
        # forward() returns (p, v, [aux], [ownership]) in that fixed order;
        # unpack positionally per the flags requested above.
        logits, v = outputs[0], outputs[1]
        idx_extra = 2
        if use_aux:
            aux_logits = outputs[idx_extra]
            idx_extra += 1
        if use_ownership:
            own_logits = outputs[idx_extra]
            idx_extra += 1
    else:
        # Byte-identical default call (no kwargs) so mock/legacy models whose
        # forward() takes only `planes` keep working unchanged.
        logits, v = model(planes)
    logp = F.log_softmax(logits, dim=-1)
    # Per-sample policy CE so we can split by side/ply without a second pass.
    per_policy_ce = -(pi * logp).sum(dim=-1)
    pl = per_policy_ce.mean()
    per_value_se = (v - z) ** 2
    vl = per_value_se.mean()
    loss = pl + value_weight * vl
    aux_l_val = float("nan")
    if use_aux:
        aux_logp = F.log_softmax(aux_logits, dim=-1)
        per_aux_ce = -(aux_pi * aux_logp).sum(dim=-1)  # (B,) soft-target CE
        mask = aux_mask.bool()
        if bool(mask.any()):
            aux_l = per_aux_ce[mask].mean()
        else:
            # No valid aux rows this batch — contribute nothing, keep graph
            # connected to the aux head with a zero-scaled term so params still
            # receive (zero) grad rather than an autograd error.
            aux_l = per_aux_ce.sum() * 0.0
        loss = loss + aux_weight * aux_l
        aux_l_val = float(aux_l.detach())
    own_l_val = float("nan")
    if use_ownership:
        # Masked MSE: per-cell (B, 81) squared error, mean over cells then mean
        # over the valid rows. The ownership head emits raw per-cell logits; the
        # target lives in [-1, 1] (winner/loser/empty), so a plain MSE (no tanh)
        # is the regression loss — matches KataGo's ownership objective shape.
        per_own_se = ((own_logits - ownership) ** 2).mean(dim=-1)  # (B,)
        omask = ownership_mask.bool()
        if bool(omask.any()):
            own_l = per_own_se[omask].mean()
        else:
            own_l = per_own_se.sum() * 0.0  # keep head connected, zero grad
        loss = loss + ownership_weight * own_l
        own_l_val = float(own_l.detach())
    if l2_weight > 0:
        l2 = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
        loss = loss + l2_weight * l2
    # Scale loss BEFORE backward when accumulating, so the summed gradient
    # over N microbatches equals the gradient of the mean loss over N*B
    # examples (modulo the L2 reg, which is identical across microbatches).
    (loss * accum_scale).backward()
    if do_optimizer_step:
        optimizer.step()
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        target = pi.argmax(dim=-1)
        acc = (pred == target).float().mean()
        target_entropy = -(pi * torch.log(pi.clamp_min(1e-9))).sum(dim=-1).mean()
        policy_kl = pl - target_entropy
        p = logp.exp()
        net_entropy = -(p * logp).sum(dim=-1).mean()
    out: dict[str, float] = {
        "loss/total": float(loss.detach()),
        "loss/policy": float(pl.detach()),
        "loss/value": float(vl.detach()),
        "train/policy_acc": float(acc),
        "train/policy_target_entropy": float(target_entropy),
        "train/policy_net_entropy": float(net_entropy),
        "train/policy_kl": float(policy_kl),
    }
    if use_aux:
        out["loss/aux_policy"] = aux_l_val
        out["train/aux_mask_frac"] = float(aux_mask.bool().float().mean())
    if use_ownership:
        out["loss/aux_ownership"] = own_l_val
        out["train/ownership_mask_frac"] = float(ownership_mask.bool().float().mean())
    if side is not None and ply is not None:
        with torch.no_grad():
            ce_cpu = per_policy_ce.detach()
            ve_cpu = per_value_se.detach()
            side_long = side.long()
            for s in (0, 1):
                mask = (side_long == s)
                if bool(mask.any()):
                    out[f"train/policy_ce/side_{s}"] = float(ce_cpu[mask].mean())
                    out[f"train/value_mse/side_{s}"] = float(ve_cpu[mask].mean())
            ply_long = ply.long()
            for lo, hi, label in ((0, 10, "ply_00_10"), (10, 25, "ply_10_25"), (25, 60, "ply_25_60")):
                mask = (ply_long >= lo) & (ply_long < hi)
                if bool(mask.any()):
                    out[f"train/policy_ce/{label}"] = float(ce_cpu[mask].mean())
                    out[f"train/value_mse/{label}"] = float(ve_cpu[mask].mean())
    return out


def ema_update(ema_model, model, tau: float) -> None:
    """EMA update on parameters AND buffers (batchnorm running stats):
    ema <- tau * ema + (1 - tau) * src. No-grad. WL2 lever #1."""
    with torch.no_grad():
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.mul_(tau).add_(p.detach(), alpha=1.0 - tau)
        # Keep buffers (e.g. BN running stats) in lockstep too. Strict copy
        # rather than EMA is fine — buffers update during training forward,
        # not via gradients — but copy keeps semantics simple and matches
        # what most EMA implementations do.
        for eb, b in zip(ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def ema_l2_distance(ema_model, model) -> float:
    """Flat-vector L2 norm of (theta - theta_ema) across all parameters."""
    with torch.no_grad():
        sq = 0.0
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            sq += float((p - ep).pow(2).sum())
        return sq ** 0.5


def average_state_dicts(state_dicts: list[dict]) -> dict:
    """V3 SWA lever: flat (uniform) average of a list of model state_dicts.

    Returns a new state_dict where each FLOAT tensor (conv/linear weights,
    BatchNorm running_mean / running_var, etc.) is the elementwise arithmetic
    mean across the inputs, and each non-float tensor (e.g. BatchNorm
    `num_batches_tracked`, an int64 counter) is taken from the LATEST
    state_dict — averaging an integer batch-counter is meaningless, and the
    latest value is the most representative for inference.

    Averaging BatchNorm running stats across checkpoints is standard for SWA
    inference: the running_mean / running_var are floats that move slowly, so a
    flat tail-average yields valid inference-time normalization without needing
    a BN-recompute pass. (Classic SWA recomputes BN stats with a forward pass;
    here the published model is for GENERATION/inference only and the running
    stats are already well-tracked from training, so averaging is sufficient.)

    All inputs must share the same keys/shapes. Computes in float64 then casts
    back to each tensor's original float dtype to avoid accumulation drift.
    """
    if not state_dicts:
        raise ValueError("average_state_dicts requires at least one state_dict")
    latest = state_dicts[-1]
    out: dict = {}
    n = len(state_dicts)
    for key, ref in latest.items():
        if torch.is_tensor(ref) and ref.is_floating_point():
            acc = state_dicts[0][key].detach().to(torch.float64)
            for sd in state_dicts[1:]:
                acc = acc + sd[key].detach().to(torch.float64)
            acc = acc / n
            out[key] = acc.to(ref.dtype)
        else:
            # int/long buffers (num_batches_tracked) or non-tensors: take latest.
            out[key] = ref.detach().clone() if torch.is_tensor(ref) else ref
    return out


def _baseline_log_key(spec) -> str:
    """Turn a PlayerSpec into a stable wandb-friendly metric key suffix.

    Examples:
        random                  -> vs_random
        heuristic               -> vs_heuristic
        lookahead:depth=4       -> vs_lookahead4
        lookahead               -> vs_lookahead
    """
    if spec.kind == "lookahead":
        depth = spec.kwargs.get("depth", "")
        return f"vs_lookahead{depth}"
    return f"vs_{spec.kind}"


def _score_validation_archive(model, archive: dict, device) -> dict[str, float]:
    """Score the model against a frozen validation archive.

    Returns overall val/* metrics plus per-bucket variants tagged by
    provenance. The archive format is the one produced by
    scripts/mine_validation_archive.py.
    """
    planes = archive["planes"].to(device)
    pi = archive["pi_mcts"].to(device)
    z = archive["z"].to(device)
    provenance: list[str] = list(archive["provenance"])
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            n = planes.shape[0]
            # Score in chunks to avoid blowing MPS memory on huge archives.
            chunk = 512
            ce_parts: list[torch.Tensor] = []
            se_parts: list[torch.Tensor] = []
            acc_parts: list[torch.Tensor] = []
            kl_parts: list[torch.Tensor] = []
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                logits, v = model(planes[start:stop])
                logp = F.log_softmax(logits, dim=-1)
                ce = -(pi[start:stop] * logp).sum(dim=-1)
                ent = -(pi[start:stop] * torch.log(pi[start:stop].clamp_min(1e-9))).sum(dim=-1)
                ce_parts.append(ce.cpu())
                kl_parts.append((ce - ent).cpu())
                se_parts.append(((v - z[start:stop]) ** 2).cpu())
                acc_parts.append((logits.argmax(dim=-1) == pi[start:stop].argmax(dim=-1)).float().cpu())
            ce_all = torch.cat(ce_parts)
            kl_all = torch.cat(kl_parts)
            se_all = torch.cat(se_parts)
            acc_all = torch.cat(acc_parts)
    finally:
        if was_training:
            model.train()
    out: dict[str, float] = {
        "val/policy_ce": float(ce_all.mean()),
        "val/policy_kl": float(kl_all.mean()),
        "val/value_mse": float(se_all.mean()),
        "val/policy_acc": float(acc_all.mean()),
    }
    buckets: dict[str, list[int]] = {}
    for i, tag in enumerate(provenance):
        buckets.setdefault(tag, []).append(i)
    for tag, idxs in buckets.items():
        if not idxs:
            continue
        idx_t = torch.tensor(idxs, dtype=torch.long)
        out[f"val/policy_ce/{tag}"] = float(ce_all[idx_t].mean())
        out[f"val/policy_kl/{tag}"] = float(kl_all[idx_t].mean())
        out[f"val/value_mse/{tag}"] = float(se_all[idx_t].mean())
    return out


def _parse_specs(s: str) -> list:
    s = s.strip()
    if not s:
        return []
    out = []
    for raw in s.split(","):
        raw = raw.strip()
        if not raw:
            continue
        spec = parse_spec(raw)
        if spec.kind == "model":
            raise SystemExit(f"--eval-baselines must not include model specs: {raw!r}")
        out.append(spec)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--size", type=str, default="small", help="tiny / small / medium / large")
    p.add_argument("--stem-padding", type=int, default=None,
                   help="Override ModelConfig.stem_padding (default 3 = michaelnny's "
                        "edge-fix). Set to 1 for the legacy 9x9 internal feature map, "
                        "which is ~2x cheaper per forward but loses the edge-blocking fix.")
    p.add_argument("--global-pool", type=int, nargs="?", const=-1, default=None,
                   help="Derby v4 'Whole-board' lever: KataGo-style global pooling "
                        "in the residual tower. Bare --global-pool applies it to the "
                        "latter half of blocks; --global-pool K applies it to the "
                        "trailing K blocks. Omitted = OFF (byte-identical current arch).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--games-per-epoch", type=int, default=64)
    p.add_argument("--n-simulations", type=int, default=100)
    p.add_argument("--wave-size", type=int, default=1,
                   help="zeb-style wave-batched MCTS: leaves collected per game per "
                        "evaluator call. 1 = original behavior, 8-32 = much bigger "
                        "batches via virtual loss (fewer MPS syncs).")
    p.add_argument("--opponent", type=str, default="self",
                   help="Self-play opponent. 'self' (default) = model vs model with "
                        "MCTS both sides. Any other valid player spec "
                        "(e.g. 'heuristic', 'random', 'lookahead:depth=2') = model "
                        "plays MCTS against a fixed picker — only model plies "
                        "contribute training examples. Stationary opponents force "
                        "decisive games and break self-play attractors.")
    p.add_argument("--model-first-frac", type=float, default=0.5,
                   help="Fraction of games where model plays the first move when "
                        "--opponent != self. 0.5 = model sees both sides equally.")
    p.add_argument("--opponent-mix-random", type=float, default=0.0,
                   help="If > 0, wraps the opponent picker so it plays a uniform "
                        "random legal move with this probability per turn, and "
                        "the configured opponent otherwise. Useful to weaken a "
                        "strong baseline so the model gets a sometimes-win signal.")
    p.add_argument("--random-opening-moves", type=int, default=0,
                   help="If > 0, each self-play / vs-baseline game starts with this "
                        "many uniformly-random legal moves played (alternating sides). "
                        "MCTS only takes over after that, and no training examples are "
                        "recorded for the random opening. Breaks the 'always-same-opening' "
                        "collapse by forcing the model to learn from diverse positions.")
    p.add_argument("--c-puct", type=float, default=1.25,
                   help="c_puct_init in the AGZ log-schedule PUCT formula. Effective "
                        "exploration constant at N_parent=0. Default 1.25 = AGZ value.")
    p.add_argument("--c-puct-base", type=float, default=19652.0,
                   help="c_puct_base in the AGZ log-schedule PUCT formula. Controls "
                        "how fast the effective exploration grows with parent visits. "
                        "AGZ default 19652 keeps it nearly constant at our sim budgets.")
    p.add_argument("--temperature-moves", type=int, default=8)
    p.add_argument("--temperature-final", type=float, default=0.1,
                   help="After the warm-up plies (--temperature-moves), sample MCTS "
                        "actions at this temperature instead of fully greedy (tau=0). "
                        "Default 0.1 matches michaelnny/alpha_zero — sharp but not "
                        "one-hot, preserves a tiny amount of whole-game exploration "
                        "and keeps the policy training target as a real distribution.")
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--replay-buffer-size", type=int, default=1_500_000,
                   help="Capacity of the replay ring buffer. Default 1.5M matches "
                        "michaelnny/alpha_zero's gomoku config — much larger than our "
                        "earlier 500k runs, which gives the network more diverse "
                        "opponent-version exposure before old positions evict.")
    p.add_argument("--training-steps", type=int, default=400,
                   help="SGD steps per epoch. Static unless --sgd-per-game is set.")
    p.add_argument("--sgd-per-game", type=float, default=None,
                   help="Dynamic balance knob. If set, each cycle computes "
                        "training_steps = max(--min-training-steps, "
                        "round(K * games_ingested_this_cycle)). Overrides "
                        "--training-steps. Keeps SGD intensity in lockstep with "
                        "actual gen throughput from the workers — if they speed up "
                        "or slow down, training scales with them. K ≈ 1.0 matches "
                        "the AZ recipe (1 SGD step per game played). NOTE: when "
                        "game length varies (model swings between attack and defense), "
                        "this knob also varies the per-cycle SGD count and the buffer "
                        "turnover rate. Prefer --sgd-per-position for stable runs. "
                        "LF1 NOTE: as an anti-runaway schedule this has LOWER gain "
                        "than --sgd-per-position, because the tile's game COUNT grows "
                        "slower than its POSITION count (games lengthen as defense is "
                        "learned). It does not fully close the runaway loop on its "
                        "own — pair with --max-tile-games / --max-sgd-steps-per-epoch "
                        "for a structural bound.")
    p.add_argument("--sgd-per-position", type=float, default=None,
                   help="Position-based version of --sgd-per-game. Each cycle "
                        "computes training_steps = max(--min-training-steps, "
                        "round(K * positions_ingested_this_cycle)). Keeps SGD "
                        "intensity proportional to fresh training examples, not "
                        "fresh games. Pair with --worker-min-positions to also "
                        "fix the ingest side, otherwise buffer turnover still "
                        "varies with game length. Overrides --sgd-per-game when set.")
    p.add_argument("--min-training-steps", type=int, default=16,
                   help="When --sgd-per-game / --sgd-per-position is set, never do "
                        "fewer than this many SGD steps in a cycle (keeps the "
                        "trainer moving even if workers stall briefly).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--aux-opponent-reply-weight", type=float, default=0.0,
                   help="V3 KataGo-style auxiliary opponent-reply policy head. "
                        "0.0 (default) = OFF = byte-identical to today (head not "
                        "constructed, no aux forward, buffer aux tensors not "
                        "allocated, SGD graph unchanged). When > 0, a second "
                        "81-way head predicts the opponent's MCTS policy at the "
                        "next ply (a free, dense supervised target already in the "
                        "self-play trajectory); its masked soft-target CE is added "
                        "to the loss scaled by this weight, and loss/aux_policy is "
                        "logged. The head is DROPPED at inference (self-play/eval "
                        "pay nothing). Suggested starting value 0.15; future sweep "
                        "{0.05, 0.15, 0.3}.")
    p.add_argument("--aux-ownership-weight", type=float, default=0.0,
                   help="V4 KataGo-style auxiliary ownership head (the biggest "
                        "sample-efficiency win in KataGo, adapted to gomoku). "
                        "0.0 (default) = OFF = byte-identical to today (head not "
                        "constructed, no ownership forward, buffer ownership "
                        "tensors not allocated, SGD graph unchanged). When > 0, a "
                        "dense 81-way head predicts per-cell final control "
                        "(+1 winner's stone / -1 loser's stone / 0 empty at the "
                        "played-out game's final board — a dense per-position "
                        "target); its masked MSE is added to the loss scaled by "
                        "this weight, and loss/aux_ownership is logged. The head "
                        "is DROPPED at inference (self-play/eval pay nothing). "
                        "Independent of --aux-opponent-reply-weight: either, "
                        "both, or neither head may be enabled. Suggested starting "
                        "value 0.15.")
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--ema-tau", type=float, default=0.0,
                   help="WL2 lever #1: EMA self-play weights. When > 0, the "
                        "trainer maintains an EMA copy of the model "
                        "(theta_ema <- tau*theta_ema + (1-tau)*theta after each "
                        "optimizer.step()) and publishes the EMA weights to "
                        "--worker-weights-path instead of the raw weights. "
                        "Decouples the self-play 'brain that plays' from the "
                        "'brain that learns' to emulate AZ-at-scale "
                        "publish-to-workers lag. 0.0 (default) = disabled, "
                        "matches pre-WL2 behavior. Recommended live value 0.99 "
                        "(~70 SGD-step half-life). EMA update is per-SGD-step.")
    p.add_argument("--swa-window", type=int, default=0,
                   help="V3 lever: SWA tail-averaging of the self-play generator "
                        "weights. When > 0, the weights published to workers "
                        "(--worker-weights-path) are the FLAT (uniform) average of "
                        "the last --swa-window saved-checkpoint state_dicts of the "
                        "LIVE learner, rather than an exponential moving average. "
                        "0 (default) = OFF = current behavior (EMA if --ema-tau>0, "
                        "else the live weights). SWA REPLACES the EMA-publish when "
                        "set: --swa-window and --ema-tau are alternative smoothing "
                        "strategies for the publish path, not composed. The ring "
                        "buffer is refreshed at each worker-weights publication "
                        "(every --save-every epochs), so window K spans K saves. "
                        "Float tensors (incl. BN running stats) are mean-averaged; "
                        "int buffers (num_batches_tracked) take the latest. The "
                        "averaged weights are GENERATION-only — the learner, "
                        "latest.pt and epochNNNN.pt remain the unaveraged live "
                        "weights. Recommended starting value 5.")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="WL2 lever #4: gradient accumulation. Accumulate "
                        "gradients across this many minibatches before "
                        "optimizer.step(). Effective batch becomes "
                        "grad_accum_steps * batch_size. 1 (default) = no "
                        "accumulation, matches pre-WL2 behavior. Recommended "
                        "live value 4. Note: --sgd-per-position still controls "
                        "the number of forward+backward passes per cycle; the "
                        "number of optimizer.step() calls becomes "
                        "ceil(steps_this_cycle / grad_accum_steps).")
    p.add_argument("--validation-archive-path", type=str, default=None,
                   help="Path to a frozen validation archive (.pt) produced by "
                        "scripts/mine_validation_archive.py. When set, the trainer "
                        "scores the current model against this fixed set every "
                        "--eval-every cycles and logs val/policy_ce, val/policy_kl, "
                        "val/value_mse, val/policy_acc plus per-bucket variants.")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-sims", type=int, default=50,
                   help="MCTS sims used by the model during eval matches")
    p.add_argument("--eval-baselines", type=str,
                   default="random,heuristic",
                   help="comma-separated player specs played every eval cycle. "
                        "Default dropped lookahead:depth=2 (45 s+ for noisy signal at "
                        "n=4 games — bring it back explicitly if needed)")
    p.add_argument("--eval-baseline-games", type=int, default=4,
                   help="games per matchup vs each fast baseline (kept small to stay "
                        "in 10 s total eval budget per cycle)")
    p.add_argument("--eval-baselines-slow", type=str, default="",
                   help="comma-separated player specs gated by --eval-slow-every (empty "
                        "to disable; default empty since lookahead:depth=4 dominates "
                        "an eval epoch)")
    p.add_argument("--eval-slow-every", type=int, default=4,
                   help="run --eval-baselines-slow every Nth eval cycle (=Nx eval-every epochs)")
    p.add_argument("--eval-slow-games", type=int, default=6,
                   help="games per matchup vs each slow baseline")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--save-buffer-every", type=int, default=20,
                   help="Rewrite `latest.pt` (which embeds the ~1.4 GB replay "
                        "buffer) every N epochs. Set higher than --save-every "
                        "to throttle disk IO; older latest.pt is still valid "
                        "for resume, just slightly stale.")
    p.add_argument("--keep-last-n", type=int, default=3,
                   help="Auto-prune older epoch checkpoints, keeping only this many "
                        "of the most recent ones (plus whatever `latest.pt` points to). "
                        "Set to 0 to disable pruning (old behaviour). The replay buffer "
                        "is still embedded in `latest.pt` for resume; intermediate "
                        "checkpoints are weights+optimizer only (cheap).")
    p.add_argument("--orphan-sweep-age-sec", type=float, default=300.0,
                   help="In each save cycle, delete .tmp / unrecognized .pt files in the "
                        "checkpoint dir AND any leftover *.pt.tmp / orphaned *.pt in the "
                        "worker-input-dir that are older than this many seconds. Keeps "
                        "interrupted writes from accumulating across runs.")
    p.add_argument("--max-wall-secs", type=float, default=0.0,
                   help="Time-cap the run: after this many wall-seconds (checked at "
                        "each epoch boundary), finish the current epoch, force-save a "
                        "fully-resumable latest.pt (WITH the replay buffer, ignoring "
                        "--save-buffer-every), and exit 0. 0 = no cap (run to --epochs). "
                        "This is what makes a training task a clean research SLICE: "
                        "resume from latest.pt never cold-refills the buffer. A SIGTERM "
                        "or SIGINT triggers the same clean-save path mid-run.")
    p.add_argument("--no-eval", dest="eval_in_trainer", action="store_false",
                   help="Skip the in-trainer eval block entirely. Pair with "
                        "`python -m gomoku.eval_worker` running separately — it polls "
                        "the published worker-weights file and posts results to the same "
                        "wandb run. Lets generation+training run at full speed.")
    p.set_defaults(eval_in_trainer=True)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--worker-input-dir", type=str, default=None,
                   help="If set, the trainer ingests game-record batches from this "
                        "directory (written by `gomoku.selfplay_worker` processes) "
                        "instead of running generate_games in-process. Files are "
                        "deleted after ingest. Pair with --worker-weights-path so "
                        "workers can refresh their model.")
    p.add_argument("--worker-weights-path", type=str, default=None,
                   help="If set, the trainer atomically writes its latest model "
                        "(state_dict + cfg) to this path after each checkpoint, "
                        "so worker processes can hot-reload.")
    p.add_argument("--worker-min-games", type=int, default=0,
                   help="When using --worker-input-dir, ingest at least this many "
                        "new games before starting an SGD cycle. 0 = use "
                        "--games-per-epoch as the target. NOTE: when game length "
                        "varies (attack vs defense regimes), positions-per-cycle "
                        "varies too, which changes the replay buffer turnover rate "
                        "and biases what each weight version contributes. Prefer "
                        "--worker-min-positions for runs where game length is "
                        "expected to swing.")
    p.add_argument("--worker-min-positions", type=int, default=0,
                   help="Position-based version of --worker-min-games. Ingest until "
                        "at least this many training positions (game records' "
                        "expanded examples) have arrived this cycle. Keeps buffer "
                        "turnover rate constant regardless of self-play game "
                        "length, so each weight version contributes a stable "
                        "footprint to the buffer's training distribution. "
                        "Recommended value: target_games * mean_plies * 8 augs "
                        "(e.g. 32 * 50 * 8 = 12800 for the AZ-mini recipe). "
                        "Overrides --worker-min-games when set (> 0).")
    p.add_argument("--wave-mode", action="store_true",
                   help="Distributed wave-lockstep mode. Workers write records under "
                        "worker-input-dir/v{version}/{worker}/ and the trainer waits "
                        "for --wave-games-per-worker games from each --wave-workers "
                        "worker before advancing to the next model version.")
    p.add_argument("--wave-workers", type=int, default=0,
                   help="Expected worker count for --wave-mode. Workers are assumed "
                        "to be named w0..w{N-1}, matching scripts/run_sweep.py.")
    p.add_argument("--wave-games-per-worker", type=int, default=0,
                   help="Per-worker barrier size for --wave-mode. Defaults to "
                        "--worker-min-games / --wave-workers when possible.")
    # --- LF1 anti-runaway structural levers (all OPT-IN; 0 = disabled = today) ---
    # See wiki/topics/perf-bench-vs-real-training-cost.md for the runaway loop:
    # large wave_size -> gen outpaces trainer -> per-version tile grows ->
    # sgd_per_position scales steps with inflow -> train phase lengthens ->
    # even more games pile up. These three flags close that open loop. When all
    # are unset (default 0/None), the trainer is byte-identical to today (WL5).
    p.add_argument("--max-tile-games", type=int, default=0,
                   help="LF1 anti-runaway: cap how many self-play games the trainer "
                        "ingests per model version (the 'tile'). 0 (default) = "
                        "uncapped = today's behavior. When > 0, excess games beyond "
                        "this many per version are DROPPED (deterministically, "
                        "keeping the earliest games in stable sorted order). This "
                        "structurally bounds the tile regardless of generation rate, "
                        "converting the open generation->ingest loop into a closed "
                        "one. Acts in the wave-mode epoch loop right AFTER the "
                        "barrier ingest, before the buffer add and before the SGD "
                        "step schedule reads the tile size. Wave-mode only.")
    p.add_argument("--max-sgd-steps-per-epoch", type=int, default=0,
                   help="LF1 anti-runaway: hard upper bound on SGD steps per epoch "
                        "(per cycle). 0 (default) = no cap = today's behavior. When "
                        "> 0, the per-cycle step count (from --sgd-per-position, "
                        "--sgd-per-game, or --training-steps) is clamped to at most "
                        "this many steps. Decouples SGD work from inflow entirely: "
                        "structurally cannot run away even if the tile grows. "
                        "Applied AFTER the schedule computes steps_this_cycle, in "
                        "every mode (wave or in-process).")
    p.add_argument("--sgd-steps-per-epoch", type=int, default=0,
                   help="TRAINER MODE (fixed-step / fast-cadence learner). 0 "
                        "(default) = OFF = today's behavior (the "
                        "--sgd-per-position / --sgd-per-game / --training-steps "
                        "schedule, then the --max-sgd-steps-per-epoch cap). When "
                        "> 0, the trainer does EXACTLY this many SGD steps per "
                        "epoch, FIXED, independent of how many games arrived — it "
                        "does NOT scale with inflow and is structurally incapable "
                        "of the inflow-driven per-epoch runaway documented in "
                        "wiki/topics/perf-bench-vs-real-training-cost.md (the LF1 "
                        "20s->7min/epoch blow-up). Differs from "
                        "--max-sgd-steps-per-epoch (a min(computed, cap) clamp that "
                        "still scales below the cap): this PINS the count. In the "
                        "NON-wave worker-ingest path (--worker-input-dir set, "
                        "--wave-mode OFF) it ALSO switches ingest to NON-BLOCKING: "
                        "the epoch drains whatever new games are present this "
                        "instant (no --worker-min-games / --worker-min-positions "
                        "barrier) and trains immediately on the buffer, so the "
                        "trainer never idles waiting for generation. Net effect = "
                        "the async continuous-learner pattern (parallel actors + "
                        "fixed-rate learner) on a fast cadence. Tune N so an epoch "
                        "lands ~5s of SGD. Wave-mode is unaffected (the fixed-step "
                        "count still applies, but the wave barrier ingest is left "
                        "intact); intended for the non-wave async path.")
    p.add_argument("--worker-poll-sec", type=float, default=0.5,
                   help="Sleep this long between dir scans when waiting for workers.")
    p.add_argument("--device", type=str, default=None, help="torch device override (e.g. cpu, mps)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.set_defaults(wandb=False)
    p.add_argument("--wandb-project", type=str, default="gomoku")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"device = {device}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # V3 aux opponent-reply lever. The single weight flag gates the head
    # (constructed iff weight > 0), the buffer aux tensors, the self-play aux
    # target recording, and the aux loss term. 0.0 (default) => everything off
    # and byte-identical to the pre-aux path.
    aux_weight = float(args.aux_opponent_reply_weight)
    aux_on = aux_weight > 0.0
    if aux_on:
        print(f"aux opponent-reply head ENABLED (weight={aux_weight})")

    # V4 aux ownership lever (KataGo's biggest sample-efficiency win). The
    # single weight flag gates the ownership head (constructed iff weight > 0),
    # the buffer ownership tensors, the self-play ownership-target recording,
    # and the masked-MSE ownership loss term. 0.0 (default) => everything off
    # and byte-identical to the pre-ownership path. Independent of aux_on:
    # a cell may run either, both, or neither head.
    ownership_weight = float(args.aux_ownership_weight)
    ownership_on = ownership_weight > 0.0
    if ownership_on:
        print(f"aux ownership head ENABLED (weight={ownership_weight})")

    # Build / load model
    start_epoch = 0
    total_games = 0
    wandb_run_id = None
    if args.resume:
        model, payload = load_checkpoint(args.resume, device=device)
        start_epoch = int(payload.get("epoch", 0))
        total_games = int(payload.get("total_games", 0))
        wandb_run_id = payload.get("wandb_run_id")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        print(f"resumed from {args.resume} @ epoch {start_epoch}, total_games={total_games}")
    else:
        # --global-pool: None=off, bare flag (-1 sentinel)=latter-half (True),
        # K>=0 = trailing K blocks.
        gp = args.global_pool
        gp_arg = None if gp is None else (True if gp == -1 else gp)
        model = build_model(
            args.size,
            stem_padding=args.stem_padding,
            aux_opponent_reply=aux_on,
            aux_ownership=ownership_on,
            global_pool=gp_arg,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        payload = {}
        print(f"fresh {args.size} model: {n_params(model):,} params")

    # WL2 lever #1: EMA self-play weights. Build a slowly-tracking copy that
    # workers see, while `model` continues to train normally. On resume,
    # restore the EMA state if it was persisted; otherwise initialize ema=model.
    # The ema architecture mirrors `model.cfg` (not args.size) so resume from
    # a checkpoint with a different cfg still gets a matching ema.
    ema_model = None
    if args.ema_tau > 0.0:
        if not (0.0 < args.ema_tau < 1.0):
            raise SystemExit(f"--ema-tau must be in (0, 1); got {args.ema_tau}")
        from gomoku.model import GomokuNet
        ema_model = GomokuNet(model.cfg).to(device)
        if args.resume and "ema_model_state_dict" in payload:
            ema_model.load_state_dict(payload["ema_model_state_dict"])
            print(f"ema model restored from checkpoint (tau={args.ema_tau})")
        else:
            ema_model.load_state_dict(model.state_dict())
            print(f"ema model initialized from current weights (tau={args.ema_tau})")
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()

    # V3 lever: SWA tail-averaging of the self-play generator weights. When
    # --swa-window > 0 the workers see the flat average of the last K saved
    # checkpoints' state_dicts (of the live learner) instead of EMA/live
    # weights. SWA REPLACES the EMA-publish: they are alternative publish-path
    # smoothers, never composed. `swa_model` holds the recomputed average for
    # publication; `swa_window_dicts` is the ring buffer of recent CPU
    # state_dicts. The learner (`model`) and saved latest.pt/epochNNNN.pt are
    # untouched — only worker_weights.pt reflects the average.
    swa_model = None
    swa_window_dicts: deque | None = None
    if args.swa_window > 0:
        if args.ema_tau > 0.0:
            print(f"warning: --swa-window={args.swa_window} REPLACES the EMA "
                  f"publish path; --ema-tau={args.ema_tau} EMA copy is ignored "
                  f"for worker publication (SWA and EMA are not composed).")
        from gomoku.model import GomokuNet
        swa_model = GomokuNet(model.cfg).to(device)
        for p in swa_model.parameters():
            p.requires_grad_(False)
        swa_model.eval()
        swa_window_dicts = deque(maxlen=args.swa_window)
        print(f"SWA generator weights enabled (window={args.swa_window}): "
              f"workers see the flat mean of the last {args.swa_window} saved "
              f"checkpoints")

    validation_archive: dict | None = None
    if args.validation_archive_path:
        validation_archive = torch.load(
            args.validation_archive_path, map_location="cpu", weights_only=False,
        )
        print(
            f"validation archive loaded: {validation_archive['planes'].shape[0]} positions "
            f"from {args.validation_archive_path}"
        )

    buffer = ReplayBuffer(args.replay_buffer_size, device=device,
                          aux_opponent_reply=aux_on,
                          aux_ownership=ownership_on)
    if args.resume:
        # Optional: restore buffer if it was saved.
        payload_buf = payload.get("replay_buffer")
        if payload_buf is not None:
            buffer.load_state_dict(payload_buf)
            print(f"replay buffer restored: {buffer.size} examples")

    # wandb setup
    run = None
    if args.wandb:
        key = load_wandb_key_from_keychain()
        if not key:
            print("warning: --wandb set but no WANDB_API_KEY in keychain or env")
        else:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                id=wandb_run_id,
                resume="allow" if wandb_run_id else None,
                config=vars(args),
            )
            wandb_run_id = run.id

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Pre-parse eval baseline specs so misconfiguration fails before the first epoch.
    fast_specs = _parse_specs(args.eval_baselines)
    slow_specs = _parse_specs(args.eval_baselines_slow)
    # Build baseline pickers once (stateless across calls — RNG is passed in).
    fast_pickers = [(s, build_player(s)) for s in fast_specs]
    slow_pickers = [(s, build_player(s)) for s in slow_specs]
    eval_counter = 0

    # Pre-build the self-play opponent picker if requested. 'self' = vanilla
    # generate_games (model-vs-model). Any other valid spec routes through
    # generate_games_vs_baseline.
    opponent_picker = None
    opponent_label = "self"
    if args.opponent != "self":
        opp_spec = parse_spec(args.opponent)
        if opp_spec.kind == "model":
            raise SystemExit("--opponent model:... not supported for self-play; use 'self' "
                             "or one of random/heuristic/lookahead:depth=N")
        base_picker = build_player(opp_spec)
        opponent_label = opp_spec.label()
        if args.opponent_mix_random > 0.0:
            from gomoku.baselines import random_player as _rand
            p_random = float(args.opponent_mix_random)
            def _mixed_picker(state, rng, *, _base=base_picker, _p=p_random):
                if rng.random() < _p:
                    return _rand(state, rng)
                return _base(state, rng)
            opponent_picker = _mixed_picker
            opponent_label = f"{opponent_label}+{p_random:.0%}random"
        else:
            opponent_picker = base_picker
        print(f"self-play opponent: {opponent_label}")

    # If worker-input-dir is set, the trainer ingests batches from N worker
    # processes via the file-handoff protocol in `selfplay_worker.py` instead
    # of running generate_games in-process. The worker-weights-path is where
    # the trainer publishes its latest state_dict for workers to hot-reload.
    worker_input_dir = Path(args.worker_input_dir) if args.worker_input_dir else None
    if worker_input_dir is not None:
        worker_input_dir.mkdir(parents=True, exist_ok=True)
        if args.wave_mode:
            if args.wave_workers <= 0:
                raise SystemExit("--wave-mode requires --wave-workers > 0")
            if args.wave_games_per_worker <= 0:
                target_games = args.worker_min_games or args.games_per_epoch
                if target_games <= 0 or target_games % args.wave_workers != 0:
                    raise SystemExit("--wave-mode requires --wave-games-per-worker, "
                                     "or an even --worker-min-games/--games-per-epoch "
                                     "split across --wave-workers")
                args.wave_games_per_worker = target_games // args.wave_workers
            print(f"distributed wave mode: ingest from {worker_input_dir} "
                  f"(barrier {args.wave_games_per_worker} games x "
                  f"{args.wave_workers} workers per model version)")
        elif args.worker_min_positions > 0:
            print(f"distributed mode: ingest from {worker_input_dir} "
                  f"(target {args.worker_min_positions} positions/epoch — constant-age mode)")
        else:
            target_games = args.worker_min_games or args.games_per_epoch
            print(f"distributed mode: ingest from {worker_input_dir} "
                  f"(target {target_games} games/epoch — buffer age will vary with plies)")
    worker_weights_path = args.worker_weights_path
    if worker_weights_path:
        print(f"workers will read weights from {worker_weights_path}")

    # --- clean-shutdown support (time-capped research slices) ---
    # A SIGTERM/SIGINT, or crossing --max-wall-secs, sets the stop flag. We
    # finish the current epoch, then force a fully-resumable save and exit 0 so
    # `--resume latest.pt` continues without a cold buffer refill (the LF1 trap).
    # The signal handler only flips a flag — no I/O in the async context.
    #
    # The handler is installed HERE (before the ingest closures and the epoch
    # loop) so a SIGTERM that arrives while a blocking worker-ingest wait is
    # spinning is observed: those waits re-check `_stop["requested"]` each poll
    # and raise `_ShutdownRequested` to unwind to the epoch-boundary stop path.
    # Without that, a wait for a wave that never arrives (workers dead/orphaned)
    # would loop forever and the process would have to be SIGKILL'd.
    _stop = {"requested": False, "reason": ""}

    def _request_stop(signum, _frame):
        _stop["requested"] = True
        _stop["reason"] = f"signal {signal.Signals(signum).name}"

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    def _publish_worker_weights(epoch: int = 0) -> None:
        """Atomic-rename the model + cfg to `worker_weights_path` so workers
        (selfplay or eval) can mtime-poll and reload. Embeds the trainer's
        wandb run id + current epoch so eval_worker can log to the same run.
        Also bumps the buffer's current_weight_version so subsequent add() calls
        get the new tag.

        WL2 lever #1: when --ema-tau > 0, workers see the EMA weights, not
        the raw training weights. The 'brain that plays' is decoupled from
        the 'brain that learns'.

        V3 SWA lever: when --swa-window > 0, this REPLACES the EMA/live publish.
        The live learner's current state_dict is pushed into a ring buffer of
        the last K checkpoints, the flat average is recomputed into `swa_model`,
        and that average is what the workers see. The learner (`model`) and the
        saved latest.pt/epochNNNN.pt are untouched — only worker_weights.pt
        carries the averaged generator weights."""
        buffer.set_weight_version(epoch)
        if not worker_weights_path:
            return
        if swa_model is not None:
            # Snapshot the live learner onto CPU and push into the tail window,
            # then recompute the flat average for publication. (CPU clones keep
            # the ring cheap and detached from the live training tensors.)
            snapshot = {k: v.detach().to("cpu").clone()
                        for k, v in model.state_dict().items()}
            swa_window_dicts.append(snapshot)
            avg = average_state_dicts(list(swa_window_dicts))
            swa_model.load_state_dict(avg)
            publish_model = swa_model
        elif ema_model is not None:
            publish_model = ema_model
        else:
            publish_model = model
        tmp = worker_weights_path + ".tmp"
        save_checkpoint(tmp, publish_model, optimizer=None,
                        epoch=epoch, total_games=total_games,
                        wandb_run_id=wandb_run_id)
        os.replace(tmp, worker_weights_path)

    def _ingest_worker_batches(target_games: int = 0, target_positions: int = 0) -> list:
        """Block until either `target_games` games OR `target_positions` positions
        have arrived in the input dir (whichever quota is set; if both, the first
        one reached wins). Returns a flat list of GameRecord, files deleted.

        target_positions counts the expanded training examples per game record
        (post-D4-augmentation). Using positions keeps buffer turnover rate
        constant regardless of self-play game length — when games are short,
        more games are ingested per cycle to hit the target; when long, fewer.
        """
        if target_games <= 0 and target_positions <= 0:
            raise ValueError("must set target_games > 0 or target_positions > 0")
        out_records: list = []
        n_games = 0
        n_positions = 0
        while True:
            # Break out of an otherwise-unbounded wait if shutdown was
            # requested (SIGTERM/SIGINT). Raised before any work so a stop that
            # arrives while no games are flowing (e.g. workers dead/orphaned)
            # unwinds to the epoch-boundary clean-stop path instead of hanging.
            if _stop["requested"]:
                raise _ShutdownRequested(_stop["reason"])
            files = sorted(
                (p for p in worker_input_dir.glob("*.pt")),
                key=lambda p: p.stat().st_mtime,
            )
            if not files:
                time.sleep(args.worker_poll_sec)
                continue
            for f in files:
                try:
                    data = torch.load(f, map_location="cpu", weights_only=False)
                    recs = data["records"]
                    out_records.extend(recs)
                    n_games += int(data.get("n_games", len(recs)))
                    n_positions += int(data.get("n_examples", sum(len(r.examples) for r in recs)))
                finally:
                    try:
                        f.unlink()
                    except OSError:
                        pass
                # Stop when whichever quota is set (positions wins if both set).
                if target_positions > 0 and n_positions >= target_positions:
                    return out_records
                if target_positions <= 0 and target_games > 0 and n_games >= target_games:
                    return out_records

    def _drain_worker_batches_nonblocking() -> tuple[list, int, int]:
        """TRAINER MODE ingest: drain whatever flat worker payloads are present
        in worker_input_dir RIGHT NOW and return immediately — do NOT wait for
        any --worker-min-games / --worker-min-positions barrier. The trainer
        trains on the buffer every cycle regardless of inflow, so it never idles
        waiting for generation. Returns (records, n_games, n_positions).

        Mirrors _ingest_worker_batches' per-file load/delete, but without the
        blocking poll loop. Honors the stop flag before doing any work so a
        SIGTERM observed here unwinds cleanly (same as the blocking path).
        Non-wave only: it scans the top-level *.pt (the layout selfplay_worker
        writes when --wave-mode is off), never the v{version}/ tree."""
        if _stop["requested"]:
            raise _ShutdownRequested(_stop["reason"])
        out_records: list = []
        n_games = 0
        n_positions = 0
        files = sorted(
            (p for p in worker_input_dir.glob("*.pt")),
            key=lambda p: p.stat().st_mtime,
        )
        for f in files:
            consumed = False
            try:
                data = torch.load(f, map_location="cpu", weights_only=False)
                if not (isinstance(data, dict) and "records" in data):
                    # Not a self-play record payload (e.g. a worker_weights.pt
                    # someone parked in the input dir). Leave it in place and
                    # skip — never delete a file we did not ingest.
                    continue
                recs = data["records"]
                out_records.extend(recs)
                n_games += int(data.get("n_games", len(recs)))
                n_positions += int(
                    data.get("n_examples", sum(len(r.examples) for r in recs))
                )
                consumed = True
            finally:
                if consumed:
                    try:
                        f.unlink()
                    except OSError:
                        pass
        return out_records, n_games, n_positions

    def _normalize_wave_worker_id(raw: object) -> str:
        """Normalize path/payload worker ids to the run_sweep style: w0, w1, ..."""
        wid = str(raw)
        if wid.startswith("worker"):
            wid = wid[len("worker"):]
        if wid.isdigit():
            wid = f"w{wid}"
        return wid

    def _wave_version_from_dir(path: Path) -> int | None:
        name = path.name
        if not name.startswith("v"):
            return None
        try:
            return int(name[1:])
        except ValueError:
            return None

    def _wave_files_for_version(version: int) -> list[Path]:
        """Return finalized worker payloads for a model-version tile."""
        version_dir = worker_input_dir / f"v{version}"
        if not version_dir.exists():
            return []
        return sorted(
            (p for p in version_dir.rglob("*.pt") if p.is_file()),
            key=lambda p: (p.stat().st_mtime, str(p)),
        )

    def _load_wave_payload(path: Path) -> dict | None:
        """Load a wave payload, tolerating a just-renamed file briefly."""
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, EOFError, RuntimeError):
            return None

    def _delete_wave_payload(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            return
        # Trim now-empty worker/version dirs so stale scans stay cheap.
        for parent in (path.parent, path.parent.parent):
            if parent == worker_input_dir:
                break
            try:
                parent.rmdir()
            except OSError:
                break

    def _records_from_wave_payload(data: dict) -> list:
        recs = data.get("records", [])
        return list(recs)

    def _summarize_wave_payloads(payloads: dict[Path, dict]) -> tuple[dict[str, int], int, int]:
        """Return games-per-worker, total games, and total training examples."""
        counts: dict[str, int] = {}
        total_games = 0
        total_positions = 0
        for path, data in payloads.items():
            wid = _normalize_wave_worker_id(data.get("worker_id", path.parent.name))
            recs = _records_from_wave_payload(data)
            n_games = int(data.get("n_games", len(recs)))
            n_examples = int(data.get("n_examples", sum(len(r.examples) for r in recs)))
            counts[wid] = counts.get(wid, 0) + n_games
            total_games += n_games
            total_positions += n_examples
        return counts, total_games, total_positions

    def _drain_wave_version(version: int, payloads: dict[Path, dict] | None = None
                            ) -> tuple[list, list[tuple[int, list]], int, int]:
        """Ingest all currently visible files for one version and delete them.

        Returns flat records, version-tagged record groups, game count, and
        position count. The caller is responsible for adding groups to the
        replay buffer with the returned version tag.
        """
        payloads = dict(payloads or {})
        for path in _wave_files_for_version(version):
            if path not in payloads:
                data = _load_wave_payload(path)
                if data is not None:
                    payloads[path] = data
        out_records: list = []
        out_groups: list[tuple[int, list]] = []
        total_games = 0
        total_positions = 0
        for path, data in sorted(payloads.items(), key=lambda item: str(item[0])):
            recs = _records_from_wave_payload(data)
            if recs:
                out_records.extend(recs)
                out_groups.append((version, recs))
            total_games += int(data.get("n_games", len(recs)))
            total_positions += int(data.get("n_examples", sum(len(r.examples) for r in recs)))
            _delete_wave_payload(path)
        return out_records, out_groups, total_games, total_positions

    def _drain_stale_wave_versions(current_version: int) -> tuple[list, list[tuple[int, list]], dict]:
        """Drain late greedy extras from older version dirs without blocking."""
        if worker_input_dir is None or not worker_input_dir.exists():
            return [], [], {}
        out_records: list = []
        out_groups: list[tuple[int, list]] = []
        late_games = 0
        late_positions = 0
        versions: list[int] = []
        for path in worker_input_dir.iterdir():
            if not path.is_dir():
                continue
            version = _wave_version_from_dir(path)
            if version is not None and version < current_version:
                versions.append(version)
        for version in sorted(versions):
            recs, groups, n_games, n_positions = _drain_wave_version(version)
            out_records.extend(recs)
            out_groups.extend(groups)
            late_games += n_games
            late_positions += n_positions
        metrics = {
            "wave/late_games_ingested": late_games,
            "wave/late_positions_ingested": late_positions,
        }
        return out_records, out_groups, metrics

    def _cap_tile_games(records: list,
                        record_groups: list[tuple[int, list]],
                        cap: int) -> tuple[list, list[tuple[int, list]], dict]:
        """LF1 per-version tile cap (DROP semantics).

        Given the flat `records` and version-tagged `record_groups` produced by
        the wave ingest, keep at most `cap` games for EACH model version and
        DROP the rest. Returns rebuilt (records, record_groups, metrics) that are
        mutually consistent so the downstream buffer add and SGD-step schedule
        both see the capped tile.

        Why DROP, not DEFER: the wave drain (`_drain_wave_version`) has already
        deleted the on-disk payloads by the time we get here, so the excess games
        only exist in memory. Deferring them would mean re-attributing games
        generated under version v to a LATER version's tile, which violates the
        wave-mode invariant "every game in tile v was generated by version v"
        (wave-of-lockstep-design.md, invariant #1). Dropping the in-memory excess
        is the only choice that both bounds the tile AND preserves per-version
        purity. The cost is wasted generation — which is exactly the runaway work
        the cap exists to stop producing.

        Determinism: groups arrive in stable sorted order (sorted by file path in
        `_drain_wave_version`, sorted by version in `_drain_stale_wave_versions`).
        We keep the earliest games per version in that order, so the kept tile is
        a deterministic prefix.
        """
        if cap <= 0:
            return records, record_groups, {}
        kept_per_version: dict[int, int] = {}
        new_groups: list[tuple[int, list]] = []
        new_records: list = []
        dropped_games = 0
        dropped_positions = 0
        for version, recs in record_groups:
            already = kept_per_version.get(version, 0)
            room = max(0, cap - already)
            if room >= len(recs):
                keep = recs
            else:
                keep = recs[:room]
            drop = recs[len(keep):]
            if keep:
                new_groups.append((version, keep))
                new_records.extend(keep)
            kept_per_version[version] = already + len(keep)
            if drop:
                dropped_games += len(drop)
                dropped_positions += sum(len(r.examples) for r in drop)
        metrics = {
            "wave/tile_cap_dropped_games": dropped_games,
            "wave/tile_cap_dropped_positions": dropped_positions,
            "wave/tile_cap_active": 1 if dropped_games > 0 else 0,
        }
        return new_records, new_groups, metrics

    def _ingest_worker_wave(version: int) -> tuple[list, list[tuple[int, list]], dict]:
        """Block until the current version has enough games from every worker."""
        if args.wave_workers <= 0 or args.wave_games_per_worker <= 0:
            raise ValueError("wave mode requires wave worker count and games per worker")
        expected = [f"w{i}" for i in range(args.wave_workers)]
        payloads: dict[Path, dict] = {}
        first_done_at: float | None = None
        wait_start = time.time()
        counts = {wid: 0 for wid in expected}
        total_games = 0
        total_positions = 0

        while True:
            # Honor a shutdown request mid-barrier. The wave barrier waits for
            # `all(done)` — every worker delivering its quota for THIS version.
            # If the workers are dead/orphaned (the derby-kill case), that is
            # never satisfied, so without this check a SIGTERM/SIGINT would
            # loop here forever and force a SIGKILL. Raising unwinds to the
            # epoch-boundary clean-stop path (save resumable latest.pt, exit 0).
            if _stop["requested"]:
                raise _ShutdownRequested(_stop["reason"])
            for path in _wave_files_for_version(version):
                if path in payloads:
                    continue
                data = _load_wave_payload(path)
                if data is not None:
                    payloads[path] = data

            counts, total_games, total_positions = _summarize_wave_payloads(payloads)
            done = [counts.get(wid, 0) >= args.wave_games_per_worker for wid in expected]
            if any(done) and first_done_at is None:
                first_done_at = time.time()
            if all(done):
                break
            time.sleep(args.worker_poll_sec)

        wait_done = time.time()
        records, groups, drained_games, drained_positions = _drain_wave_version(version, payloads)
        total_games = drained_games
        total_positions = drained_positions
        per_worker_values = [counts.get(wid, 0) for wid in expected]
        metrics = {
            "wave/version": version,
            "wave/games_in_tile": total_games,
            "wave/positions_in_tile": total_positions,
            "wave/games_per_worker_min": min(per_worker_values) if per_worker_values else 0,
            "wave/games_per_worker_max": max(per_worker_values) if per_worker_values else 0,
            "wave/games_per_worker_mean": float(np.mean(per_worker_values)) if per_worker_values else 0.0,
            "wave/greedy_extras_count": max(0, total_games - args.wave_workers * args.wave_games_per_worker),
            "wave/wait_for_slowest_s": (
                wait_done - first_done_at if first_done_at is not None else 0.0
            ),
            "wave/total_s": wait_done - wait_start,
        }
        for wid in expected:
            metrics[f"wave/games_per_worker/{wid}"] = counts.get(wid, 0)
        return records, groups, metrics

    def _sweep_orphans() -> None:
        """Delete stale .tmp files and unrecognized .pt files in checkpoint dir
        + worker_input_dir. Anything older than --orphan-sweep-age-sec that
        doesn't match {latest.pt, worker_weights.pt, epochNNNN.pt} gets removed.
        Catches interrupted writes and leftover artifacts from killed runs."""
        cutoff = time.time() - args.orphan_sweep_age_sec
        protected = {"latest.pt", "latest.pt.tmp", "eval_results.jsonl"}
        if worker_weights_path:
            protected.add(os.path.basename(worker_weights_path))
        ck_dir = Path(args.checkpoint_dir)
        for f in ck_dir.iterdir() if ck_dir.exists() else []:
            if not f.is_file():
                continue
            if f.name in protected:
                continue
            # epochNNNN.pt is governed by --keep-last-n, leave to that path.
            if f.name.startswith("epoch") and f.suffix == ".pt":
                continue
            # Sweep .tmp leftovers and unexpected files older than cutoff.
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
        if worker_input_dir is not None and worker_input_dir.exists():
            for f in worker_input_dir.rglob("*"):
                if not f.is_file():
                    continue
                # In the records dir, the only valid live files are *.pt batches
                # ingested then deleted by _ingest_worker_batches. *.pt.tmp left
                # over from interrupted worker writes is the main thing to sweep.
                if f.suffix != ".tmp":
                    continue
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                except OSError:
                    pass

    # Eval JSONL: when running with --no-eval, an out-of-process gomoku.eval_worker
    # appends one JSON line per pass to <checkpoint_dir>/eval_results.jsonl.
    # The trainer is the SOLE wandb writer (two processes calling wandb.init for the
    # same run id silently drops one writer's logs), so we tail this file and merge
    # any new eval rows into the current cycle's log dict.
    eval_jsonl_path = Path(args.checkpoint_dir) / "eval_results.jsonl"
    eval_jsonl_offset = 0
    if eval_jsonl_path.exists():
        # Don't replay rows that predate this trainer run.
        eval_jsonl_offset = eval_jsonl_path.stat().st_size

    def _consume_eval_jsonl() -> dict:
        """Read new lines appended to eval_results.jsonl since last consumed.
        Returns the merged metric dict (later rows win on key collision)."""
        nonlocal eval_jsonl_offset
        if not eval_jsonl_path.exists():
            return {}
        try:
            cur_size = eval_jsonl_path.stat().st_size
        except OSError:
            return {}
        if cur_size <= eval_jsonl_offset:
            return {}
        merged: dict = {}
        try:
            with open(eval_jsonl_path, "r") as f:
                f.seek(eval_jsonl_offset)
                new = f.read()
                eval_jsonl_offset = f.tell()
        except OSError:
            return {}
        for line in new.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, v in rec.items():
                if k == "ts":
                    continue
                merged[k] = v
        return merged

    # Initial weight publication + orphan sweep, if applicable.
    # Pass start_epoch so workers see the correct model_version tag and write
    # to v{start_epoch}/. Without this, a --resume from a non-zero epoch
    # publishes worker_weights.pt tagged epoch=0, workers write to v0/, and
    # the wave-mode barrier (which waits on v{start_epoch}) stalls forever.
    # Bit us on WL4 resume from WL3.1 e1500.
    _publish_worker_weights(epoch=start_epoch)
    _sweep_orphans()

    # Stop flag + SIGTERM/SIGINT handler are installed earlier (right after
    # worker_weights_path is resolved) so the ingest waits can observe them.
    _wall_start = time.monotonic()

    def _save_resumable(epoch_num: int) -> None:
        """Force-write a fully-resumable latest.pt RIGHT NOW (ignores
        --save-buffer-every) plus a labeled snapshot, and republish worker
        weights. Called at clean shutdown so a capped slice always leaves a
        latest.pt that embeds the replay buffer."""
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_extra: dict = {"replay_buffer": buffer.state_dict()}
        if ema_model is not None:
            ckpt_extra["ema_model_state_dict"] = ema_model.state_dict()
        # Labeled snapshot (weights+optimizer; small) for lineage.
        snap = os.path.join(args.checkpoint_dir, f"epoch{epoch_num:04d}.pt")
        snap_extra = ({"ema_model_state_dict": ema_model.state_dict()}
                      if ema_model is not None else None)
        save_checkpoint(snap, model, optimizer, epoch=epoch_num,
                        total_games=total_games, wandb_run_id=wandb_run_id,
                        extra=snap_extra)
        # Resumable latest.pt with the buffer embedded.
        latest = os.path.join(args.checkpoint_dir, "latest.pt")
        latest_tmp = latest + ".tmp"
        save_checkpoint(latest_tmp, model, optimizer, epoch=epoch_num,
                        total_games=total_games, wandb_run_id=wandb_run_id,
                        extra=ckpt_extra)
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.replace(latest_tmp, latest)
        except OSError:
            pass
        _publish_worker_weights(epoch=epoch_num)

    # TRAINER MODE running totals (only meaningful when --sgd-steps-per-epoch>0).
    # cumulative_sgd_steps: SGD microbatch steps across all epochs this slice —
    # the x-axis for the "loss/val-CE vs CUMULATIVE steps" productive-vs-redundant
    # test (perf-bench-vs-real-training-cost.md follow-up #5).
    # cumulative_sgd_positions: steps * batch_size = positions CONSUMED by SGD.
    # cumulative_new_positions: positions INGESTED from workers.
    # reuse ratio = consumed / ingested (>1 re-using buffer; <1 fresh gen aging
    # out unused). These are extra log keys; they do not affect training.
    cumulative_sgd_steps = 0
    cumulative_sgd_positions = 0
    cumulative_new_positions = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()

        # --- self-play (in-process) or ingest (workers) ---
        gen_start = time.time()
        record_groups: list[tuple[int, list]] = []
        wave_metrics: dict = {}
        if worker_input_dir is not None:
            # The worker-ingest waits below can block indefinitely if no games
            # arrive (workers dead/orphaned). They re-check the stop flag each
            # poll and raise _ShutdownRequested on SIGTERM/SIGINT; catch it here
            # and route to the same clean-stop save path used at the epoch
            # boundary. This epoch did no training (no records ingested), so we
            # save at `epoch` (not epoch + 1) and exit 0 with a resumable
            # latest.pt. Identical to the max-wall-secs stop, just reached from
            # inside the wait instead of after it.
            try:
                if args.wave_mode:
                    stale_records, stale_groups, stale_metrics = _drain_stale_wave_versions(epoch)
                    wave_records, wave_groups, wave_metrics = _ingest_worker_wave(epoch)
                    records = stale_records + wave_records
                    record_groups = stale_groups + wave_groups
                    wave_metrics = {**stale_metrics, **wave_metrics}
                    # LF1 per-version tile cap (opt-in; 0 = uncapped = today's path).
                    # Acts AFTER the barrier ingest and BEFORE buffer add / SGD
                    # schedule, so the capped tile flows consistently to both.
                    if args.max_tile_games > 0:
                        records, record_groups, cap_metrics = _cap_tile_games(
                            records, record_groups, args.max_tile_games
                        )
                        wave_metrics = {**wave_metrics, **cap_metrics}
                elif args.sgd_steps_per_epoch > 0:
                    # TRAINER MODE (non-wave): NON-BLOCKING ingest. Drain
                    # whatever new games are present right now and proceed to
                    # SGD immediately — no --worker-min-games / -positions
                    # barrier, so the trainer never idles waiting for inflow.
                    # It cycles fast and publishes fresh weights each epoch
                    # while the workers keep generating in the background.
                    records, _tm_drained_games, _tm_drained_positions = (
                        _drain_worker_batches_nonblocking()
                    )
                elif args.worker_min_positions > 0:
                    records = _ingest_worker_batches(target_positions=args.worker_min_positions)
                else:
                    records = _ingest_worker_batches(
                        target_games=args.worker_min_games or args.games_per_epoch,
                    )
            except _ShutdownRequested:
                print(f"[stop] {_stop['reason']} during wave ingest at epoch "
                      f"{epoch}; saving resumable checkpoint (buffer embedded), "
                      f"exiting", flush=True)
                _save_resumable(epoch)
                break
        elif opponent_picker is None:
            evaluator = make_torch_evaluator(model, device)
            records = generate_games(
                args.games_per_epoch,
                evaluator,
                n_simulations=args.n_simulations,
                c_puct=args.c_puct,
                c_puct_base=args.c_puct_base,
                temperature_moves=args.temperature_moves,
                temperature_final=args.temperature_final,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
                rng=rng,
                wave_size=args.wave_size,
                random_opening_moves=args.random_opening_moves,
                record_aux=aux_on,
                record_ownership=ownership_on,
            )
        else:
            evaluator = make_torch_evaluator(model, device)
            records = generate_games_vs_baseline(
                args.games_per_epoch,
                evaluator,
                opponent_picker,
                n_simulations=args.n_simulations,
                c_puct=args.c_puct,
                c_puct_base=args.c_puct_base,
                temperature_moves=args.temperature_moves,
                temperature_final=args.temperature_final,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
                rng=rng,
                wave_size=args.wave_size,
                model_first_frac=args.model_first_frac,
                random_opening_moves=args.random_opening_moves,
            )
        gen_time = time.time() - gen_start

        n_examples_new = sum(len(r.examples) for r in records)
        if record_groups:
            for version, group_records in record_groups:
                buffer.set_weight_version(version)
                for r in group_records:
                    buffer.add(r.examples)
            buffer.set_weight_version(epoch)
        else:
            for r in records:
                buffer.add(r.examples)

        plies_mean = float(np.mean([r.plies for r in records])) if records else 0.0
        # Plies-distribution percentiles for the new records — quick check on
        # game-length shape going INTO the buffer this cycle.
        plies_array = np.array([r.plies for r in records]) if records else np.array([0])
        plies_p10 = float(np.percentile(plies_array, 10))
        plies_p50 = float(np.percentile(plies_array, 50))
        plies_p90 = float(np.percentile(plies_array, 90))
        outcomes = [r.outcome for r in records]
        # In self-play, outcome is +1 if first-mover (black) won.
        # In vs-baseline, outcome is +1 if model won.
        wins_pos = sum(1 for o in outcomes if o > 0)
        wins_neg = sum(1 for o in outcomes if o < 0)
        draws = sum(1 for o in outcomes if o == 0)

        # --- training ---
        # Decide how many SGD steps this cycle. Priority (highest first):
        #   0. --sgd-steps-per-epoch N (TRAINER MODE): FIXED N, independent of
        #      inflow — structurally incapable of the inflow-driven runaway.
        #   1. --sgd-per-position * positions_ingested  (constant-age recipe)
        #   2. --sgd-per-game * games_ingested          (legacy K-based)
        #   3. --training-steps                          (static fallback)
        n_games_this_cycle = len(records)
        n_positions_this_cycle = sum(len(r.examples) for r in records)
        if args.sgd_steps_per_epoch > 0:
            # TRAINER MODE: pin the per-epoch step count. Does NOT read the
            # inflow (tile size / positions), so the train phase cannot grow
            # with generation rate. Bypasses the schedule AND the
            # --max-sgd-steps-per-epoch cap (a min(N, cap) clamp would be a
            # no-op anyway against a fixed N).
            steps_this_cycle = args.sgd_steps_per_epoch
        elif args.sgd_per_position is not None and n_positions_this_cycle > 0:
            steps_this_cycle = max(
                args.min_training_steps,
                round(args.sgd_per_position * n_positions_this_cycle),
            )
        elif args.sgd_per_game is not None and n_games_this_cycle > 0:
            steps_this_cycle = max(
                args.min_training_steps,
                round(args.sgd_per_game * n_games_this_cycle),
            )
        else:
            steps_this_cycle = args.training_steps

        # LF1 hard per-epoch SGD step cap (opt-in; 0 = no cap = today's path).
        # Decouples SGD work from inflow entirely: clamps the computed step count
        # from ANY schedule above, so the train phase cannot run away even if the
        # tile grows. Note this clamps ABOVE the --min-training-steps floor; if a
        # cap below min_training_steps were ever set, the cap wins (it is the
        # explicit upper bound the operator asked for).
        steps_before_cap = steps_this_cycle
        # TRAINER MODE pins the count, so the cap is bypassed (a fixed N is
        # already inflow-independent; a min(N, cap) clamp would just be a
        # confusing second knob). Cap applies only in the schedule path.
        if args.sgd_steps_per_epoch <= 0 and args.max_sgd_steps_per_epoch > 0:
            steps_this_cycle = min(steps_this_cycle, args.max_sgd_steps_per_epoch)
        sgd_step_cap_active = (
            args.sgd_steps_per_epoch <= 0
            and args.max_sgd_steps_per_epoch > 0
            and steps_before_cap > steps_this_cycle
        )

        train_metrics_acc: dict[str, list[float]] = {}
        train_start = time.time()
        # WL2 lever #4: grad accumulation. steps_this_cycle is the number of
        # forward+backward passes (microbatches); optimizer.step() happens
        # every --grad-accum-steps microbatches. A final partial window at
        # end-of-cycle still steps (the partial accumulator's gradient is
        # under-scaled by partial_size/grad_accum_steps, which acts as a
        # smaller effective LR for that one step — preferable to discarding
        # the backward work we already paid for).
        accum_n = max(1, int(args.grad_accum_steps))
        optimizer_steps_this_cycle = 0
        microbatch_steps_this_cycle = 0  # actual forward+backward passes run
        if buffer.size >= args.batch_size:
            for i in range(steps_this_cycle):
                microbatch_steps_this_cycle += 1
                aux_pi = aux_mask = ownership = ownership_mask = None
                if aux_on or ownership_on:
                    sampled = buffer.sample(
                        args.batch_size,
                        return_aux=aux_on,
                        return_ownership=ownership_on,
                    )
                    planes, pi, z, side, ply = sampled[:5]
                    extra = list(sampled[5:])
                    if aux_on:
                        aux_pi, aux_mask = extra.pop(0), extra.pop(0)
                    if ownership_on:
                        ownership, ownership_mask = extra.pop(0), extra.pop(0)
                else:
                    planes, pi, z, side, ply = buffer.sample(args.batch_size)
                is_first_in_window = (i % accum_n == 0)
                is_last_in_window = ((i + 1) % accum_n == 0) or (i + 1 == steps_this_cycle)
                m = train_step(
                    model, optimizer, planes, pi, z,
                    value_weight=args.value_weight,
                    l2_weight=args.l2,
                    accum_scale=1.0 / accum_n,
                    do_optimizer_step=is_last_in_window,
                    zero_grad=is_first_in_window,
                    side=side,
                    ply=ply,
                    aux_pi=aux_pi,
                    aux_mask=aux_mask,
                    aux_weight=aux_weight,
                    ownership=ownership,
                    ownership_mask=ownership_mask,
                    ownership_weight=ownership_weight,
                )
                for k, v in m.items():
                    train_metrics_acc.setdefault(k, []).append(v)
                if is_last_in_window:
                    optimizer_steps_this_cycle += 1
                    # WL2 lever #1: EMA tracks the trained model per optimizer
                    # step (BYOL-style per-step cadence, not per-wave). Default
                    # picked for ablation consistency with the design doc; can
                    # be revisited per-wave if per-step is too cheap to see in
                    # the ema_l2_distance trace.
                    if ema_model is not None:
                        ema_update(ema_model, model, args.ema_tau)
        train_time = time.time() - train_start
        train_metrics = {k: float(np.mean(v)) for k, v in train_metrics_acc.items()}

        # TRAINER MODE idle-guard: if this epoch did NO SGD (buffer not yet
        # warm, or no games have arrived from the still-spinning-up workers),
        # the non-blocking epoch would otherwise busy-spin and peg a CPU. A
        # short poll-sleep keeps the cadence from free-running before the
        # buffer fills, WITHOUT ever sleeping when there is real SGD work to do
        # (so it never wastes the GPU). Re-checks the stop flag so SIGTERM
        # stays prompt during warm-up.
        if (args.sgd_steps_per_epoch > 0 and microbatch_steps_this_cycle == 0
                and not _stop["requested"]):
            time.sleep(args.worker_poll_sec)

        total_games += n_games_this_cycle if worker_input_dir is not None else args.games_per_epoch

        # Log key names depend on opponent: self-play tracks black/white, vs-baseline
        # tracks model wins/losses.
        if opponent_picker is None:
            wins_key, losses_key = "selfplay/black_wins", "selfplay/white_wins"
        else:
            wins_key, losses_key = "selfplay/model_wins", "selfplay/model_losses"
        log = {
            "epoch": epoch + 1,
            "total_games": total_games,
            "buffer_size": buffer.size,
            "selfplay/new_examples": n_examples_new,
            "selfplay/new_games": n_games_this_cycle,
            "selfplay/plies_mean": plies_mean,
            "selfplay/plies_p10": plies_p10,
            "selfplay/plies_p50": plies_p50,
            "selfplay/plies_p90": plies_p90,
            wins_key: wins_pos,
            losses_key: wins_neg,
            "selfplay/draws": draws,
            "time/gen_s": gen_time,
            "time/train_s": train_time,
            "train/steps_this_cycle": steps_this_cycle,
            "train/steps_before_cap": steps_before_cap,
            "train/sgd_step_cap_active": 1 if sgd_step_cap_active else 0,
            "train/optimizer_steps_this_cycle": optimizer_steps_this_cycle,
            "train/actual_sgd_per_game": (
                steps_this_cycle / n_games_this_cycle if n_games_this_cycle else 0.0
            ),
            "train/actual_sgd_per_position": (
                steps_this_cycle / n_positions_this_cycle if n_positions_this_cycle else 0.0
            ),
            "train/positions_this_cycle": n_positions_this_cycle,
            **wave_metrics,
            **train_metrics,
            # Buffer-shape snapshot: distribution over n_stones buckets + z mix.
            # Computed every cycle so wandb shows shape evolution over time.
            **buffer.shape_stats(),
        }
        # TRAINER MODE metrics (only when --sgd-steps-per-epoch>0; otherwise
        # NO new keys are emitted, so wave-mode / schedule cells log exactly as
        # before). These measure the whole point of the mode:
        #   - cumulative_sgd_steps: x-axis for "loss vs CUMULATIVE steps" — the
        #     productive-vs-redundant-SGD test (perf page follow-up #5).
        #   - sample_reuse_ratio: positions CONSUMED by SGD (steps*batch) /
        #     positions INGESTED. >1 = re-using buffer positions; <1 = fresh
        #     generation aging out unused.
        if args.sgd_steps_per_epoch > 0:
            cumulative_sgd_steps += microbatch_steps_this_cycle
            cumulative_sgd_positions += microbatch_steps_this_cycle * args.batch_size
            cumulative_new_positions += n_positions_this_cycle
            log["train/cumulative_sgd_steps"] = cumulative_sgd_steps
            log["train/cumulative_sgd_positions"] = cumulative_sgd_positions
            log["train/cumulative_new_positions"] = cumulative_new_positions
            log["train/sgd_positions_this_cycle"] = (
                microbatch_steps_this_cycle * args.batch_size
            )
            log["train/sample_reuse_ratio"] = (
                cumulative_sgd_positions / cumulative_new_positions
                if cumulative_new_positions > 0 else 0.0
            )
            log["train/sample_reuse_ratio_epoch"] = (
                (microbatch_steps_this_cycle * args.batch_size) / n_positions_this_cycle
                if n_positions_this_cycle > 0 else 0.0
            )

        # WL2 lever #1: diagnose runaway divergence between learn-brain and
        # play-brain. Logged only when EMA is enabled.
        if ema_model is not None:
            log["train/ema_l2_distance"] = ema_l2_distance(ema_model, model)

        # --- forward any new eval_worker results into wandb ---
        ext_eval = _consume_eval_jsonl()
        if ext_eval:
            log.update(ext_eval)

        # --- in-trainer eval (only when --no-eval not passed) ---
        if validation_archive is not None and (epoch + 1) % args.eval_every == 0:
            va_start = time.time()
            log.update(_score_validation_archive(model, validation_archive, device))
            log["time/val_archive_s"] = time.time() - va_start

        if args.eval_in_trainer and (epoch + 1) % args.eval_every == 0:
            eval_counter += 1
            eval_start = time.time()
            eval_evaluator = make_torch_evaluator(model, device)
            model_picker = mcts_picker(eval_evaluator,
                                       n_simulations=args.eval_sims,
                                       c_puct=args.c_puct)

            run_slow = bool(slow_pickers) and (eval_counter % args.eval_slow_every == 0)
            batches = [("fast", fast_pickers, args.eval_baseline_games)]
            if run_slow:
                batches.append(("slow", slow_pickers, args.eval_slow_games))

            for batch_label, pickers, n_games in batches:
                for spec_idx, (spec, baseline_picker) in enumerate(pickers):
                    m_start = time.time()
                    res = play_match_pickers(
                        model_picker, baseline_picker,
                        n_games=n_games,
                        seed=args.seed + epoch * 1000 + spec_idx,
                    )
                    key = _baseline_log_key(spec)
                    log[f"eval/{key}_winrate"] = res.win_rate
                    log[f"eval/{key}_wins"] = res.wins
                    log[f"eval/{key}_losses"] = res.losses
                    log[f"eval/{key}_draws"] = res.draws
                    log[f"time/eval_{key}_s"] = time.time() - m_start

            log["time/eval_s"] = time.time() - eval_start

        epoch_time = time.time() - epoch_start
        log["time/epoch_s"] = epoch_time

        # Print
        msg = (
            f"epoch {epoch + 1}/{start_epoch + args.epochs} "
            f"games={total_games} buf={buffer.size} "
            f"new={n_games_this_cycle} steps={steps_this_cycle} "
            f"pl={train_metrics.get('loss/policy', float('nan')):.3f} "
            f"vl={train_metrics.get('loss/value', float('nan')):.3f} "
            f"plies={plies_mean:.1f} "
            f"age={log.get('buffer/age_p50', 0):.0f} "
            f"({epoch_time:.1f}s: gen={gen_time:.1f}s train={train_time:.1f}s)"
        )
        if args.wave_mode and wave_metrics:
            msg += (
                f" wave[v{wave_metrics.get('wave/version', epoch)} "
                f"tile={wave_metrics.get('wave/games_in_tile', 0)} "
                f"workers={wave_metrics.get('wave/games_per_worker_min', 0)}-"
                f"{wave_metrics.get('wave/games_per_worker_max', 0)} "
                f"extras={wave_metrics.get('wave/greedy_extras_count', 0)}]"
            )
        if args.sgd_steps_per_epoch > 0:
            msg += (
                f" trainer[cumsteps={log.get('train/cumulative_sgd_steps', 0)} "
                f"reuse={log.get('train/sample_reuse_ratio', 0.0):.2f}]"
            )
        wr_bits = [
            f"{key[3:]}={log[f'eval/{key}_winrate']:.0%}"
            for spec in fast_specs + slow_specs
            for key in [_baseline_log_key(spec)]
            if f"eval/{key}_winrate" in log
        ]
        # Also surface any eval/* keys forwarded from eval_worker that aren't in our fast/slow lists.
        seen = {f"eval/{_baseline_log_key(s)}_winrate" for s in (fast_specs + slow_specs)}
        for k, v in log.items():
            if k.startswith("eval/") and k.endswith("_winrate") and k not in seen:
                wr_bits.append(f"{k[len('eval/'):-len('_winrate')]}={v:.0%}")
        if wr_bits:
            msg += " wr[" + " ".join(wr_bits) + "]"
        # eval_worker forwards a maximum-likelihood model_elo per cycle when
        # at least one baseline matches an anchor in gomoku.rating.ANCHOR_ELOS.
        # Surface it in the epoch print line for quick reading.
        if "eval/model_elo" in log:
            msg += f" elo={log['eval/model_elo']:.0f}"
        print(msg, flush=True)

        if run is not None:
            run.log(log)

        # --- save ---
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch{epoch + 1:04d}.pt")
            # Intermediate snapshots are weights+optimizer only (small, ~4 MB).
            # The replay buffer (~1.4 GB) only goes in latest.pt to support
            # resume — old "epochNNNN.pt" files don't need it.
            # WL2: persist EMA state alongside so resume restores the play-brain too.
            ckpt_extra: dict | None = None
            if ema_model is not None:
                ckpt_extra = {"ema_model_state_dict": ema_model.state_dict()}
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                epoch=epoch + 1,
                total_games=total_games,
                wandb_run_id=wandb_run_id,
                extra=ckpt_extra,
            )
            # Write a separate latest.pt that includes the replay buffer.
            # Throttled by --save-buffer-every since this is the ~1.4 GB write.
            if (epoch + 1) % args.save_buffer_every == 0:
                latest = os.path.join(args.checkpoint_dir, "latest.pt")
                latest_tmp = latest + ".tmp"
                latest_extra: dict = {"replay_buffer": buffer.state_dict()}
                if ema_model is not None:
                    latest_extra["ema_model_state_dict"] = ema_model.state_dict()
                save_checkpoint(
                    latest_tmp,
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    total_games=total_games,
                    wandb_run_id=wandb_run_id,
                    extra=latest_extra,
                )
                try:
                    if os.path.islink(latest) or os.path.exists(latest):
                        os.remove(latest)
                    os.replace(latest_tmp, latest)
                except OSError:
                    pass
            # Publish a lean weights-only file for any worker processes
            # (self-play workers AND the eval_worker).
            _publish_worker_weights(epoch=epoch + 1)
            # Auto-prune old intermediate checkpoints to keep disk bounded.
            if args.keep_last_n > 0:
                ck_files = sorted(
                    Path(args.checkpoint_dir).glob("epoch*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
                to_drop = ck_files[: -args.keep_last_n] if len(ck_files) > args.keep_last_n else []
                for f in to_drop:
                    try:
                        f.unlink()
                    except OSError:
                        pass
            # Sweep .tmp leftovers and unknown files past the age threshold.
            _sweep_orphans()

        # --- clean stop: SIGTERM/SIGINT received, or wall-time cap reached ---
        # Checked at the epoch boundary so a partial epoch is never interrupted.
        # Either path force-saves a resumable latest.pt (buffer embedded) so the
        # next slice resumes without a cold refill, then exits 0 (wandb flushes
        # on normal exit).
        if not _stop["requested"] and args.max_wall_secs > 0:
            if (time.monotonic() - _wall_start) >= args.max_wall_secs:
                _stop["requested"] = True
                _stop["reason"] = f"--max-wall-secs {args.max_wall_secs:g}"
        if _stop["requested"]:
            print(f"[stop] {_stop['reason']} after epoch {epoch + 1}; "
                  f"saving resumable checkpoint (buffer embedded), exiting",
                  flush=True)
            _save_resumable(epoch + 1)
            break


if __name__ == "__main__":
    main()
