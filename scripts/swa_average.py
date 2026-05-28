"""SWA (Stochastic Weight Averaging) — post-training peak averaging.

Averages parameters of the last K saved `peak*.pt` checkpoints for a lane into
a single `peak_swa.pt`. The averaged checkpoint is then a *free new contestant*
the GPU runner can H2H against the un-averaged peak.

Source: Izmailov et al. 2018, "Averaging Weights Leads to Wider Optima and
Better Generalization" (https://arxiv.org/abs/1803.05407). True SWA averages
*saved checkpoints* POST-training; this is distinct from `--ema-tau 0.99`
which is a *training-time* EMA inside the gen workers.

Format note: our checkpoint is the standard `gomoku.model.save_checkpoint`
payload — a dict with `model_state_dict`, `model_config`, `epoch`,
`total_games`, optional `optimizer_state_dict`, `wandb_run_id`. We average only
`model_state_dict` (params + buffers); the optimizer state is NOT averaged
(averaging an optimizer's running moments across runs is not meaningful), nor
is the replay buffer (it's data, not weights). Non-weight metadata is
preserved from the most recent peak so the output remains a normal,
trainer/eval-loadable checkpoint.

BatchNorm caveat: our model contains `nn.BatchNorm2d` (running_mean,
running_var). These tensors live in `state_dict()` and are averaged
element-wise alongside the conv weights. The standard SWA recipe recommends
re-estimating BN stats with a fresh forward pass over training data; we DON'T
do that here (this is a CPU-only offline tool, no data, no GPU). For our
purposes — H2H eval immediately downstream — the uniform mean of the per-peak
running stats is a reasonable approximation; if the averaged ckpt
underperforms its individual sources, the BN-recalibration variant is a
follow-up lever (out of scope here per the bead).

Usage:
    python scripts/swa_average.py \\
        --lane-dir sweep_runs/<lane>/_peaks/<idea>/ \\
        --k 10 \\
        --output sweep_runs/<lane>/_peaks/<idea>/peak_swa.pt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# Peak discovery
# ---------------------------------------------------------------------------

def discover_peaks(lane_dir: Path) -> list[Path]:
    """Return all `peak*.pt` files in `lane_dir`, sorted oldest -> newest by
    (epoch metadata if present, else mtime). Excludes any prior SWA outputs
    (`peak_swa*.pt`) so re-running the tool doesn't average its own outputs in.
    """
    if not lane_dir.is_dir():
        raise FileNotFoundError(f"lane-dir not found or not a directory: {lane_dir}")
    candidates: list[Path] = []
    for p in lane_dir.glob("peak*.pt"):
        if p.name.startswith("peak_swa"):
            continue
        if not p.is_file():
            continue
        candidates.append(p)
    if not candidates:
        return []

    # Sort by embedded epoch if available, else mtime. We sort lexicographic
    # ties consistently to make the tool deterministic.
    def _key(path: Path) -> tuple[int, float, str]:
        # Try epoch from metadata; fall back to mtime.
        epoch = -1
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "epoch" in payload:
                ep = payload["epoch"]
                if isinstance(ep, int):
                    epoch = ep
        except Exception:
            epoch = -1
        mtime = path.stat().st_mtime
        return (epoch, mtime, path.name)

    candidates.sort(key=_key)
    return candidates


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def _validate_shapes(state_dicts: Sequence[dict], names: Sequence[str]) -> None:
    """Loud error if any two state_dicts disagree on key set or tensor shapes."""
    if not state_dicts:
        return
    ref_keys = set(state_dicts[0].keys())
    ref_shapes = {k: tuple(v.shape) for k, v in state_dicts[0].items() if torch.is_tensor(v)}
    for sd, name in zip(state_dicts[1:], names[1:]):
        keys = set(sd.keys())
        if keys != ref_keys:
            extra = keys - ref_keys
            missing = ref_keys - keys
            raise ValueError(
                f"SWA refuses: state_dict key set differs between {names[0]!r} and {name!r}. "
                f"Missing={sorted(missing)[:5]} Extra={sorted(extra)[:5]}"
            )
        for k, ref_shape in ref_shapes.items():
            v = sd[k]
            if not torch.is_tensor(v):
                continue
            shape = tuple(v.shape)
            if shape != ref_shape:
                raise ValueError(
                    f"SWA refuses: shape mismatch on key {k!r}: "
                    f"{names[0]!r} has {ref_shape} vs {name!r} has {shape}"
                )


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------

def average_state_dicts(state_dicts: Sequence[dict]) -> dict:
    """Element-wise uniform mean of a sequence of state_dicts.

    For each key:
      - if the value is a floating tensor, compute the mean across the stack.
      - if the value is an integer tensor (e.g. BN's `num_batches_tracked`),
        take the value from the LAST state_dict (most recent peak). Integer
        tensors are counters, not parameters; averaging them to non-integers
        would be wrong.
      - if the value is a non-tensor (rare in a model state_dict), take the
        LAST value.
    """
    if not state_dicts:
        raise ValueError("average_state_dicts: empty input")
    out: dict = {}
    last = state_dicts[-1]
    for key in state_dicts[0].keys():
        ref = state_dicts[0][key]
        if not torch.is_tensor(ref):
            out[key] = last[key]
            continue
        if ref.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                         torch.uint8, torch.bool):
            # Integer counters / masks — take latest, don't average.
            out[key] = last[key].clone()
            continue
        # Floating tensor: element-wise mean. Cast to float32 for the
        # accumulation, then back to the original dtype to preserve format.
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for sd in state_dicts:
            acc.add_(sd[key].to(torch.float32))
        acc.div_(float(len(state_dicts)))
        out[key] = acc.to(ref.dtype)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Average the last K peak*.pt checkpoints in a lane "
                    "into a single peak_swa.pt (uniform mean, CPU-only).",
    )
    parser.add_argument(
        "--lane-dir", type=Path, required=True,
        help="Directory holding peak*.pt files (e.g. sweep_runs/<lane>/_peaks/<idea>/).",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Average the last K peaks (by epoch metadata, falling back to mtime). "
             "If fewer than K are present, average all available.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: <lane-dir>/peak_swa.pt).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the peaks that would be averaged, then exit without writing.",
    )
    args = parser.parse_args(argv)

    if args.k <= 0:
        print(f"swa_average: --k must be >= 1 (got {args.k})", file=sys.stderr)
        return 2

    lane_dir: Path = args.lane_dir
    output: Path = args.output if args.output is not None else lane_dir / "peak_swa.pt"

    peaks = discover_peaks(lane_dir)
    if not peaks:
        print(f"swa_average: no peak*.pt files found in {lane_dir}", file=sys.stderr)
        return 1

    if len(peaks) < args.k:
        print(
            f"swa_average: requested k={args.k} but only {len(peaks)} peak(s) "
            f"available; averaging all {len(peaks)}.",
            file=sys.stderr,
        )
    selected = peaks[-args.k:]
    src_names = [p.name for p in selected]
    print(f"swa_average: averaging {len(selected)} peak(s) from {lane_dir}:")
    for p in selected:
        print(f"  - {p.name}")

    if args.dry_run:
        print(f"swa_average: --dry-run, would write to {output}")
        return 0

    # Load each peak (CPU-only) and pull out the model_state_dict.
    payloads = []
    state_dicts = []
    for p in selected:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            print(
                f"swa_average: refusing — {p.name} is not a standard checkpoint "
                f"(no 'model_state_dict' key).",
                file=sys.stderr,
            )
            return 3
        payloads.append(payload)
        state_dicts.append(payload["model_state_dict"])

    _validate_shapes(state_dicts, src_names)

    avg_sd = average_state_dicts(state_dicts)

    # Build the output payload: start from the LAST peak's metadata (config,
    # epoch, total_games, wandb_run_id), substitute the averaged state_dict,
    # and DROP optimizer_state_dict (averaging optimizer running moments is
    # not meaningful).
    last_payload = payloads[-1]
    out_payload: dict = {}
    for k, v in last_payload.items():
        if k == "model_state_dict":
            continue
        if k == "optimizer_state_dict":
            continue
        out_payload[k] = v
    out_payload["model_state_dict"] = avg_sd
    out_payload["swa"] = {
        "k": len(selected),
        "method": "uniform_mean",
        "source_peaks": src_names,
        "source_epochs": [p.get("epoch") for p in payloads],
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "lane_dir": str(lane_dir),
        "note": (
            "Uniform mean of model_state_dict (params + buffers, incl. BN "
            "running stats). Optimizer state dropped. No BN recalibration "
            "performed; if downstream eval underperforms, recalibrate BN "
            "stats with a fresh forward pass over training data."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(out_payload, tmp)
    tmp.replace(output)
    print(f"swa_average: wrote {output}")
    print(f"swa_average: meta = {json.dumps({k: out_payload['swa'][k] for k in ('k','method','source_peaks','source_epochs')})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
