"""L09i diagnostic: diff the ML-program ops of the two Core ML export paths.

We export the same gomoku model through:

  A. gomoku.coreml_evaluator.export_model_to_coreml  (the production lab path,
     proven CPU/BNNS-resident by L09e').
  B. scripts.coreml_ane_residency_scout.export_coreml_model (the "scout" path,
     proven ANE-resident on 2026-05-22).

Then it walks each .mlpackage's MIL program, builds an op-type histogram, and
prints the diff so we can name the ANE-hostile op(s).

CPU-only. No self-play, no training. Conversion may use CPU/GPU; that is fine.

Usage:
    python scripts/l09i_op_diff.py            # tiny config, fp16, default outdir
    python scripts/l09i_op_diff.py --size small
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import coremltools as ct  # noqa: E402

from gomoku.coreml_evaluator import export_model_to_coreml  # noqa: E402
from gomoku.game import BOARD_SIZE, N_INPUT_PLANES  # noqa: E402
from gomoku.model import GomokuNet, ModelConfig, SIZE_PRESETS, fuse_model_for_inference  # noqa: E402

# Import the scout's own exporter + plan plumbing so path B is byte-for-byte the
# real scout code, not a reimplementation.
import scripts.coreml_ane_residency_scout as scout  # noqa: E402


def build_lab_model(size: str, *, fuse: bool) -> GomokuNet:
    """Lab path uses the real ModelConfig preset (default stem_padding=3)."""
    cfg = SIZE_PRESETS[size]
    model = GomokuNet(cfg).eval()
    if fuse:
        model = fuse_model_for_inference(model)
    return model


def build_scout_model(size: str, *, fuse: bool) -> tuple[GomokuNet, scout.ModelPlan]:
    """Scout path: build via the scout's own make_plan/build_model.

    The scout's gomoku branch hardcodes ModelConfig(stem_padding=1, policy_filters=2,
    value_filters=1) and sizes n_filters/n_blocks/value_hidden from CLI flags.
    We feed it the same numbers as the chosen preset so the only differences are
    the export-path differences, not capacity.
    """
    preset = SIZE_PRESETS[size]
    ns = argparse.Namespace(
        channels=None,
        height=None,
        width=None,
        hidden=preset.value_hidden,
        depth=2,
        filters=preset.n_filters,
        blocks=preset.n_blocks,
        output_width=scout.N_ACTIONS,
    )
    plan = scout.make_plan("gomoku", ns)
    model = scout.build_model(plan).eval()
    if fuse:
        model = fuse_model_for_inference(model)
    return model, plan


def op_histogram(mlpackage_path: Path) -> Counter:
    """Walk the MIL program of an .mlpackage and count op types."""
    spec = ct.models.MLModel(str(mlpackage_path)).get_spec()
    counts: Counter = Counter()
    program = spec.mlProgram
    for func in program.functions.values():
        block = func.block_specializations[func.opset]
        for op in block.operations:
            counts[op.type] += 1
    return counts


def describe_io(mlpackage_path: Path) -> dict[str, Any]:
    spec = ct.models.MLModel(str(mlpackage_path)).get_spec()
    desc = spec.description
    return {
        "inputs": [(i.name, i.type.WhichOneof("Type")) for i in desc.input],
        "outputs": [(o.name, o.type.WhichOneof("Type")) for o in desc.output],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="tiny", choices=list(SIZE_PRESETS))
    parser.add_argument(
        "--fuse",
        action="store_true",
        default=True,
        help="Fuse conv+bn for both paths (matches lab export_checkpoint fuse=True default).",
    )
    parser.add_argument("--no-fuse", dest="fuse", action="store_false")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("sweep_logs/l09i_op_diff"),
    )
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"== L09i op diff ==  size={args.size} fuse={args.fuse} ct={ct.__version__}")
    print(f"input shape = (B, {N_INPUT_PLANES}, {BOARD_SIZE}, {BOARD_SIZE})\n")

    # --- Path A: lab production export ---
    lab_model = build_lab_model(args.size, fuse=args.fuse)
    lab_path = args.outdir / f"lab_{args.size}.mlpackage"
    export_model_to_coreml(lab_model, lab_path, max_batch_size=512, compute_precision="FLOAT16")
    lab_hist = op_histogram(lab_path)
    lab_io = describe_io(lab_path)
    print(f"[A] lab  -> {lab_path}")
    print(f"    IO: {lab_io}")

    # --- Path B: scout export ---
    scout_model, plan = build_scout_model(args.size, fuse=args.fuse)
    scout_path = args.outdir / f"scout_{args.size}.mlpackage"
    scout.export_coreml_model(
        scout_model,
        plan,
        scout_path,
        batch_size=128,
        max_batch_size=512,
        batch_shape="range",
        compute_precision="FLOAT16",
        input_dtype="float32",
    )
    scout_hist = op_histogram(scout_path)
    scout_io = describe_io(scout_path)
    print(f"[B] scout -> {scout_path}")
    print(f"    IO: {scout_io}\n")

    # --- Histograms ---
    all_ops = sorted(set(lab_hist) | set(scout_hist))
    print(f"{'op_type':<28} {'lab(A)':>8} {'scout(B)':>9} {'delta(A-B)':>11}")
    print("-" * 60)
    for op in all_ops:
        a = lab_hist.get(op, 0)
        b = scout_hist.get(op, 0)
        d = a - b
        flag = "  <-- DIFF" if d != 0 else ""
        print(f"{op:<28} {a:>8} {b:>9} {d:>11}{flag}")

    print("\n== ops only in / heavier in LAB (A) — ANE-hostility suspects ==")
    suspects = {op: lab_hist.get(op, 0) - scout_hist.get(op, 0) for op in all_ops}
    for op, d in sorted(suspects.items(), key=lambda kv: -kv[1]):
        if d > 0:
            print(f"  {op}: +{d}")

    print("\n== ops only in / heavier in SCOUT (B) ==")
    for op, d in sorted(suspects.items(), key=lambda kv: kv[1]):
        if d < 0:
            print(f"  {op}: {d}")


if __name__ == "__main__":
    main()
