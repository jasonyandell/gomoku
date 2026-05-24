"""L09i probe B: does fixed-batch vs RangeDim-flexible-batch change the MIL graph?

The op-type histogram diff (l09i_op_diff.py) showed the lab and scout exports
produce IDENTICAL op graphs when both use RangeDim. So op *type* is not the
ANE-hostility lever. The remaining candidate the scout exposes and the lab path
hard-codes-away is the batch geometry: scout supports `--batch-shape fixed`,
the lab path always uses ct.RangeDim(1..max). Dynamic/symbolic batch dims are
on hollance's ANE-hostile list.

This probe exports the SAME model both ways and prints:
  - the MIL op histogram for each,
  - the declared input shape in the spec (static int vs RangeDim),
so we can see whether RangeDim injects shape-handling ops or a symbolic dim.

CPU-only. No training/self-play.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import coremltools as ct  # noqa: E402

from gomoku.game import BOARD_SIZE, N_INPUT_PLANES  # noqa: E402
from gomoku.model import GomokuNet, SIZE_PRESETS, fuse_model_for_inference  # noqa: E402

INPUT_NAME = "planes"


def op_histogram(path: Path) -> Counter:
    spec = ct.models.MLModel(str(path)).get_spec()
    counts: Counter = Counter()
    for func in spec.mlProgram.functions.values():
        block = func.block_specializations[func.opset]
        for op in block.operations:
            counts[op.type] += 1
    return counts


def input_shape_desc(path: Path) -> str:
    spec = ct.models.MLModel(str(path)).get_spec()
    inp = spec.description.input[0]
    mt = inp.type.multiArrayType
    dims = []
    # static shape
    static = list(mt.shape)
    # flexible shape ranges
    ranges = []
    if mt.HasField("shapeRange"):
        for sr in mt.shapeRange.sizeRanges:
            ranges.append((sr.lowerBound, sr.upperBound))
    return f"static_shape={static} shapeRange={ranges or 'none'}"


def export(model: GomokuNet, path: Path, *, fixed: bool, batch: int = 128, maxb: int = 512) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_b = batch if fixed else 1
    dummy = torch.zeros(dummy_b, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    traced = torch.jit.trace(model.cpu(), dummy)
    batch_dim = batch if fixed else ct.RangeDim(lower_bound=1, upper_bound=maxb)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=INPUT_NAME,
                              shape=(batch_dim, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE),
                              dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.save(str(path))


def main() -> None:
    outdir = Path("sweep_logs/l09i_op_diff")
    cfg = SIZE_PRESETS["tiny"]
    fixed_path = outdir / "fixed_tiny.mlpackage"
    range_path = outdir / "range_tiny.mlpackage"

    export(fuse_model_for_inference(GomokuNet(cfg).eval()), fixed_path, fixed=True)
    export(fuse_model_for_inference(GomokuNet(cfg).eval()), range_path, fixed=False)

    fh, rh = op_histogram(fixed_path), op_histogram(range_path)
    print("== fixed vs range batch ==")
    print(f"fixed input: {input_shape_desc(fixed_path)}")
    print(f"range input: {input_shape_desc(range_path)}\n")
    all_ops = sorted(set(fh) | set(rh))
    print(f"{'op_type':<24}{'fixed':>8}{'range':>8}{'delta':>8}")
    print("-" * 48)
    for op in all_ops:
        f, r = fh.get(op, 0), rh.get(op, 0)
        flag = "  <-- DIFF" if f != r else ""
        print(f"{op:<24}{f:>8}{r:>8}{f - r:>8}{flag}")


if __name__ == "__main__":
    main()
