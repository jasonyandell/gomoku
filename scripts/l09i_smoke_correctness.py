"""L09i smoke A: correctness of the static-shape (EnumeratedShapes) Core ML export.

Builds a tiny gomoku model, exports it via the patched export_model_to_coreml
(now static EnumeratedShapes instead of RangeDim), then runs the CoreMLEvaluator
on batches of size 1, 7, and 64. Asserts:
  * outputs finite, correctly shaped (one policy vector + one scalar per leaf),
  * priors softmax to ~1, values in [-1, 1],
  * Core ML outputs reasonably match the torch model on the same inputs.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gomoku.coreml_evaluator import (  # noqa: E402
    CoreMLEvaluator,
    export_model_to_coreml,
)
from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES  # noqa: E402
from gomoku.model import build_model, fuse_model_for_inference  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model = fuse_model_for_inference(build_model("tiny"))
    model.eval()

    tmp = Path(tempfile.mkdtemp(prefix="l09i-smoke-"))
    out = tmp / "tiny_static.mlpackage"
    export_model_to_coreml(model, out, max_batch_size=64)

    evaluator = CoreMLEvaluator(model_path=str(out), compute_units="CPU_AND_NE")
    print("discovered static_batch_sizes:", evaluator.static_batch_sizes)
    print("evaluator.max_batch_size:", evaluator.max_batch_size)
    assert evaluator.static_batch_sizes, "expected static (non-symbolic) batch sizes"

    for batch in (1, 7, 64):
        x = rng.standard_normal(
            (batch, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
        ).astype(np.float32)

        priors, values = evaluator.evaluate_planes(x)

        assert priors.shape == (batch, N_ACTIONS), (batch, priors.shape)
        assert values.shape == (batch,), (batch, values.shape)
        assert np.all(np.isfinite(priors)), f"non-finite priors at batch={batch}"
        assert np.all(np.isfinite(values)), f"non-finite values at batch={batch}"

        soft = np.exp(priors - priors.max(axis=1, keepdims=True))
        soft /= soft.sum(axis=1, keepdims=True)
        sums = soft.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-4), f"softmax sums {sums}"
        assert np.all(values >= -1.001) and np.all(values <= 1.001), values

        # Compare against torch on the same inputs.
        with torch.no_grad():
            t_logits, t_values = model(torch.from_numpy(x))
        t_logits = t_logits.numpy()
        t_values = t_values.numpy().reshape(batch)

        pol_mae = float(np.abs(priors - t_logits).mean())
        val_mae = float(np.abs(values - t_values).mean())
        # fp16 Core ML vs fp32 torch: expect small but nonzero deltas.
        assert pol_mae < 5e-2, f"policy MAE too high at batch={batch}: {pol_mae}"
        assert val_mae < 5e-2, f"value MAE too high at batch={batch}: {val_mae}"
        print(
            f"batch={batch:>3}: OK  policy_MAE={pol_mae:.5f} value_MAE={val_mae:.5f} "
            f"val_range=[{values.min():.3f},{values.max():.3f}]"
        )

    print("SMOKE A PASS: static-shape export is correct on batches 1, 7, 64.")


if __name__ == "__main__":
    main()
