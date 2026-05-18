"""Export a trained gomoku checkpoint to ONNX for browser inference.

Loads checkpoints/latest.pt by default (or --checkpoint PATH), exports the
PyTorch model with a dynamic batch axis, and writes the .onnx file plus a
sibling .meta.json with provenance info.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint checkpoints/epoch0050.pt
    python scripts/export_onnx.py --out app/public/model.onnx
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import torch

from gomoku.model import load_checkpoint


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "latest.pt"
DEFAULT_OUT = REPO_ROOT / "app" / "public" / "model.onnx"


def export_onnx(checkpoint_path: Path, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, payload = load_checkpoint(str(checkpoint_path), device="cpu")
    model.eval()

    dummy = torch.zeros(1, 3, 9, 9, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # The dynamo-based exporter (torch >= 2.5) writes weights to a sibling
    # `<name>.onnx.data` file by default. For browser inference we want one
    # self-contained ONNX file, so re-save with weights inlined and delete
    # the external blob.
    import onnx  # noqa: WPS433
    m = onnx.load(str(out_path))
    onnx.save(m, str(out_path), save_as_external_data=False)
    external_blob = out_path.with_name(out_path.name + ".data")
    if external_blob.exists():
        external_blob.unlink()

    # Numerical fidelity check
    try:
        import onnxruntime as ort  # noqa: WPS433

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        x = rng.random((4, 3, 9, 9), dtype=np.float32)
        with torch.no_grad():
            p_t, v_t = model(torch.from_numpy(x))
        ort_out = sess.run(None, {"input": x})
        p_o, v_o = ort_out[0], ort_out[1]
        max_p = float(np.max(np.abs(p_t.numpy() - p_o)))
        max_v = float(np.max(np.abs(v_t.numpy() - v_o)))
        print(f"  ONNX fidelity check: max |dp|={max_p:.2e}  max |dv|={max_v:.2e}")
    except ImportError:
        print("  (onnxruntime not installed; skipping fidelity check)")

    cfg = payload.get("model_config", {})
    meta = {
        "epoch": int(payload.get("epoch", 0)),
        "total_games": int(payload.get("total_games", 0)),
        "n_filters": int(cfg.get("n_filters", 0)),
        "n_blocks": int(cfg.get("n_blocks", 0)),
        "n_input_planes": int(cfg.get("n_input_planes", 3)),
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_source": os.path.basename(str(checkpoint_path)),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    size_bytes = out_path.stat().st_size
    print(f"Wrote {out_path} ({size_bytes/1_000_000:.2f} MB)")
    print(f"Wrote {meta_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    out = Path(args.out).resolve()
    export_onnx(checkpoint, out)


if __name__ == "__main__":
    main()
