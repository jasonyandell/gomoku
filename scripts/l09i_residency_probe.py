"""L09i smoke B: ANE residency micro-probe for the static-shape Core ML export.

Two modes:

  loop  -- export (if needed) a static-shape model, load it CPU_AND_NE, warm up
           ~20 predicts (ANE lazy-load), then run a tight predict loop for the
           requested duration. Prints its own PID first so a driver can `sample`
           it. This is the subprocess that gets sampled.

  drive -- spawn the loop subprocess, wait for warmup, run `sample <pid> 3
           -mayDie` and `ps -M <pid>` while it loops, then grep the call graph
           for BNNS (CPU) vs ANE indicators and print a verdict.

CPU-only / diagnostic: no GPU self-play, no training.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gomoku.game import BOARD_SIZE, N_INPUT_PLANES  # noqa: E402

INPUT_NAME = "planes"


def _build_static_model(out: Path, batch: int, *, single_fixed: bool = False) -> None:
    import torch

    from gomoku.coreml_evaluator import export_model_to_coreml
    from gomoku.model import build_model, fuse_model_for_inference

    torch.manual_seed(0)
    model = fuse_model_for_inference(build_model("small"))
    model.eval()
    if single_fixed:
        # Single concrete batch dim (matches scout --batch-shape fixed).
        export_model_to_coreml(model, out, batch_sizes=[batch])
    else:
        export_model_to_coreml(model, out, max_batch_size=batch)


def run_loop(args: argparse.Namespace) -> None:
    import coremltools as ct

    model_path = Path(args.model_path)
    if not model_path.exists():
        _build_static_model(model_path, args.batch, single_fixed=args.single_fixed)

    mlmodel = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    x = np.zeros((args.batch, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    print(f"LOOP_PID={os.getpid()}", flush=True)
    for _ in range(args.warmup):
        mlmodel.predict({INPUT_NAME: x})
    print("WARMUP_DONE", flush=True)

    calls = 0
    deadline = time.perf_counter() + args.duration_s
    while time.perf_counter() < deadline:
        mlmodel.predict({INPUT_NAME: x})
        calls += 1
    print(f"LOOP_CALLS={calls}", flush=True)


# Decisive signal: which engine actually *executes* the conv/matmul layers in
# the hot path. ANE-resident runs dispatch through the ANE runtime engine;
# CPU/BNNS runs dispatch through Espresso::BNNSEngine (or BnnsCpuInferenceOp).
# Generic framework names (plain "Espresso", "ANEServices" load lines,
# "ANECompiler") are NOT proof of residency, so they are excluded.
ANE_INDICATORS = (
    "ANERuntimeEngine",
    "H11ANEServicesThread",
    "ANEServicesThread",
    "_ANEModel",
    "ANEProgram",
    "aneSubmit",
    "ANEDeviceController",
    "_ANEDeviceExecute",
)
CPU_INDICATORS = (
    "BnnsCpuInferenceOperation",
    "Espresso::BNNSEngine",
    "BNNSEngine::convolution_kernel",
    "BNNSEngine::inner_product_kernel",
)


def _grep(text: str, needles) -> list[str]:
    hits = []
    for line in text.splitlines():
        for n in needles:
            if n in line:
                hits.append(line.strip())
                break
    return hits


def drive(args: argparse.Namespace) -> None:
    if args.out_dir:
        tmp = Path(args.out_dir)
        tmp.mkdir(parents=True, exist_ok=True)
    else:
        tmp = Path(tempfile.mkdtemp(prefix="l09i-resid-"))
    kind = "single_fixed" if args.single_fixed else "enumerated"
    model_path = tmp / f"small_{kind}.mlpackage"
    print(f"[drive] building {kind} static model at {model_path} ...", flush=True)
    _build_static_model(model_path, args.batch, single_fixed=args.single_fixed)

    cmd = [
        sys.executable,
        __file__,
        "loop",
        "--model-path",
        str(model_path),
        "--batch",
        str(args.batch),
        "--warmup",
        str(args.warmup),
        "--duration-s",
        str(args.duration_s),
    ]
    if args.single_fixed:
        cmd.append("--single-fixed")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    pid = None
    warmed = False
    assert proc.stdout is not None
    t0 = time.time()
    while time.time() - t0 < 60:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip()
        if line.startswith("LOOP_PID="):
            pid = int(line.split("=", 1)[1])
            print(f"[drive] loop pid={pid}", flush=True)
        elif line == "WARMUP_DONE":
            warmed = True
            print("[drive] warmup done; sampling now", flush=True)
            break
        elif line.startswith("E5RT") or "scikit-learn" in line or "Torch version" in line:
            continue

    if pid is None or not warmed:
        print("[drive] FAILED to reach warmup; child output follows:", flush=True)
        print(proc.stdout.read())
        proc.wait()
        return

    # Sample while the loop runs.
    sample_txt = ""
    try:
        sample_txt = subprocess.run(
            ["sample", str(pid), "3", "-mayDie"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as exc:  # pragma: no cover
        sample_txt = f"<sample failed: {exc}>"

    ps_txt = ""
    try:
        ps_out = subprocess.run(["ps", "-M", str(pid)], capture_output=True, text=True, timeout=10).stdout
        ps_txt = "\n".join(l for l in ps_out.splitlines() if re.search(r"ANEServices", l, re.I))
    except Exception as exc:  # pragma: no cover
        ps_txt = f"<ps failed: {exc}>"

    # Let the loop finish.
    rest = proc.stdout.read()
    proc.wait()

    sample_path = tmp / "sample.txt"
    sample_path.write_text(sample_txt)

    ane_hits = _grep(sample_txt, ANE_INDICATORS)
    cpu_hits = _grep(sample_txt, CPU_INDICATORS)

    print("\n========== RESIDENCY VERDICT ==========")
    print(f"sample saved to: {sample_path}")
    print(f"child tail: {rest.strip().splitlines()[-3:] if rest.strip() else '(none)'}")
    print(f"\nANE indicator lines ({len(ane_hits)}):")
    for l in dict.fromkeys(ane_hits):
        print("  ", l)
    print(f"\nCPU/BNNS indicator lines ({len(cpu_hits)}):")
    for l in dict.fromkeys(cpu_hits):
        print("  ", l)
    print("\nps -M ANEServices threads:")
    print(ps_txt or "  (none)")

    if ane_hits and not cpu_hits:
        verdict = "ANE RESIDENCY RESTORED (ANE indicators present, no BNNS in hot path)"
    elif ane_hits and cpu_hits:
        verdict = "MIXED (both ANE and BNNS indicators present)"
    elif cpu_hits:
        verdict = "CPU/BNNS (ANE FAILED -- BnnsCpuInferenceOperation in hot path)"
    else:
        verdict = "INCONCLUSIVE (neither indicator found; inspect sample.txt)"
    print(f"\nVERDICT: {verdict}")
    print("=======================================")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    lp = sub.add_parser("loop")
    lp.add_argument("--model-path", required=True)
    lp.add_argument("--batch", type=int, default=64)
    lp.add_argument("--warmup", type=int, default=20)
    lp.add_argument("--duration-s", type=float, default=10.0)
    lp.add_argument("--single-fixed", action="store_true")

    dp = sub.add_parser("drive")
    dp.add_argument("--batch", type=int, default=64)
    dp.add_argument("--warmup", type=int, default=20)
    dp.add_argument("--duration-s", type=float, default=12.0)
    dp.add_argument("--out-dir", default=None, help="Persist model + sample.txt here.")
    dp.add_argument("--single-fixed", action="store_true")

    args = p.parse_args()
    if args.mode == "loop":
        run_loop(args)
    else:
        drive(args)


if __name__ == "__main__":
    main()
