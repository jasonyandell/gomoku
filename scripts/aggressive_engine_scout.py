"""Aggressive Apple Silicon engine scout for gomoku eval/training overlap.

This is a bounded "hit the wall once" harness. It does not try to be a clean
microbenchmark; it tries to answer whether the M5 Max split is promising:

- PyTorch eval on MPS / CPU
- Core ML eval on CPU_ONLY / CPU_AND_NE, FP16 and optionally INT8 weights
- trainer-like PyTorch MPS steps alone vs under eval pressure

The output is JSON so wiki receipts can cite the exact numbers.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gomoku.coreml_evaluator import (  # noqa: E402
    CoreMLUnavailableError,
    export_model_to_coreml,
    make_coreml_evaluator,
    require_coremltools,
)
from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES  # noqa: E402
from gomoku.mcts import make_torch_evaluator  # noqa: E402
from gomoku.model import (  # noqa: E402
    GomokuNet,
    ModelConfig,
    build_model,
    fuse_model_for_inference,
    load_checkpoint,
    n_params,
)
from gomoku.util import pick_device  # noqa: E402


EvalFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class Lane:
    name: str
    evaluator: EvalFn | None = None
    setup: dict | None = None
    skipped: str | None = None
    error: str | None = None


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def sync_device(device: torch.device) -> None:
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def summarize_seconds(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)
    idx90 = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return {
        "min_ms": ordered[0] * 1000.0,
        "median_ms": statistics.median(ordered) * 1000.0,
        "p90_ms": ordered[idx90] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
    }


def build_base_model(args: argparse.Namespace) -> tuple[GomokuNet, str, dict]:
    if args.checkpoint:
        model, payload = load_checkpoint(args.checkpoint, device="cpu")
        source = f"checkpoint:{args.checkpoint}"
        metadata = {
            "checkpoint": args.checkpoint,
            "epoch": payload.get("epoch"),
            "total_games": payload.get("total_games"),
            "model_config": asdict(model.cfg),
        }
        return model.cpu().eval(), source, metadata

    torch.manual_seed(args.seed)
    model = build_model(args.size, stem_padding=args.stem_padding).cpu().eval()
    source = f"fresh:{args.size}:stem_padding={args.stem_padding}"
    metadata = {"model_config": asdict(model.cfg)}
    return model, source, metadata


def clone_for_eval(base: GomokuNet, device: torch.device, *, fuse: bool) -> GomokuNet:
    model = copy.deepcopy(base).to(device).eval()
    if fuse:
        model = fuse_model_for_inference(model)
    return model


def make_torch_lane(
    base: GomokuNet,
    *,
    name: str,
    device: torch.device,
    fuse: bool,
) -> Lane:
    try:
        start = time.perf_counter()
        model = clone_for_eval(base, device, fuse=fuse)
        evaluator = make_torch_evaluator(model, device).evaluate_planes  # type: ignore[attr-defined]
        return Lane(
            name=name,
            evaluator=evaluator,
            setup={
                "device": str(device),
                "fused_eval": fuse,
                "materialize_s": time.perf_counter() - start,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive scout path
        return Lane(name=name, error=f"{type(exc).__name__}: {exc}")


def quantize_coreml_weights(mlpackage: Path, output_path: Path) -> Path:
    ct = require_coremltools()
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    model = ct.models.MLModel(str(mlpackage), compute_units=ct.ComputeUnit.CPU_ONLY)
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric")
    )
    quantized = linear_quantize_weights(model, config=config)
    quantized.save(str(output_path))
    return output_path


def build_coreml_lanes(
    base: GomokuNet,
    args: argparse.Namespace,
    tmpdir: Path,
) -> list[Lane]:
    lanes: list[Lane] = []
    try:
        require_coremltools()
    except CoreMLUnavailableError as exc:
        for precision in args.coreml_precisions:
            for units in args.coreml_compute_units:
                lanes.append(Lane(name=f"coreml_{precision}_{units}", skipped=str(exc)))
        return lanes

    fp16_path = tmpdir / "gomoku_scout_fp16.mlpackage"
    export_start = time.perf_counter()
    try:
        export_model_to_coreml(
            copy.deepcopy(base).eval(),
            fp16_path,
            max_batch_size=max(max(args.batches), args.overlap_batch),
            compute_precision="FLOAT16",
        )
        fp16_export_s = time.perf_counter() - export_start
    except Exception as exc:
        for precision in args.coreml_precisions:
            for units in args.coreml_compute_units:
                lanes.append(
                    Lane(
                        name=f"coreml_{precision}_{units}",
                        error=f"export failed: {type(exc).__name__}: {exc}",
                    )
                )
        return lanes

    model_paths: dict[str, tuple[Path, dict]] = {
        "fp16": (fp16_path, {"export_s": fp16_export_s, "quantize_s": 0.0})
    }
    if "int8" in args.coreml_precisions:
        int8_path = tmpdir / "gomoku_scout_int8.mlpackage"
        quant_start = time.perf_counter()
        try:
            quantize_coreml_weights(fp16_path, int8_path)
            model_paths["int8"] = (
                int8_path,
                {
                    "export_s": fp16_export_s,
                    "quantize_s": time.perf_counter() - quant_start,
                },
            )
        except Exception as exc:
            for units in args.coreml_compute_units:
                lanes.append(
                    Lane(
                        name=f"coreml_int8_{units}",
                        error=f"quantize failed: {type(exc).__name__}: {exc}",
                    )
                )

    for precision in args.coreml_precisions:
        if precision not in model_paths:
            continue
        model_path, setup_base = model_paths[precision]
        for units in args.coreml_compute_units:
            lane_name = f"coreml_{precision}_{units}"
            load_start = time.perf_counter()
            try:
                evaluator = make_coreml_evaluator(
                    model_path,
                    compute_units=units,
                    max_batch_size=max(max(args.batches), args.overlap_batch),
                )
                setup = {
                    **setup_base,
                    "load_s": time.perf_counter() - load_start,
                    "compute_units": units,
                    "model_path": str(model_path),
                    "retained_after_run": False,
                }
                lanes.append(Lane(name=lane_name, evaluator=evaluator.evaluate_planes, setup=setup))
            except Exception as exc:
                lanes.append(Lane(name=lane_name, error=f"load failed: {type(exc).__name__}: {exc}"))
    return lanes


def make_planes(seed: int, batches: list[int]) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        batch: rng.normal(
            size=(batch, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
        ).astype(np.float32)
        for batch in batches
    }


def bench_eval_lanes(
    lanes: list[Lane],
    planes_by_batch: dict[int, np.ndarray],
    *,
    warmup: int,
    repeats: int,
) -> dict:
    results: dict[str, dict] = {}
    for lane in lanes:
        lane_result: dict = {"setup": lane.setup or {}}
        if lane.skipped:
            lane_result["skipped"] = lane.skipped
            results[lane.name] = lane_result
            continue
        if lane.error:
            lane_result["error"] = lane.error
            results[lane.name] = lane_result
            continue
        assert lane.evaluator is not None
        batch_results = {}
        first_predict_s: float | None = None
        for batch, planes in planes_by_batch.items():
            try:
                for i in range(warmup + repeats):
                    start = time.perf_counter()
                    priors, values = lane.evaluator(planes)
                    elapsed = time.perf_counter() - start
                    if first_predict_s is None:
                        first_predict_s = elapsed
                    if i >= warmup:
                        if priors.shape != (batch, N_ACTIONS) or values.shape != (batch,):
                            raise ValueError(
                                f"bad output shapes priors={priors.shape} values={values.shape}"
                            )
                        batch_results[str(batch)] = batch_results.get(str(batch), []) + [elapsed]
            except Exception as exc:
                batch_results[str(batch)] = {"error": f"{type(exc).__name__}: {exc}"}
        lane_result["first_predict_s"] = first_predict_s
        lane_result["batches"] = {}
        for batch_s, samples_or_error in batch_results.items():
            if isinstance(samples_or_error, dict):
                lane_result["batches"][batch_s] = samples_or_error
                continue
            samples = samples_or_error
            batch = int(batch_s)
            summary = summarize_seconds(samples)
            median_s = statistics.median(samples)
            summary["positions_per_s_median"] = batch / median_s if median_s > 0 else 0.0
            lane_result["batches"][batch_s] = summary
        results[lane.name] = lane_result
    return results


def make_train_stepper(
    base: GomokuNet,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> Callable[[], None]:
    torch.manual_seed(seed)
    model = copy.deepcopy(base).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(batch_size, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE, device=device)
    pi = torch.randn(batch_size, N_ACTIONS, device=device).softmax(dim=1)
    z = torch.randn(batch_size, device=device).tanh()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(x)
        policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(values, z)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

    return step


def measure_train_stepper(
    stepper: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    steps: int,
) -> dict[str, float]:
    for _ in range(warmup):
        stepper()
    sync_device(device)
    samples = []
    for _ in range(steps):
        start = time.perf_counter()
        stepper()
        sync_device(device)
        samples.append(time.perf_counter() - start)
    summary = summarize_seconds(samples)
    summary["steps"] = float(steps)
    return summary


def run_eval_pressure_process(
    lane_name: str,
    lane_setup: dict,
    cfg_dict: dict,
    state_dict: dict,
    fuse_eval: bool,
    planes: np.ndarray,
    pressure_warmup: int,
    pressure_max_calls: int,
    pressure_sleep_s: float,
    ready: mp.Event,
    stop: mp.Event,
    queue: mp.Queue,
) -> None:
    """Run eval pressure in a child process.

    MPS eval in a Python thread can trip Metal command-buffer assertions. The
    production worker shape is multi-process, so the scout uses a subprocess
    for pressure lanes as well.
    """
    calls = 0
    positions = 0
    errors: list[str] = []
    stopped_by = "stop_event"
    start = time.perf_counter()

    try:
        if lane_name.startswith("torch_"):
            device = torch.device(lane_setup.get("device", "cpu"))
            model = GomokuNet(ModelConfig(**cfg_dict))
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            if fuse_eval:
                model = fuse_model_for_inference(model)
            evaluator = make_torch_evaluator(model, device).evaluate_planes  # type: ignore[attr-defined]
        elif lane_name.startswith("coreml_"):
            evaluator = make_coreml_evaluator(
                lane_setup["model_path"],
                compute_units=lane_setup["compute_units"],
                max_batch_size=int(planes.shape[0]),
            ).evaluate_planes
        else:
            raise ValueError(f"unknown pressure lane {lane_name!r}")

        for _ in range(max(0, pressure_warmup)):
            evaluator(planes)
        ready.set()

        while not stop.is_set():
            evaluator(planes)
            calls += 1
            positions += int(planes.shape[0])
            if calls >= pressure_max_calls:
                stopped_by = "max_calls"
                stop.set()
                break
            if pressure_sleep_s > 0:
                time.sleep(pressure_sleep_s)
    except Exception as exc:  # pragma: no cover - defensive scout path
        errors.append(f"{type(exc).__name__}: {exc}")
        ready.set()

    elapsed = max(time.perf_counter() - start, 1e-9)
    queue.put(
        {
            "eval_calls": calls,
            "eval_positions": positions,
            "eval_positions_per_s": positions / elapsed,
            "eval_elapsed_s": elapsed,
            "max_calls": pressure_max_calls,
            "sleep_s": pressure_sleep_s,
            "stopped_by": stopped_by,
            "errors": errors,
        }
    )


def bench_overlap(
    base: GomokuNet,
    lanes: list[Lane],
    args: argparse.Namespace,
    planes_by_batch: dict[int, np.ndarray],
) -> dict:
    device = pick_device(args.train_device)
    if device.type != "mps":
        return {"skipped": f"train_device={device}; overlap scout is intended for MPS"}

    prewarm_stepper = make_train_stepper(
        base,
        device=device,
        batch_size=args.train_batch_size,
        seed=args.seed + 100,
    )
    prewarm = measure_train_stepper(
        prewarm_stepper,
        device=device,
        warmup=args.train_warmup,
        steps=args.train_prewarm_steps,
    )

    baseline_stepper = make_train_stepper(
        base,
        device=device,
        batch_size=args.train_batch_size,
        seed=args.seed + 1,
    )
    baseline = measure_train_stepper(
        baseline_stepper,
        device=device,
        warmup=args.train_warmup,
        steps=args.train_steps,
    )
    baseline_median = baseline.get("median_ms", 0.0)

    pressure_batch = min(args.overlap_batch, max(args.batches))
    planes = planes_by_batch.get(pressure_batch)
    if planes is None:
        planes = make_planes(args.seed + 2, [pressure_batch])[pressure_batch]

    overlap_results = {}
    lane_names = set(args.overlap_lanes)
    ctx = mp.get_context("spawn")
    state_dict = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    cfg_dict = asdict(base.cfg)
    for lane in lanes:
        if lane.name not in lane_names:
            continue
        if lane.evaluator is None:
            overlap_results[lane.name] = {"skipped": lane.skipped or lane.error or "no evaluator"}
            continue
        if not lane.setup:
            overlap_results[lane.name] = {"skipped": "lane has no setup metadata"}
            continue
        stepper = make_train_stepper(
            base,
            device=device,
            batch_size=args.train_batch_size,
            seed=args.seed + 10,
        )
        stop = ctx.Event()
        ready = ctx.Event()
        queue = ctx.Queue()
        proc = ctx.Process(
            target=run_eval_pressure_process,
            args=(
                lane.name,
                lane.setup,
                cfg_dict,
                state_dict,
                not args.no_fuse_eval,
                planes,
                args.pressure_warmup,
                args.pressure_max_calls,
                args.pressure_sleep_ms / 1000.0,
                ready,
                stop,
                queue,
            ),
            daemon=True,
        )
        proc.start()
        became_ready = ready.wait(timeout=args.overlap_ready_timeout_s)
        time.sleep(args.overlap_spinup_s)
        try:
            train_summary = measure_train_stepper(
                stepper,
                device=device,
                warmup=args.train_warmup,
                steps=args.train_steps,
            )
        finally:
            stop.set()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        pressure_alive_after_join = proc.is_alive()
        if not queue.empty():
            pressure_stats = queue.get()
        else:
            pressure_stats = {
                "eval_calls": 0,
                "eval_positions": 0,
                "eval_positions_per_s": 0.0,
                "eval_elapsed_s": 0.0,
                "errors": [f"pressure process produced no stats; exitcode={proc.exitcode}"],
            }
        pressure_stats["exitcode"] = proc.exitcode
        pressure_stats["ready_before_train"] = bool(became_ready)
        pressure_stats["alive_after_join"] = pressure_alive_after_join
        median = train_summary.get("median_ms", 0.0)
        train_summary["slowdown_vs_baseline"] = (
            median / baseline_median if baseline_median > 0 else None
        )
        overlap_results[lane.name] = {
            "train": train_summary,
            "pressure": pressure_stats,
        }
    return {
        "prewarm_train": prewarm,
        "baseline_train": baseline,
        "under_pressure": overlap_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--size", default="tiny", choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--stem-padding", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=parse_int_list, default=parse_int_list("8,32,64"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-fuse-eval", action="store_true")
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument(
        "--coreml-compute-units",
        type=parse_str_list,
        default=parse_str_list("CPU_ONLY,CPU_AND_NE"),
    )
    parser.add_argument(
        "--coreml-precisions",
        type=parse_str_list,
        default=parse_str_list("fp16"),
        help="Comma list from fp16,int8. int8 means linear weight quantization of the FP16 package.",
    )
    parser.add_argument("--train-device", default="mps")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-warmup", type=int, default=1)
    parser.add_argument("--train-prewarm-steps", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--overlap-batch", type=int, default=32)
    parser.add_argument("--overlap-spinup-s", type=float, default=0.2)
    parser.add_argument("--overlap-ready-timeout-s", type=float, default=15.0)
    parser.add_argument("--pressure-warmup", type=int, default=1)
    parser.add_argument("--pressure-max-calls", type=int, default=2000)
    parser.add_argument("--pressure-sleep-ms", type=float, default=1.0)
    parser.add_argument(
        "--overlap-lanes",
        type=parse_str_list,
        default=parse_str_list("torch_mps,coreml_fp16_CPU_AND_NE,coreml_fp16_CPU_ONLY"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.coreml_precisions = [p.lower() for p in args.coreml_precisions]
    args.coreml_compute_units = [u.upper() for u in args.coreml_compute_units]

    load_start = time.perf_counter()
    base, source, metadata = build_base_model(args)
    metadata["load_or_build_s"] = time.perf_counter() - load_start
    fuse_eval = not args.no_fuse_eval
    planes_by_batch = make_planes(args.seed, args.batches)

    lanes = [
        make_torch_lane(
            base,
            name="torch_cpu",
            device=torch.device("cpu"),
            fuse=fuse_eval,
        )
    ]
    if torch.backends.mps.is_available():
        lanes.append(
            make_torch_lane(
                base,
                name="torch_mps",
                device=torch.device("mps"),
                fuse=fuse_eval,
            )
        )
    else:
        lanes.append(Lane(name="torch_mps", skipped="torch.backends.mps.is_available() is false"))

    with tempfile.TemporaryDirectory(prefix="gomoku-engine-scout-") as tmp:
        tmpdir = Path(tmp)
        if not args.skip_coreml:
            lanes.extend(build_coreml_lanes(base, args, tmpdir))
        latency = bench_eval_lanes(
            lanes,
            planes_by_batch,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        overlap = bench_overlap(base, lanes, args, planes_by_batch)

    result = {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "model": {
            "source": source,
            "metadata": metadata,
            "params": n_params(base),
            "fused_eval": fuse_eval,
        },
        "args": {
            "batches": args.batches,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "coreml_precisions": args.coreml_precisions,
            "coreml_compute_units": args.coreml_compute_units,
            "train_batch_size": args.train_batch_size,
            "train_steps": args.train_steps,
            "pressure_max_calls": args.pressure_max_calls,
            "pressure_sleep_ms": args.pressure_sleep_ms,
            "overlap_lanes": args.overlap_lanes,
        },
        "latency": latency,
        "overlap": overlap,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
