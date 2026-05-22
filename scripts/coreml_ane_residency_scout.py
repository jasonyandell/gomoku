"""Core ML / ANE residency scout for Apple-silicon toy models.

This harness exports small Core ML ML Programs, runs each package under
``CPU_ONLY`` and ``CPU_AND_NE`` pressure, and emits a JSON receipt. It is meant
to answer the residency question that a compute-units request alone cannot:
does this shape actually move the ANE power rail on this Mac?

The script is safe to run without sudo. With ``--powermetrics auto`` it samples
``powermetrics`` only when passwordless/cached sudo is already available; use
``--powermetrics never`` or ``--dry-run`` for completely unprivileged scouts.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gomoku.coreml_evaluator import (  # noqa: E402
    CoreMLUnavailableError,
    normalize_compute_units_name,
    require_coremltools,
    resolve_compute_units,
)
from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES  # noqa: E402
from gomoku.model import GomokuNet, ModelConfig, fuse_model_for_inference, n_params  # noqa: E402


MODEL_KINDS = ("mlp", "conv", "resnet", "gomoku")
DEFAULT_COMPUTE_UNITS = "CPU_ONLY,CPU_AND_NE"
INPUT_NAME = "input"


@dataclass(frozen=True)
class Shape:
    channels: int
    height: int
    width: int


@dataclass(frozen=True)
class ModelPlan:
    kind: str
    shape: Shape
    hidden: int
    depth: int
    filters: int
    blocks: int
    output_width: int


class FlattenMLP(nn.Module):
    def __init__(self, shape: Shape, hidden: int, depth: int, output_width: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = shape.channels * shape.height * shape.width
        for i in range(max(1, depth)):
            layers.append(nn.Linear(in_features if i == 0 else hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, output_width))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvToy(nn.Module):
    def __init__(
        self,
        shape: Shape,
        *,
        filters: int,
        depth: int,
        hidden: int,
        output_width: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = shape.channels
        for _ in range(max(1, depth)):
            layers.extend(
                [
                    nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(),
                ]
            )
            in_channels = filters
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * shape.height * shape.width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class ResidualToyBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=True),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class ResnetToy(nn.Module):
    def __init__(
        self,
        shape: Shape,
        *,
        filters: int,
        blocks: int,
        hidden: int,
        output_width: int,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(shape.channels, filters, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualToyBlock(filters) for _ in range(max(1, blocks))])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * shape.height * shape.width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_model_kinds(raw: str) -> list[str]:
    kinds = parse_csv(raw.lower())
    if not kinds:
        raise argparse.ArgumentTypeError("expected at least one model kind")
    unknown = sorted(set(kinds) - set(MODEL_KINDS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown model kind(s): {', '.join(unknown)}; options: {', '.join(MODEL_KINDS)}"
        )
    return kinds


def parse_compute_units(raw: str) -> list[str]:
    units = []
    for part in parse_csv(raw):
        try:
            units.append(normalize_compute_units_name(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    if not units:
        raise argparse.ArgumentTypeError("expected at least one Core ML compute unit")
    return units


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def default_shape(kind: str) -> Shape:
    if kind == "mlp":
        return Shape(channels=1, height=1, width=1024)
    if kind in {"conv", "resnet"}:
        return Shape(channels=16, height=32, width=32)
    if kind == "gomoku":
        return Shape(channels=N_INPUT_PLANES, height=BOARD_SIZE, width=BOARD_SIZE)
    raise ValueError(f"unknown model kind: {kind}")


def make_plan(kind: str, args: argparse.Namespace) -> ModelPlan:
    base = default_shape(kind)
    shape = Shape(
        channels=args.channels if args.channels is not None else base.channels,
        height=args.height if args.height is not None else base.height,
        width=args.width if args.width is not None else base.width,
    )
    if kind == "gomoku" and (shape.height != BOARD_SIZE or shape.width != BOARD_SIZE):
        raise ValueError(
            f"gomoku shape must be {BOARD_SIZE}x{BOARD_SIZE}; got {shape.height}x{shape.width}"
        )
    return ModelPlan(
        kind=kind,
        shape=shape,
        hidden=args.hidden,
        depth=args.depth,
        filters=args.filters,
        blocks=args.blocks,
        output_width=args.output_width,
    )


def build_model(plan: ModelPlan) -> nn.Module:
    if plan.kind == "mlp":
        return FlattenMLP(plan.shape, plan.hidden, plan.depth, plan.output_width)
    if plan.kind == "conv":
        return ConvToy(
            plan.shape,
            filters=plan.filters,
            depth=plan.depth,
            hidden=plan.hidden,
            output_width=plan.output_width,
        )
    if plan.kind == "resnet":
        return ResnetToy(
            plan.shape,
            filters=plan.filters,
            blocks=plan.blocks,
            hidden=plan.hidden,
            output_width=plan.output_width,
        )
    if plan.kind == "gomoku":
        cfg = ModelConfig(
            n_input_planes=plan.shape.channels,
            n_filters=plan.filters,
            n_blocks=plan.blocks,
            policy_filters=2,
            value_filters=1,
            value_hidden=plan.hidden,
            stem_padding=1,
        )
        return GomokuNet(cfg)
    raise ValueError(f"unknown model kind: {plan.kind}")


def export_coreml_model(
    model: nn.Module,
    plan: ModelPlan,
    output_path: Path,
    *,
    batch_size: int,
    max_batch_size: int,
    batch_shape: str,
    compute_precision: str,
    input_dtype: str,
) -> dict[str, Any]:
    ct = require_coremltools()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_batch = batch_size if batch_shape == "fixed" else 1
    dummy = torch.zeros(
        dummy_batch,
        plan.shape.channels,
        plan.shape.height,
        plan.shape.width,
        dtype=torch.float32,
    )
    traced = torch.jit.trace(model.cpu(), dummy)
    batch_dim: int | Any
    if batch_shape == "fixed":
        batch_dim = batch_size
    else:
        batch_dim = ct.RangeDim(lower_bound=1, upper_bound=max_batch_size)
    precision = getattr(ct.precision, compute_precision.upper())
    tensor_dtype = np.float16 if input_dtype == "float16" else np.float32
    start = time.perf_counter()
    convert_kwargs = {}
    if input_dtype == "float16":
        # Float16 MultiArray inputs require iOS16/macOS13 or newer. coremltools
        # exposes deployment targets through the iOS names even for macOS runs.
        convert_kwargs["minimum_deployment_target"] = ct.target.iOS16
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name=INPUT_NAME,
                shape=(
                    batch_dim,
                    plan.shape.channels,
                    plan.shape.height,
                    plan.shape.width,
                ),
                dtype=tensor_dtype,
            )
        ],
        compute_precision=precision,
        **convert_kwargs,
    )
    convert_s = time.perf_counter() - start
    save_start = time.perf_counter()
    mlmodel.save(str(output_path))
    return {
        "path": str(output_path),
        "convert_s": convert_s,
        "save_s": time.perf_counter() - save_start,
        "batch_shape": batch_shape,
        "batch_size": batch_size,
        "compute_precision": compute_precision.upper(),
        "input_dtype": input_dtype,
        "max_batch_size": max_batch_size,
    }


POWER_RE = re.compile(r"\b(CPU|GPU|ANE) Power:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b")


def parse_powermetrics_text(text: str) -> dict[str, Any]:
    rails: dict[str, list[float]] = {"CPU": [], "GPU": [], "ANE": []}
    for rail, raw_value, unit in POWER_RE.findall(text):
        value_mw = float(raw_value)
        if unit == "W":
            value_mw *= 1000.0
        rails[rail].append(value_mw)

    summary = {}
    for rail, samples in rails.items():
        if samples:
            summary[rail.lower()] = {
                "samples": len(samples),
                "min_mw": min(samples),
                "mean_mw": sum(samples) / len(samples),
                "max_mw": max(samples),
                "nonzero_samples": sum(1 for sample in samples if sample > 1.0),
                "active_samples_10mw": sum(1 for sample in samples if sample >= 10.0),
            }
        else:
            summary[rail.lower()] = {
                "samples": 0,
                "min_mw": None,
                "mean_mw": None,
                "max_mw": None,
                "nonzero_samples": 0,
                "active_samples_10mw": 0,
            }
    return summary


def sudo_cached() -> bool:
    return subprocess.run(
        ["sudo", "-n", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


@dataclass
class PowermetricsSession:
    status: str
    output_path: str | None = None
    command: list[str] | None = None
    reason: str | None = None
    process: subprocess.Popen | None = None


def start_powermetrics(
    args: argparse.Namespace,
    *,
    label: str,
    raw_dir: Path,
) -> PowermetricsSession:
    if args.powermetrics == "never":
        return PowermetricsSession(status="skipped", reason="--powermetrics never")
    if shutil.which("powermetrics") is None:
        if args.powermetrics == "required":
            return PowermetricsSession(status="error", reason="powermetrics not found")
        return PowermetricsSession(status="skipped", reason="powermetrics not found")
    if not sudo_cached():
        reason = "sudo -n true failed; cached/passwordless sudo is unavailable"
        if args.powermetrics == "required":
            return PowermetricsSession(status="error", reason=reason)
        return PowermetricsSession(status="skipped", reason=reason)

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{label}.powermetrics.txt"
    samples = max(1, int(math.ceil(args.duration_s / (args.powermetrics_interval_ms / 1000.0))) + 1)
    command = [
        "sudo",
        "-n",
        "powermetrics",
        "--samplers",
        args.powermetrics_samplers,
        "-i",
        str(args.powermetrics_interval_ms),
        "-n",
        str(samples),
    ]
    handle = output_path.open("w")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    return PowermetricsSession(
        status="running",
        output_path=str(output_path),
        command=command,
        process=process,
    )


def finish_powermetrics(session: PowermetricsSession, timeout_s: float = 5.0) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": session.status,
        "output_path": session.output_path,
        "command": session.command,
        "reason": session.reason,
    }
    if session.process is not None:
        try:
            session.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            session.process.terminate()
            try:
                session.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                session.process.kill()
                session.process.wait(timeout=timeout_s)
        result["exitcode"] = session.process.returncode

    if session.output_path and Path(session.output_path).exists():
        text = Path(session.output_path).read_text(errors="replace")
        result["summary"] = parse_powermetrics_text(text)
        result["bytes"] = len(text.encode("utf-8", errors="replace"))
    return result


def pressure_worker(
    model_path: str,
    compute_units: str,
    shape: tuple[int, int, int],
    batch_size: int,
    input_dtype: str,
    seed: int,
    warmup: int,
    duration_s: float,
    max_iters: int,
    ready: mp.Event,
    stop: mp.Event,
    queue: mp.Queue,
) -> None:
    started = time.perf_counter()
    calls = 0
    positions = 0
    errors: list[str] = []
    first_predict_s: float | None = None
    load_s: float | None = None
    stopped_by = "duration"
    compute_start: float | None = None
    compute_elapsed_s = 0.0
    try:
        ct = require_coremltools()
        load_start = time.perf_counter()
        mlmodel = ct.models.MLModel(
            model_path,
            compute_units=resolve_compute_units(compute_units),
        )
        load_s = time.perf_counter() - load_start
        rng = np.random.default_rng(seed)
        dtype = np.float16 if input_dtype == "float16" else np.float32
        x = rng.normal(size=(batch_size, *shape)).astype(dtype)
        for _ in range(warmup):
            mlmodel.predict({INPUT_NAME: x})
        ready.set()
        compute_start = time.perf_counter()
        deadline = time.perf_counter() + duration_s
        while not stop.is_set() and time.perf_counter() < deadline:
            start = time.perf_counter()
            mlmodel.predict({INPUT_NAME: x})
            elapsed = time.perf_counter() - start
            if first_predict_s is None:
                first_predict_s = elapsed
            calls += 1
            positions += batch_size
            if max_iters and calls >= max_iters:
                stopped_by = "max_iters"
                break
        if stop.is_set():
            stopped_by = "stop_event"
    except Exception as exc:  # pragma: no cover - worker failure path
        errors.append(f"{type(exc).__name__}: {exc}")
        ready.set()

    finished = time.perf_counter()
    process_elapsed_s = max(finished - started, 1e-9)
    if compute_start is not None:
        compute_elapsed_s = max(finished - compute_start, 1e-9)
    queue.put(
        {
            "calls": calls,
            "positions": positions,
            "process_elapsed_s": process_elapsed_s,
            "compute_elapsed_s": compute_elapsed_s,
            "positions_per_s": positions / compute_elapsed_s if compute_elapsed_s > 0 else 0.0,
            "first_predict_s": first_predict_s,
            "load_s": load_s,
            "stopped_by": stopped_by,
            "errors": errors,
        }
    )


def run_pressure_phase(
    model_path: Path,
    plan: ModelPlan,
    compute_units: str,
    args: argparse.Namespace,
    *,
    label: str,
    raw_dir: Path,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    ready_events = [ctx.Event() for _ in range(args.workers)]
    stop = ctx.Event()
    queue = ctx.Queue()
    processes = []
    powermetrics = start_powermetrics(args, label=label, raw_dir=raw_dir)
    if powermetrics.status == "error":
        return {
            "compute_units": compute_units,
            "error": powermetrics.reason,
            "powermetrics": finish_powermetrics(powermetrics),
        }

    for worker_id in range(args.workers):
        proc = ctx.Process(
            target=pressure_worker,
            args=(
                str(model_path),
                compute_units,
                (plan.shape.channels, plan.shape.height, plan.shape.width),
                args.batch_size,
                args.input_dtype,
                args.seed + worker_id,
                args.warmup,
                args.duration_s,
                args.max_iters,
                ready_events[worker_id],
                stop,
                queue,
            ),
            daemon=True,
        )
        proc.start()
        processes.append(proc)

    ready_deadline = time.perf_counter() + args.ready_timeout_s
    ready_count = 0
    for event in ready_events:
        remaining = max(0.0, ready_deadline - time.perf_counter())
        if event.wait(timeout=remaining):
            ready_count += 1

    for proc in processes:
        proc.join(timeout=args.duration_s + args.ready_timeout_s + 10.0)
    stop.set()
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)

    worker_results = []
    while not queue.empty():
        worker_results.append(queue.get())
    powermetrics_result = finish_powermetrics(powermetrics)
    total_positions = sum(int(item.get("positions", 0)) for item in worker_results)
    compute_elapsed_values = [
        float(item.get("compute_elapsed_s", 0.0)) for item in worker_results if item
    ]
    process_elapsed_values = [
        float(item.get("process_elapsed_s", 0.0)) for item in worker_results if item
    ]
    phase_compute_elapsed_s = max(compute_elapsed_values) if compute_elapsed_values else 0.0
    phase_process_elapsed_s = max(process_elapsed_values) if process_elapsed_values else 0.0
    return {
        "compute_units": compute_units,
        "workers": args.workers,
        "ready_workers": ready_count,
        "batch_size": args.batch_size,
        "duration_s": args.duration_s,
        "max_iters": args.max_iters,
        "worker_results": worker_results,
        "exitcodes": [proc.exitcode for proc in processes],
        "total_positions": total_positions,
        "compute_elapsed_s": phase_compute_elapsed_s,
        "process_elapsed_s": phase_process_elapsed_s,
        "positions_per_s": (
            total_positions / phase_compute_elapsed_s if phase_compute_elapsed_s > 0 else 0.0
        ),
        "powermetrics": powermetrics_result,
    }


def host_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    try:
        ct = require_coremltools()
        info["coremltools"] = getattr(ct, "__version__", "unknown")
    except CoreMLUnavailableError as exc:
        info["coremltools_error"] = str(exc)
    return info


def run(args: argparse.Namespace) -> dict[str, Any]:
    plans = [make_plan(kind, args) for kind in args.model_kinds]
    raw_dir = args.raw_dir
    if raw_dir is None and args.output is not None:
        raw_dir = args.output.with_suffix("").parent / f"{args.output.with_suffix('').name}-raw"
    if raw_dir is None:
        raw_dir = Path(tempfile.mkdtemp(prefix="coreml-ane-residency-raw-"))

    receipt: dict[str, Any] = {
        "host": host_info(),
        "args": {
            "model_kinds": args.model_kinds,
            "compute_units": args.compute_units,
            "batch_size": args.batch_size,
            "duration_s": args.duration_s,
            "workers": args.workers,
            "warmup": args.warmup,
            "max_iters": args.max_iters,
            "powermetrics": args.powermetrics,
            "powermetrics_samplers": args.powermetrics_samplers,
            "powermetrics_interval_ms": args.powermetrics_interval_ms,
            "compute_precision": args.compute_precision,
            "batch_shape": args.batch_shape,
            "input_dtype": args.input_dtype,
            "fuse_gomoku_eval": args.fuse_gomoku_eval,
            "dry_run": args.dry_run,
            "retain_models": args.retain_models,
        },
        "models": {},
    }

    for plan in plans:
        model_receipt: dict[str, Any] = {
            "plan": asdict(plan),
            "input_name": INPUT_NAME,
        }
        if args.dry_run:
            model_receipt["skipped"] = "dry_run"
            receipt["models"][plan.kind] = model_receipt
            continue

        model = build_model(plan).eval()
        if plan.kind == "gomoku" and args.fuse_gomoku_eval:
            model = fuse_model_for_inference(model)
        model_receipt["params"] = n_params(model)
        model_dir_ctx = None
        if args.retain_models:
            model_dir = args.model_dir
            model_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_dir_ctx = tempfile.TemporaryDirectory(prefix=f"coreml-ane-{plan.kind}-")
            model_dir = Path(model_dir_ctx.name)

        try:
            model_path = model_dir / f"{plan.kind}_{args.compute_precision.lower()}.mlpackage"
            try:
                model_receipt["export"] = export_coreml_model(
                    model,
                    plan,
                    model_path,
                    batch_size=args.batch_size,
                    max_batch_size=max(args.batch_size, args.max_batch_size),
                    batch_shape=args.batch_shape,
                    compute_precision=args.compute_precision,
                    input_dtype=args.input_dtype,
                )
            except Exception as exc:
                model_receipt["error"] = f"export failed: {type(exc).__name__}: {exc}"
                receipt["models"][plan.kind] = model_receipt
                continue

            phases = {}
            for compute_units in args.compute_units:
                phase_label = f"{plan.kind}-{compute_units.lower()}"
                phases[compute_units] = run_pressure_phase(
                    model_path,
                    plan,
                    compute_units,
                    args,
                    label=phase_label,
                    raw_dir=raw_dir,
                )
            model_receipt["pressure"] = phases
        finally:
            if model_dir_ctx is not None:
                model_dir_ctx.cleanup()
        receipt["models"][plan.kind] = model_receipt
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export toy Core ML models and pressure CPU_ONLY/CPU_AND_NE while optionally sampling ANE power.",
    )
    parser.add_argument(
        "--model-kinds",
        type=parse_model_kinds,
        default=parse_model_kinds("mlp,conv,resnet,gomoku"),
        help=f"Comma list of model families: {', '.join(MODEL_KINDS)}.",
    )
    parser.add_argument("--channels", type=positive_int, default=None)
    parser.add_argument("--height", type=positive_int, default=None)
    parser.add_argument("--width", type=positive_int, default=None)
    parser.add_argument("--hidden", type=positive_int, default=128)
    parser.add_argument("--depth", type=positive_int, default=2)
    parser.add_argument("--filters", type=positive_int, default=32)
    parser.add_argument("--blocks", type=positive_int, default=2)
    parser.add_argument("--output-width", type=positive_int, default=N_ACTIONS)
    parser.add_argument(
        "--compute-units",
        type=parse_compute_units,
        default=parse_compute_units(DEFAULT_COMPUTE_UNITS),
        help="Comma list such as CPU_ONLY,CPU_AND_NE. Aliases like ane are accepted.",
    )
    parser.add_argument("--compute-precision", default="FLOAT16", choices=["FLOAT16", "FLOAT32"])
    parser.add_argument(
        "--batch-shape",
        default="range",
        choices=["fixed", "range"],
        help="Use a fixed batch dimension or the previous RangeDim flexible batch.",
    )
    parser.add_argument(
        "--input-dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Core ML input feature dtype and NumPy pressure dtype.",
    )
    parser.add_argument(
        "--fuse-gomoku-eval",
        action="store_true",
        help="Fold Conv+BatchNorm before exporting the Gomoku-shaped model.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--max-batch-size", type=positive_int, default=512)
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument(
        "--max-iters",
        type=nonnegative_int,
        default=0,
        help="Per-worker iteration cap. 0 means use duration only.",
    )
    parser.add_argument("--ready-timeout-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--powermetrics", choices=["auto", "never", "required"], default="auto")
    parser.add_argument(
        "--powermetrics-samplers",
        default="cpu_power,gpu_power,ane_power",
        help="Passed directly to powermetrics --samplers.",
    )
    parser.add_argument("--powermetrics-interval-ms", type=positive_int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path("sweep_logs/coreml_ane_models"))
    parser.add_argument("--retain-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
