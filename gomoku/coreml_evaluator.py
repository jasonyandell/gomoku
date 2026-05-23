"""Core ML inference wrapper for eval-only gomoku models.

This module is intentionally lazy about importing ``coremltools``. The normal
training/test path should not require Core ML, but scout harnesses can opt into
loading an existing ``.mlmodel``/``.mlpackage`` or exporting a PyTorch
checkpoint when the package is available.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gomoku.game import BOARD_SIZE, N_ACTIONS, N_INPUT_PLANES, GameState


DEFAULT_INPUT_NAME = "planes"
DEFAULT_POLICY_OUTPUT_NAME = "policy_logits"
DEFAULT_VALUE_OUTPUT_NAME = "value"
SUPPORTED_COMPUTE_UNITS = ("CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU", "ALL")


class CoreMLUnavailableError(RuntimeError):
    """Raised when a Core ML path is requested without coremltools installed."""


def require_coremltools() -> Any:
    """Import coremltools or raise an actionable error."""
    try:
        return importlib.import_module("coremltools")
    except ModuleNotFoundError as exc:
        if exc.name != "coremltools":
            raise
        raise CoreMLUnavailableError(
            "Core ML evaluation requires coremltools. Install it with "
            "`uv pip install coremltools` in this environment, or run the "
            "scout with the existing Torch evaluator backend."
        ) from exc


def normalize_compute_units_name(compute_units: str) -> str:
    """Normalize a user-facing Core ML compute-units string."""
    name = compute_units.upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "CPU": "CPU_ONLY",
        "ANE": "CPU_AND_NE",
        "CPU_NE": "CPU_AND_NE",
        "CPU_AND_ANE": "CPU_AND_NE",
    }
    name = aliases.get(name, name)
    if name not in SUPPORTED_COMPUTE_UNITS:
        options = ", ".join(SUPPORTED_COMPUTE_UNITS)
        raise ValueError(f"unknown Core ML compute_units {compute_units!r}; options: {options}")
    return name


def resolve_compute_units(compute_units: str) -> Any:
    """Resolve a compute-units name to ``coremltools.ComputeUnit``."""
    ct = require_coremltools()
    name = normalize_compute_units_name(compute_units)
    try:
        return getattr(ct.ComputeUnit, name)
    except AttributeError as exc:
        raise ValueError(
            f"installed coremltools does not expose ComputeUnit.{name}; "
            f"try one of: {', '.join(SUPPORTED_COMPUTE_UNITS)}"
        ) from exc


def _spec_io_names(mlmodel: Any) -> tuple[list[str], list[str]]:
    try:
        spec = mlmodel.get_spec()
    except Exception:
        return [], []
    description = getattr(spec, "description", None)
    if description is None:
        return [], []
    inputs = [item.name for item in getattr(description, "input", [])]
    outputs = [item.name for item in getattr(description, "output", [])]
    return inputs, outputs


def _discover_static_batch_sizes(mlmodel: Any, input_name: str | None) -> list[int]:
    """Return the model's allowed (static) input batch sizes, sorted ascending.

    Static-shape ANE models declare either a single fixed batch dimension or a
    set of ``EnumeratedShapes``. We read those off the model spec so the
    evaluator can pad each incoming leaf-batch up to a declared size (the ANE
    only accepts the exact shapes it was compiled for) and slice the outputs
    back down. Returns ``[]`` for symbolic/flexible-batch models (e.g. the old
    RangeDim export), in which case the evaluator feeds the real batch as-is.
    """
    try:
        spec = mlmodel.get_spec()
    except Exception:
        return []
    description = getattr(spec, "description", None)
    if description is None:
        return []
    inputs = list(getattr(description, "input", []))
    chosen = None
    for item in inputs:
        if input_name is not None and item.name == input_name:
            chosen = item
            break
    if chosen is None and inputs:
        chosen = inputs[0]
    if chosen is None:
        return []
    try:
        multi = chosen.type.multiArrayType
    except Exception:
        return []
    sizes: set[int] = set()
    try:
        if multi.HasField("enumeratedShapes"):
            for shape in multi.enumeratedShapes.shapes:
                dims = list(shape.shape)
                if dims:
                    sizes.add(int(dims[0]))
    except (AttributeError, ValueError):
        pass
    if not sizes:
        # No enumerated set: a plain static shape has a concrete leading dim,
        # while a RangeDim/symbolic export leaves it as a non-positive sentinel.
        dims = list(getattr(multi, "shape", []))
        if dims and int(dims[0]) > 0:
            sizes.add(int(dims[0]))
    return sorted(sizes)


def _states_to_batch(states: list[GameState]) -> np.ndarray:
    x = np.empty(
        (len(states), N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )
    for i, state in enumerate(states):
        x[i] = state.to_planes()
    return x


def _validate_planes(planes: np.ndarray) -> np.ndarray:
    x = np.asarray(planes, dtype=np.float32)
    expected = (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    if x.ndim != 4 or tuple(x.shape[1:]) != expected:
        raise ValueError(
            "CoreMLEvaluator.evaluate_planes expected shape "
            f"(B, {expected[0]}, {expected[1]}, {expected[2]}), got {tuple(x.shape)}"
        )
    return x


def _pick_named_or_first(
    result: dict[str, Any],
    preferred_name: str | None,
    fallback_names: tuple[str, ...],
    output_index: int,
) -> np.ndarray:
    if preferred_name is not None and preferred_name in result:
        return np.asarray(result[preferred_name])
    for name in fallback_names:
        if name in result:
            return np.asarray(result[name])
    values = list(result.values())
    if output_index < len(values):
        return np.asarray(values[output_index])
    available = ", ".join(result.keys()) or "<none>"
    raise KeyError(f"Core ML prediction did not return output {output_index}; keys: {available}")


@dataclass
class CoreMLEvaluator:
    """Evaluator-compatible wrapper around a Core ML model.

    Parameters
    ----------
    model_path:
        Existing ``.mlpackage`` or ``.mlmodel`` path. This is the preferred scout
        path because it keeps conversion out of the hot benchmark loop.
    compute_units:
        Core ML compute unit selection, for example ``CPU_ONLY`` or
        ``CPU_AND_NE``.
    mlmodel:
        Optional already-loaded model object. This exists mainly for focused
        tests and custom harnesses.
    """

    model_path: str | Path | None = None
    compute_units: str = "CPU_AND_NE"
    input_name: str | None = None
    policy_output_name: str | None = None
    value_output_name: str | None = None
    max_batch_size: int | None = None
    mlmodel: Any | None = None
    static_batch_sizes: list[int] | None = None

    def __post_init__(self) -> None:
        if self.mlmodel is None:
            if self.model_path is None:
                raise ValueError("model_path is required when mlmodel is not provided")
            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(f"Core ML model not found: {path}")
            ct = require_coremltools()
            self.mlmodel = ct.models.MLModel(
                str(path),
                compute_units=resolve_compute_units(self.compute_units),
            )

        inputs, outputs = _spec_io_names(self.mlmodel)
        if self.input_name is None:
            self.input_name = inputs[0] if inputs else DEFAULT_INPUT_NAME
        if self.policy_output_name is None:
            self.policy_output_name = (
                DEFAULT_POLICY_OUTPUT_NAME if DEFAULT_POLICY_OUTPUT_NAME in outputs else None
            )
        if self.value_output_name is None:
            self.value_output_name = (
                DEFAULT_VALUE_OUTPUT_NAME if DEFAULT_VALUE_OUTPUT_NAME in outputs else None
            )
        if self.policy_output_name is None and len(outputs) >= 1:
            self.policy_output_name = outputs[0]
        if self.value_output_name is None and len(outputs) >= 2:
            self.value_output_name = outputs[1]

        # Static-shape (ANE-resident) models accept only the batch sizes they
        # were compiled for. Discover them so evaluate_planes can pad-up + slice.
        # An empty list means a symbolic/flexible-batch model (old RangeDim
        # export), where the real batch can be fed as-is.
        if self.static_batch_sizes is None:
            self.static_batch_sizes = _discover_static_batch_sizes(
                self.mlmodel, self.input_name
            )
        if self.static_batch_sizes:
            largest_static = self.static_batch_sizes[-1]
            if self.max_batch_size is None or self.max_batch_size > largest_static:
                self.max_batch_size = largest_static

    def _static_target(self, batch: int) -> int | None:
        """Smallest declared static batch size >= ``batch`` (None if symbolic)."""
        if not self.static_batch_sizes:
            return None
        for size in self.static_batch_sizes:
            if size >= batch:
                return size
        return self.static_batch_sizes[-1]

    def __call__(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluate_planes(_states_to_batch(states))

    def evaluate_planes(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = _validate_planes(planes)
        if x.shape[0] == 0:
            return (
                np.zeros((0, N_ACTIONS), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        if self.max_batch_size is not None and x.shape[0] > self.max_batch_size:
            prior_chunks = []
            value_chunks = []
            for start in range(0, x.shape[0], self.max_batch_size):
                priors, values = self.evaluate_planes(x[start : start + self.max_batch_size])
                prior_chunks.append(priors)
                value_chunks.append(values)
            return np.concatenate(prior_chunks, axis=0), np.concatenate(value_chunks, axis=0)

        assert self.mlmodel is not None
        batch = x.shape[0]
        # Static-shape (ANE) models accept only their declared batch sizes, so
        # pad the real leaf-batch up to the smallest declared size, then slice
        # the policy+value outputs back to the real batch. Symbolic models
        # (static_batch_sizes == []) feed the real batch directly.
        target = self._static_target(batch)
        if target is not None and target != batch:
            padded = np.zeros(
                (target, x.shape[1], x.shape[2], x.shape[3]), dtype=x.dtype
            )
            padded[:batch] = x
            x = padded
        result = self.mlmodel.predict({self.input_name: x})
        priors = _pick_named_or_first(
            result,
            self.policy_output_name,
            ("policy_logits", "policy", "logits"),
            0,
        )
        values = _pick_named_or_first(
            result,
            self.value_output_name,
            ("value", "values"),
            1,
        )
        run_batch = x.shape[0]
        priors = np.asarray(priors, dtype=np.float32).reshape(run_batch, -1)
        values = np.asarray(values, dtype=np.float32).reshape(run_batch)
        # Slice padded rows back off so the MCTS caller gets exactly one
        # (policy, value) per real leaf.
        if run_batch != batch:
            priors = priors[:batch]
            values = values[:batch]
        if priors.shape != (batch, N_ACTIONS):
            raise ValueError(
                f"Core ML policy output expected shape ({batch}, {N_ACTIONS}), "
                f"got {tuple(priors.shape)}"
            )
        if not np.all(np.isfinite(priors)):
            priors = np.nan_to_num(priors, nan=0.0, posinf=1e6, neginf=-1e6)
        if not np.all(np.isfinite(values)):
            values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
        return priors, values


def make_coreml_evaluator(
    model_path: str | Path,
    *,
    compute_units: str = "CPU_AND_NE",
    input_name: str | None = None,
    policy_output_name: str | None = None,
    value_output_name: str | None = None,
    max_batch_size: int | None = None,
) -> CoreMLEvaluator:
    """Create an evaluator from an existing Core ML model path."""
    return CoreMLEvaluator(
        model_path=model_path,
        compute_units=compute_units,
        input_name=input_name,
        policy_output_name=policy_output_name,
        value_output_name=value_output_name,
        max_batch_size=max_batch_size,
    )


def export_model_to_coreml(
    model: Any,
    output_path: str | Path,
    *,
    max_batch_size: int = 512,
    batch_sizes: list[int] | None = None,
    input_name: str = DEFAULT_INPUT_NAME,
    policy_output_name: str = DEFAULT_POLICY_OUTPUT_NAME,
    value_output_name: str = DEFAULT_VALUE_OUTPUT_NAME,
    compute_precision: str = "FLOAT16",
) -> Path:
    """Export an eval-only PyTorch model to a *static-shape* Core ML package.

    By default the export declares a SINGLE fixed batch dimension equal to
    ``max_batch_size``. This is the shape that restores Apple Neural Engine
    residency: a symbolic ``RangeDim`` batch forces Core ML to compile the whole
    program to CPU/BNNS (lane L09i diagnostic), and on this model+hardware so
    does ``ct.EnumeratedShapes`` (lane L09i-fix residency probe: enumerated
    shapes ran on BNNSEngine, a single fixed shape ran on the ANE). The
    companion ``CoreMLEvaluator`` reads the declared batch size off the model
    spec and pads each real leaf-batch up to it, then slices the policy+value
    outputs back to the real batch size.

    ``batch_sizes`` is an opt-in escape hatch: pass an explicit list to emit
    ``ct.EnumeratedShapes`` over those sizes (kept only for residency
    experiments — it currently lands on CPU/BNNS for this model). A single-item
    list emits a fixed shape, identical to the default behaviour.
    """
    ct = require_coremltools()
    import torch

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    traced = torch.jit.trace(model.cpu(), dummy)

    if batch_sizes is None:
        # Single fixed batch == max_batch_size: the only shape confirmed
        # ANE-resident for this model in the L09i-fix probe.
        batch_sizes = [int(max_batch_size)]
    batch_sizes = sorted({int(b) for b in batch_sizes if int(b) >= 1})
    if not batch_sizes:
        raise ValueError("batch_sizes must contain at least one positive batch size")

    plane_shape = (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    precision = getattr(ct.precision, compute_precision.upper())

    if len(batch_sizes) == 1:
        # A single declared batch is the simplest fully-static shape and the
        # one that stays ANE-placeable.
        input_shape: Any = (batch_sizes[0], *plane_shape)
    else:
        input_shape = ct.EnumeratedShapes(
            shapes=[(b, *plane_shape) for b in batch_sizes],
            default=(batch_sizes[-1], *plane_shape),
        )

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name=input_name,
                shape=input_shape,
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name=policy_output_name),
            ct.TensorType(name=value_output_name),
        ],
        compute_precision=precision,
    )
    mlmodel.save(str(output))
    return output


def export_checkpoint_to_coreml(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    max_batch_size: int = 512,
    fuse: bool = True,
    compute_precision: str = "FLOAT16",
) -> Path:
    """Load a gomoku checkpoint and export an eval-only Core ML package."""
    from gomoku.model import fuse_model_for_inference, load_checkpoint

    model, _payload = load_checkpoint(str(checkpoint_path), device="cpu")
    if fuse:
        model = fuse_model_for_inference(model)
    else:
        model.eval()
    return export_model_to_coreml(
        model,
        output_path,
        max_batch_size=max_batch_size,
        compute_precision=compute_precision,
    )


__all__ = [
    "CoreMLEvaluator",
    "CoreMLUnavailableError",
    "export_checkpoint_to_coreml",
    "export_model_to_coreml",
    "make_coreml_evaluator",
    "normalize_compute_units_name",
    "require_coremltools",
    "resolve_compute_units",
]
