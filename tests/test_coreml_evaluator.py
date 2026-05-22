import importlib

import numpy as np
import pytest

from gomoku.coreml_evaluator import (
    CoreMLEvaluator,
    CoreMLUnavailableError,
    normalize_compute_units_name,
    require_coremltools,
)
from gomoku.game import GameState, N_ACTIONS, N_INPUT_PLANES


class _FakeCoreMLModel:
    def __init__(self):
        self.calls = []

    def predict(self, inputs):
        self.calls.append(inputs)
        x = inputs["planes"]
        batch = x.shape[0]
        priors = np.arange(batch * N_ACTIONS, dtype=np.float32).reshape(batch, N_ACTIONS)
        priors[0, 0] = np.nan
        values = np.linspace(-0.5, 0.5, batch, dtype=np.float32)
        if batch > 1:
            values[-1] = np.inf
        return {"policy_logits": priors, "value": values}


def test_coreml_evaluator_matches_planes_boundary():
    model = _FakeCoreMLModel()
    evaluator = CoreMLEvaluator(mlmodel=model)
    x = np.zeros((2, N_INPUT_PLANES, 9, 9), dtype=np.float32)

    priors, values = evaluator.evaluate_planes(x)

    assert priors.shape == (2, N_ACTIONS)
    assert values.shape == (2,)
    assert priors.dtype == np.float32
    assert values.dtype == np.float32
    assert priors[0, 0] == 0.0
    assert values[-1] == 1.0
    assert model.calls[0]["planes"].shape == x.shape


def test_coreml_evaluator_can_be_called_with_states():
    evaluator = CoreMLEvaluator(mlmodel=_FakeCoreMLModel())

    priors, values = evaluator([GameState.initial()])

    assert priors.shape == (1, N_ACTIONS)
    assert values.shape == (1,)


def test_coreml_evaluator_chunks_by_max_batch_size():
    model = _FakeCoreMLModel()
    evaluator = CoreMLEvaluator(mlmodel=model, max_batch_size=2)
    x = np.zeros((5, N_INPUT_PLANES, 9, 9), dtype=np.float32)

    priors, values = evaluator.evaluate_planes(x)

    assert priors.shape == (5, N_ACTIONS)
    assert values.shape == (5,)
    assert [call["planes"].shape[0] for call in model.calls] == [2, 2, 1]


def test_compute_units_name_normalization():
    assert normalize_compute_units_name("cpu_only") == "CPU_ONLY"
    assert normalize_compute_units_name("cpu-and-ne") == "CPU_AND_NE"
    assert normalize_compute_units_name("ane") == "CPU_AND_NE"
    with pytest.raises(ValueError, match="unknown Core ML compute_units"):
        normalize_compute_units_name("neural_fire")


def test_require_coremltools_message_when_missing(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "coremltools":
            raise ModuleNotFoundError("No module named 'coremltools'", name=name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(CoreMLUnavailableError, match="uv pip install coremltools"):
        require_coremltools()
