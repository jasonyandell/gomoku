import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coreml_ane_residency_scout.py"


def load_scout_module():
    spec = importlib.util.spec_from_file_location("coreml_ane_residency_scout", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_powermetrics_text_summarizes_ane_mw_and_watts():
    scout = load_scout_module()
    summary = scout.parse_powermetrics_text(
        """
        CPU Power: 1.25 W
        GPU Power: 250 mW
        ANE Power: 0 mW
        CPU Power: 1500 mW
        GPU Power: 0 mW
        ANE Power: 4.5 W
        """
    )

    assert summary["cpu"]["samples"] == 2
    assert summary["cpu"]["mean_mw"] == 1375.0
    assert summary["gpu"]["max_mw"] == 250.0
    assert summary["ane"]["samples"] == 2
    assert summary["ane"]["max_mw"] == 4500.0
    assert summary["ane"]["active_samples_10mw"] == 1


def test_dry_run_receipt_does_not_require_coremltools_or_sudo():
    scout = load_scout_module()
    args = scout.parse_args(
        [
            "--dry-run",
            "--model-kinds",
            "mlp,gomoku",
            "--powermetrics",
            "never",
            "--duration-s",
            "0.1",
        ]
    )

    receipt = scout.run(args)

    assert receipt["args"]["dry_run"] is True
    assert sorted(receipt["models"]) == ["gomoku", "mlp"]
    assert receipt["models"]["mlp"]["skipped"] == "dry_run"
    assert receipt["models"]["gomoku"]["plan"]["shape"]["channels"] == 17
