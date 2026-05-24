"""Smoke tests for the L08-driver-per-cell-envvars lane.

Asserts that:
  1. parse_cell_env handles None/empty/malformed inputs correctly.
  2. load_cells_csv reads the optional `env` column when present and
     populates cell["env"] as a dict.
  3. load_cells_csv preserves backward compat with pre-L08 cells.csv
     files (no `env` column) — cell["env"] = {} in that case.
  4. write_cells_csv round-trips an env-bearing csv losslessly.
  5. write_cells_csv emits a byte-identical pre-L08 schema when no cell
     has env overrides (no env column at all).

Pure function-level tests; no subprocess.Popen is invoked.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.canonical_sweep import (
    ENV_KV_SEP,
    format_cell_env,
    load_cells_csv,
    parse_cell_env,
    write_cells_csv,
)


def test_parse_cell_env_empty_and_none() -> None:
    assert parse_cell_env(None) == {}
    assert parse_cell_env("") == {}
    assert parse_cell_env("   ") == {}


def test_parse_cell_env_single_pair() -> None:
    assert parse_cell_env("MY_TEST_VAR=hello") == {"MY_TEST_VAR": "hello"}


def test_parse_cell_env_multiple_pairs() -> None:
    raw = f"PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0{ENV_KV_SEP}FOO=bar"
    assert parse_cell_env(raw) == {
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
        "FOO": "bar",
    }


def test_parse_cell_env_value_with_equals() -> None:
    # The first `=` is the splitter, so URL-like values survive.
    assert parse_cell_env("URL=https://x.example?a=b") == {
        "URL": "https://x.example?a=b",
    }


def test_parse_cell_env_strips_whitespace() -> None:
    raw = f"  FOO=1  {ENV_KV_SEP}  BAR=2  "
    assert parse_cell_env(raw) == {"FOO": "1", "BAR": "2"}


def test_parse_cell_env_malformed_no_equals_raises() -> None:
    with pytest.raises(SystemExit):
        parse_cell_env("MALFORMED_NO_EQUALS")


def test_parse_cell_env_empty_key_raises() -> None:
    with pytest.raises(SystemExit):
        parse_cell_env("=value")


def test_format_cell_env_roundtrip() -> None:
    d = {"B": "2", "A": "1"}
    rendered = format_cell_env(d)
    # Stable key order.
    assert rendered == f"A=1{ENV_KV_SEP}B=2"
    assert parse_cell_env(rendered) == d


def test_format_cell_env_empty() -> None:
    assert format_cell_env({}) == ""


# --- cells.csv I/O ----------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def test_load_cells_csv_backward_compat_no_env_column(tmp_path: Path) -> None:
    """Pre-L08 cells.csv (no env column) must still parse; cell["env"] = {}."""
    csv_path = tmp_path / "cells.csv"
    _write_csv(
        csv_path,
        ["cell_id", "model", "workers", "games_per_batch", "n_simulations", "wave_size"],
        [["ignored", "small", "8", "8", "400", "64"]],
    )
    cells = load_cells_csv(csv_path)
    assert len(cells) == 1
    assert cells[0]["model"] == "small"
    assert cells[0]["workers"] == 8
    assert cells[0]["env"] == {}
    # cell_id is re-derived from params; the "ignored" value is dropped.
    assert cells[0]["cell_id"] == "small_W08_G08_S400_V064"


def test_load_cells_csv_with_env_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "cells.csv"
    _write_csv(
        csv_path,
        ["model", "workers", "games_per_batch", "n_simulations", "wave_size", "env"],
        [
            ["small", "8", "8", "400", "512", "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"],
            ["small", "8", "8", "400", "512", "PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.4"],
            ["small", "8", "8", "400", "512", ""],  # empty -> no overrides
        ],
    )
    cells = load_cells_csv(csv_path)
    assert len(cells) == 3
    assert cells[0]["env"] == {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}
    assert cells[1]["env"] == {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.4"}
    assert cells[2]["env"] == {}


def test_load_cells_csv_with_multikey_env(tmp_path: Path) -> None:
    csv_path = tmp_path / "cells.csv"
    env_str = f"FOO=1{ENV_KV_SEP}BAR=two{ENV_KV_SEP}BAZ=three"
    _write_csv(
        csv_path,
        ["model", "workers", "games_per_batch", "n_simulations", "wave_size", "env"],
        [["tiny", "4", "4", "200", "64", env_str]],
    )
    cells = load_cells_csv(csv_path)
    assert cells[0]["env"] == {"FOO": "1", "BAR": "two", "BAZ": "three"}


def test_write_cells_csv_no_env_emits_pre_L08_schema(tmp_path: Path) -> None:
    """If no cell has env, the written csv must have the legacy 6-column header."""
    cells = [
        {
            "cell_id": "small_W08_G08_S400_V064",
            "model": "small",
            "workers": 8,
            "games_per_batch": 8,
            "n_simulations": 400,
            "wave_size": 64,
            "env": {},
        }
    ]
    out_path = tmp_path / "cells.csv"
    write_cells_csv(out_path, cells)
    with out_path.open() as f:
        header = f.readline().strip()
    assert header == "cell_id,model,workers,games_per_batch,n_simulations,wave_size"


def test_write_cells_csv_with_env_emits_env_column(tmp_path: Path) -> None:
    cells = [
        {
            "cell_id": "small_W08_G08_S400_V512",
            "model": "small",
            "workers": 8,
            "games_per_batch": 8,
            "n_simulations": 400,
            "wave_size": 512,
            "env": {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"},
        },
        {
            "cell_id": "small_W08_G08_S400_V064",
            "model": "small",
            "workers": 8,
            "games_per_batch": 8,
            "n_simulations": 400,
            "wave_size": 64,
            "env": {},
        },
    ]
    out_path = tmp_path / "cells.csv"
    write_cells_csv(out_path, cells)
    # Round-trip via load_cells_csv: should reproduce the env dicts exactly.
    reloaded = load_cells_csv(out_path)
    assert reloaded[0]["env"] == {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}
    assert reloaded[1]["env"] == {}
