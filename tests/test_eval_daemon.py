"""Sensei daemon: parse helpers + the FastAPI service end-to-end (no Rapfi)."""

from __future__ import annotations

import pytest

from gomoku.eval_daemon import (
    build_rulers,
    create_app,
    parse_opening,
    parse_ruler_specs,
)
from gomoku.game import GameState
from gomoku.model import build_model, save_checkpoint

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


# --- parse helpers ---------------------------------------------------------
def test_parse_opening_variants():
    assert parse_opening("empty") is None
    assert parse_opening("initial") is None
    idx2 = parse_opening("idx2")
    assert isinstance(idx2, GameState) and idx2.move_count == 3
    custom = parse_opening("0,0;1,1")
    assert custom.move_count == 2


def test_parse_ruler_specs():
    specs = parse_ruler_specs(["self126=/x/a.pt", "rapfi=rapfi", "look=lookahead:depth=4"])
    assert specs == [("self126", "/x/a.pt"), ("rapfi", "rapfi"), ("look", "lookahead:depth=4")]
    with pytest.raises(ValueError):
        parse_ruler_specs(["no-equals-sign"])


def test_build_rulers_rapfi_runs_net_deterministic():
    rulers = build_rulers(
        [("rapfi", "rapfi"), ("look", "lookahead:depth=2")],
        n_games=8, sims=100, temp_until_ply=9, temperature=1.0, rapfi_timeout_ms=500,
    )
    by = {r.label: r for r in rulers}
    # Rapfi supplies its own variety -> net plays deterministically (temp 0).
    assert by["rapfi"].temp_until_ply == 0
    # Net-vs-baseline needs early-ply sampling for variety.
    assert by["look"].temp_until_ply == 9


# --- FastAPI service (baseline rulers, pool=None) --------------------------
@pytest.fixture
def app_client(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    save_checkpoint(ckpt, build_model("small"), epoch=3)
    rulers = build_rulers(
        [("look1", "lookahead:depth=1")],
        n_games=2, sims=6, temp_until_ply=99, temperature=1.0, rapfi_timeout_ms=500,
    )
    app = create_app(rulers=rulers, opening=None, pool=None)
    return TestClient(app), ckpt


def test_health(app_client):
    client, _ = app_client
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["pool_size"] == 0
    assert body["rulers"] == ["look1"]


def test_eval_endpoint_reports_white_split(app_client):
    client, ckpt = app_client
    r = client.post("/eval", json={
        "checkpoint": ckpt, "opponent": "lookahead:depth=1",
        "n_games": 2, "sims": 6, "opening": "empty", "temp_until_ply": 99,
    })
    assert r.status_code == 200
    row = r.json()
    assert row["n_games"] == 2
    assert "white_score" in row and "white_loss_rate" in row
    assert "black_score" in row


def test_panel_endpoint(app_client):
    client, ckpt = app_client
    r = client.post("/panel", json={"checkpoint": ckpt, "epoch": 3, "opening": "empty"})
    assert r.status_code == 200
    row = r.json()
    assert row["epoch"] == 3
    assert any(rr["ruler"] == "look1" for rr in row["rulers"])
    assert "look1__white_score" in row


def test_eval_missing_checkpoint_404(app_client):
    client, _ = app_client
    r = client.post("/eval", json={"checkpoint": "/no/such.pt", "opponent": "lookahead:depth=1", "n_games": 2})
    assert r.status_code == 404
