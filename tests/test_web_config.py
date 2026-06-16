"""The web UI must learn the board size from the server, not hardcode 9.

These tests pin the `/api/config` seam that `web/static/app.js` now consumes at
startup (issue #24). The JS render itself is mechanical and has no in-repo test
harness, so we prove the server half: the endpoint reports the *live*,
process-level board size resolved by `gomoku.board_config`.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from fastapi.testclient import TestClient

import gomoku.board_config as board_config
from web.server import create_app


def test_config_reports_live_board_size(tmp_path):
    """/api/config returns the board size this process was started with."""
    app = create_app(checkpoints_dir=str(tmp_path))
    client = TestClient(app)

    resp = client.get("/api/config")
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["board_size"] == board_config.BOARD_SIZE
    # n_actions is board_size**2; expose it too so the JS could size policy arrays.
    assert payload["n_actions"] == board_config.BOARD_SIZE ** 2


def test_config_reflects_15x15_in_a_fresh_process():
    """With GOMOKU_BOARD_SIZE=15, the endpoint reports 15 — not the default 9.

    Board size is locked at import time (board_config), so a 15x15 server must be
    a *fresh* process. Running this in a subprocess is what makes the assertion
    non-tautological: it proves the endpoint tracks the live size rather than a
    baked-in constant. (The current test process is 9x9, so an in-process check
    could not distinguish "reports BOARD_SIZE" from "always returns 9".)
    """
    script = textwrap.dedent(
        """
        import json
        from fastapi.testclient import TestClient
        from web.server import create_app
        import gomoku.board_config as bc

        client = TestClient(create_app(checkpoints_dir="."))
        cfg = client.get("/api/config").json()
        # Sanity: this child genuinely came up as a 15x15 process.
        assert bc.BOARD_SIZE == 15, bc.BOARD_SIZE
        print(json.dumps(cfg))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"GOMOKU_BOARD_SIZE": "15", "PATH": __import__("os").environ.get("PATH", "")},
    )
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    cfg = __import__("json").loads(proc.stdout.strip().splitlines()[-1])
    assert cfg["board_size"] == 15
    assert cfg["n_actions"] == 225
