"""Tests for the autolab monitor digest (epic #53, P6/P7).

Pure read-of-durable-state: build synthetic ledger rows into a tmp ledger,
point AUTOLAB_HOME at tmp_path, fold + probe, and assert the digest renders.
GPU-free and offline: osascript is mocked (the only subprocess), no torch,
no network. The script lives in scripts/ (not a package) so it is imported by
absolute path — from THIS worktree, which is what we want to exercise.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

from gomoku.lab import daemon as D
from gomoku.lab import ledger as L

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")


def _load_monitor():
    """Import scripts/autolab_monitor.py by path (scripts/ is not a package)."""
    path = os.path.join(_SCRIPTS, "autolab_monitor.py")
    spec = importlib.util.spec_from_file_location("autolab_monitor", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


M = _load_monitor()


# ---- synthetic ledger builders ------------------------------------------


def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    return D.default_ledger_path()


def _seed_train(path, lane="9x9-champ-recipe", cell="derby-v9-small",
                seq_n=0, priority=10):
    """Append an open train experiment (the seed/continuation shape)."""
    L.append(path, L.experiment(
        id=f"{lane}@{seq_n}", role="train", commit="deadbeef", base="scratch",
        config={"lane": lane, "cell": cell, "max_wall_secs": 3600, "seq_n": seq_n},
        priority=priority, note="seed"))


def _done_train(path, lane="9x9-champ-recipe", cell="derby-v9-small",
                seq_n=0, elo=1200.0, epochs=1000, wall_s=3600.0, priority=10):
    """Append a train experiment AND its done result (a finished slice)."""
    rid = f"{lane}@{seq_n}"
    L.append(path, L.experiment(
        id=rid, role="train", commit="deadbeef", base="scratch",
        config={"lane": lane, "cell": cell, "seq_n": seq_n}, priority=priority))
    L.append(path, L.result(
        rid, status=L.DONE, model=f"local://runs/{lane}/latest.pt",
        metrics={"eval/model_elo": elo, "epochs_ran": epochs, "lane": lane,
                 "cell": cell}, wall_s=wall_s))
    return rid


def _eval(path, ev_id="ev-eval-slice0", model="hf://repo@rev", gate="PROMOTE",
          win_rate=0.62, ci=(0.51, 0.73), n=40, promoted=True, vs="(first champion)"):
    L.append(path, L.eval_row(
        ev_id, model=model, panel=[vs, "candidate"],
        metrics={"verdict": gate, "win_rate": win_rate, "ci": list(ci),
                 "n_games": n, "promoted": promoted, "vs": vs}))
    L.append(path, L.verdict(
        ev_id, gate=gate, win_rate=win_rate, ci=list(ci), n=n,
        note="promoted champion" if promoted else ""))


def _fold(path):
    return L.fold(L.read_all(path))


def _daemons(train=False, arena=False):
    return {"train": {"alive": train, "meta": {}},
            "arena": {"alive": arena, "meta": {}}}


# ---- four-section presence ----------------------------------------------


def test_digest_has_all_four_sections(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1180.0)
    _done_train(path, seq_n=1, elo=1240.0)
    _eval(path)
    digest = M.build_digest(_fold(path), _daemons(train=True, arena=True))
    assert "## Researcher" in digest
    assert "## Lanes (tickets)" in digest
    assert "## Training" in digest
    assert "## Evals" in digest
    # header one-liner
    assert digest.splitlines()[0].startswith("# Autolab digest")
    assert "slice 9x9-champ-recipe@1" in digest
    assert "train ALIVE" in digest and "arena ALIVE" in digest


def test_researcher_section_no_notes(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path)
    digest = M.build_digest(_fold(path), _daemons())
    assert "(no notes yet)" in digest


def test_researcher_section_reads_note(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path)
    note = tmp_path / "research" / "latest.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# Autolab researcher — current thinking\n\n"
                    "> ⚠️ HONESTY: ranks on PROXIES (#61), not a verdict.\n")
    digest = M.build_digest(_fold(path), _daemons())
    assert "HONESTY: ranks on PROXIES" in digest


def test_events_surface_preflight_defer(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _seed_train(path)
    L.append(path, L.event(scope="train",
                           summary="preflight deferred: foreign GPU tenant(s): vibe",
                           data={"item": "9x9-champ-recipe@0"}))
    digest = M.build_digest(_fold(path), _daemons())
    assert "preflight deferred" in digest


# ---- None-elo safety (Risk #5: the most likely crash) -------------------


def test_none_elo_renders_dash_no_raise(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    rid = f"9x9-champ-recipe@0"
    L.append(path, L.experiment(id=rid, role="train", base="scratch",
                                config={"lane": "9x9-champ-recipe",
                                        "cell": "derby-v9-small", "seq_n": 0},
                                priority=10))
    # a DONE slice whose eval produced NO model_elo (None) — Risk #5
    L.append(path, L.result(rid, status=L.DONE,
                            metrics={"eval/model_elo": None, "epochs_ran": 5,
                                     "lane": "9x9-champ-recipe",
                                     "cell": "derby-v9-small"}, wall_s=42.0))
    digest = M.build_digest(_fold(path), _daemons())  # must not raise
    assert "elo —" in digest                      # header renders em-dash
    # and the Δelo is omitted (no None arithmetic)
    assert "Δ=—" in digest


def test_single_point_delta_omitted(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1200.0)        # exactly one data point
    digest = M.build_digest(_fold(path), _daemons())
    # one point => delta is undefined => em-dash, NOT (+0)
    assert "(—)" in digest
    assert "(+0)" not in digest


def test_two_point_delta_correct_sign(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1100.0)
    _done_train(path, seq_n=1, elo=1175.0)        # +75
    state = _fold(path)
    assert M._lane_delta(state, "9x9-champ-recipe") == pytest.approx(75.0)
    digest = M.build_digest(state, _daemons())
    assert "Δ=+75" in digest
    assert "(+75)" in digest                       # header carries it too


def test_two_point_delta_negative_sign(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1300.0)
    _done_train(path, seq_n=1, elo=1240.0)        # -60
    state = _fold(path)
    assert M._lane_delta(state, "9x9-champ-recipe") == pytest.approx(-60.0)
    digest = M.build_digest(state, _daemons())
    assert "Δ=-60" in digest


def test_empty_ledger_warms_up_no_raise(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)            # nothing appended
    digest = M.build_digest(_fold(path), _daemons())   # must not raise
    assert "## Researcher" in digest
    assert "first slice in flight" in digest
    assert "elo —" in digest


# ---- osascript escaping --------------------------------------------------


def test_escape_quote_and_backslash():
    raw = 'win 5\\3 say "hi"\nthere'
    out = M._escape(raw)
    assert "\n" not in out                          # newlines stripped
    assert '\\"' in out                             # double-quote escaped
    assert "\\\\" in out                            # backslash escaped
    # backslash escaped BEFORE quote (no double-escaping of the escape)
    assert out == 'win 5\\\\3 say \\"hi\\" there'


def test_notification_string_shape(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1100.0)
    _done_train(path, seq_n=1, elo=1175.0)
    _eval(path, gate="PROMOTE")
    body = M.build_notification(_fold(path), _daemons(train=True))
    assert body.startswith("Autolab · slice 9x9-champ-recipe@1")
    assert "elo 1175 (+75)" in body
    assert "eval:PROMOTE" in body


def test_notify_guarded_and_escapes(monkeypatch):
    calls = []

    def fake_runner(cmd, **kw):
        calls.append(cmd)
        class R:  # noqa: D401
            returncode = 0
        return R()

    # force osascript "present" so the body is built + dispatched
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/osascript")
    fired = M.notify('he said "yo" \\ done', runner=fake_runner)
    assert fired is True
    assert calls and calls[0][0] == "osascript"
    script = calls[0][2]
    assert '\\"yo\\"' in script                      # quote escaped inside the AppleScript
    assert "display notification" in script


def test_notify_noop_without_osascript(monkeypatch):
    monkeypatch.setattr(M.shutil, "which", lambda name: None)
    calls = []
    fired = M.notify("anything", runner=lambda *a, **k: calls.append(a))
    assert fired is False
    assert calls == []


# ---- write + notify-on-change diff --------------------------------------


def test_write_digest_atomic_and_log_append(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1200.0)
    state = _fold(path)
    daemons = _daemons(train=True)
    digest = M.build_digest(state, daemons)
    line = M.build_log_line(state, daemons)
    M.write_digest(digest, line)
    assert M.latest_md_path().exists()
    assert M.latest_md_path().read_text().startswith("# Autolab digest")
    log_txt = M.log_md_path().read_text()
    assert log_txt.count("\n") == 1                  # exactly one line appended
    assert "slice 9x9-champ-recipe@0" in log_txt


def test_notify_only_on_state_change(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1200.0)
    _eval(path, gate="PROMOTE")
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/osascript")

    fires = []
    runner = lambda cmd, **kw: fires.append(cmd) or type("R", (), {"returncode": 0})()

    # tick 1: there IS a prior baseline write first so the first ping is gated...
    # simulate the "baseline" tick (no prior log line) -> writes but does NOT ping
    M.run_once(do_notify=True, do_print=False, notify_runner=runner)
    assert fires == []                               # baseline tick: no notify

    # tick 2: SAME state again -> header unchanged -> still no notify
    M.run_once(do_notify=True, do_print=False, notify_runner=runner)
    assert fires == []                               # unchanged -> no second notify

    # now change the state (a REVERT eval) -> header changes -> notify fires once
    _eval(path, ev_id="ev-eval-slice1", gate="REVERT", promoted=False)
    M.run_once(do_notify=True, do_print=False, notify_runner=runner)
    assert len(fires) == 1                            # exactly one ping on change


def test_always_notify_overrides_gate(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1200.0)
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/osascript")
    fires = []
    runner = lambda cmd, **kw: fires.append(cmd) or type("R", (), {"returncode": 0})()
    # even the baseline tick pings under --always-notify
    M.run_once(do_notify=True, always_notify=True, do_print=False, notify_runner=runner)
    assert len(fires) == 1


def test_no_notify_suppresses(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1200.0)
    _eval(path, ev_id="ev-x", gate="PROMOTE")
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/osascript")
    fires = []
    runner = lambda cmd, **kw: fires.append(cmd) or type("R", (), {"returncode": 0})()
    M.run_once(do_notify=True, do_print=False, notify_runner=runner)   # baseline
    _eval(path, ev_id="ev-y", gate="REVERT", promoted=False)           # change
    M.run_once(do_notify=False, do_print=False, notify_runner=runner)  # suppressed
    assert fires == []


def test_plies_trend_from_log(tmp_path, monkeypatch):
    path = _home(tmp_path, monkeypatch)
    _done_train(path, seq_n=0, elo=1100.0, cell="derby-v9-small")
    _done_train(path, seq_n=1, elo=1175.0, cell="derby-v9-small")
    log = (tmp_path / "runs" / "9x9-champ-recipe" / "sweep_logs"
           / "derby-v9-small" / "trainer.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("epoch=1 gen=20 train=0.5 plies=18.4 buf=1000\n"
                   "epoch=2 gen=20 train=0.4 plies=15.1 buf=2000\n"
                   "epoch=3 gen=20 train=0.3 plies=12.7 buf=3000\n")
    digest = M.build_digest(_fold(path), _daemons(train=True))
    assert "plies:" in digest
    assert "collapse-watch" in digest                # falling trend flagged
