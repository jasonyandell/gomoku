"""Tests for the autolab supervisor (epic #53, P7). launchctl + git mocked.

GPU-free, offline. AUTOLAB_HOME + the LaunchAgents dir are redirected to tmp_path
so nothing touches the real machine; launchctl/git are injected via a recording
runner / commit override.
"""
from __future__ import annotations

import plistlib

import pytest

from gomoku.lab import daemon as D
from gomoku.lab import ledger as L
from gomoku.lab import up as U


# ---- a recording launchctl runner ---------------------------------------

class _Runner:
    """Records every argv; returns rc=0 by default (override per-prefix)."""

    def __init__(self, rc_for=None):
        self.calls = []
        self._rc_for = rc_for or {}

    def __call__(self, argv):
        import subprocess
        self.calls.append(list(argv))
        rc = 0
        for prefix, code in self._rc_for.items():
            if argv[:len(prefix)] == list(prefix):
                rc = code
        return subprocess.CompletedProcess(argv, rc, "", "")


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """Redirect AUTOLAB_HOME + the LaunchAgents dir into tmp_path."""
    h = tmp_path / "autolab"
    la = tmp_path / "LaunchAgents"
    monkeypatch.setenv("AUTOLAB_HOME", str(h))
    monkeypatch.setenv("AUTOLAB_LAUNCHD_DIR", str(la))
    return h


def _args(cmd, **kw):
    """A minimal argparse-Namespace stand-in for cmd_* (so we skip launchctl
    bind by passing our own runner where the cmd reads it)."""
    import types
    base = dict(cmd=cmd, ledger=None, launchd_dir=None, commit="cafef00d",
                dry_run=False, force=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


# ---- plist render shape -------------------------------------------------

def test_render_plists_labels_and_count(home):
    plists = U.render_plists()
    assert set(plists) == {
        "com.gomoku.autolab.train", "com.gomoku.autolab.arena",
        "com.gomoku.autolab.monitor", "com.gomoku.autolab.research"}
    assert tuple(plists) == U.LABELS  # order = load order (daemons first)


def test_daemon_plists_keepalive_and_program(home):
    plists = U.render_plists()
    for label in ("com.gomoku.autolab.train", "com.gomoku.autolab.arena"):
        doc = plists[label]
        # KeepAlive(SuccessfulExit=false) — respawn on crash, honor clean exit 0
        assert doc["KeepAlive"] == {"SuccessfulExit": False}
        assert "StartInterval" not in doc
        prog = doc["ProgramArguments"]
        assert prog[0] == U.VENV_PY
        assert prog[1] == "-m"
        assert doc["WorkingDirectory"] == U.WORKDIR
    train = plists["com.gomoku.autolab.train"]["ProgramArguments"]
    assert train[2] == "gomoku.lab.trainer" and "--prod" in train
    assert "--stop-file" in train and train[-1].endswith("/stop")
    arena = plists["com.gomoku.autolab.arena"]["ProgramArguments"]
    assert arena[2] == "gomoku.lab.arena" and "--prod" not in arena
    assert "--stop-file" in arena


def test_periodic_plists_startinterval_no_keepalive(home):
    plists = U.render_plists()
    mon = plists["com.gomoku.autolab.monitor"]
    assert mon["StartInterval"] == 600 and "KeepAlive" not in mon
    assert mon["ProgramArguments"][1].endswith("autolab_monitor.py")
    res = plists["com.gomoku.autolab.research"]
    assert res["StartInterval"] == 1800 and "KeepAlive" not in res
    prog = res["ProgramArguments"]
    assert prog[2] == "gomoku.lab.research" and "--once" in prog


def test_plist_env_has_home_autolab_wandb(home):
    plists = U.render_plists()
    # daemons carry the full env incl. WANDB_MODE=offline
    for label in ("com.gomoku.autolab.train", "com.gomoku.autolab.arena"):
        env = plists[label]["EnvironmentVariables"]
        assert env["HOME"] == "/Users/jason"
        assert env["AUTOLAB_HOME"] == str(home)        # resolved from AUTOLAB_HOME
        assert env["WANDB_MODE"] == "offline"
        assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        assert ".venv/bin" in env["PATH"] and "/opt/homebrew/bin" in env["PATH"]
    # periodics: minimal env (HOME + AUTOLAB_HOME for ledger + osascript), no MPS
    for label in ("com.gomoku.autolab.monitor", "com.gomoku.autolab.research"):
        env = plists[label]["EnvironmentVariables"]
        assert env["HOME"] == "/Users/jason"
        assert env["AUTOLAB_HOME"] == str(home)
        assert "WANDB_MODE" not in env


def test_write_plists_are_valid_and_roundtrip(home):
    written = U.write_plists()
    assert set(written) == set(U.LABELS)
    for label, path in written.items():
        with open(path, "rb") as f:
            doc = plistlib.load(f)
        assert doc["Label"] == label   # parseable, correct label


# ---- seed: pins a real commit, idempotent -------------------------------

def test_seed_pins_non_null_commit(home):
    path = D.default_ledger_path()
    row = U.seed_ledger(path, commit="abc123def")
    assert row is not None
    assert row["id"] == "9x9-champ-recipe@0"
    assert row["commit"] == "abc123def"           # NOT null
    assert row["base"] == "scratch"
    assert row["priority"] == 10
    assert row["config"] == {"lane": "9x9-champ-recipe", "cell": "derby-v9-small",
                             "max_wall_secs": 3600, "seq_n": 0}


def test_seed_resolves_head_via_injected_git(home):
    path = D.default_ledger_path()
    row = U.seed_ledger(path, git_head=lambda: "deadbeefSHA")
    assert row["commit"] == "deadbeefSHA"


def test_seed_idempotent_no_double_seed(home):
    path = D.default_ledger_path()
    first = U.seed_ledger(path, commit="sha1")
    assert first is not None
    second = U.seed_ledger(path, commit="sha2")
    assert second is None                          # never double-seed
    # exactly one experiment row in the ledger
    rows = L.read_all(path)
    exps = [r for r in rows if r.get("type") == L.EXPERIMENT]
    assert len(exps) == 1


def test_seed_pick_train_returns_the_seed(home):
    path = D.default_ledger_path()
    U.seed_ledger(path, commit="sha1")
    state = L.fold(L.read_all(path))
    picked = state.pick("train", L.utcnow())
    assert picked is not None and picked["id"] == "9x9-champ-recipe@0"


def test_seed_skips_when_open_train_exists(home):
    """A pre-existing open train lane (e.g. a research continuation) blocks seed."""
    path = D.default_ledger_path()
    L.append(path, L.experiment(id="other@0", role="train", priority=10))
    assert U.seed_ledger(path, commit="sha1") is None
    exps = [r for r in L.read_all(path) if r.get("type") == L.EXPERIMENT]
    assert len(exps) == 1 and exps[0]["id"] == "other@0"


def test_seed_when_only_done_train_exists(home):
    """A finished (done) train lane does NOT count as open → seed proceeds."""
    path = D.default_ledger_path()
    L.append(path, L.experiment(id="old@0", role="train", priority=10))
    L.append(path, L.result("old@0", status=L.DONE))
    row = U.seed_ledger(path, commit="sha1")
    assert row is not None and row["id"] == "9x9-champ-recipe@0"


# ---- bootout-before-bootstrap argv order --------------------------------

def test_load_agent_bootout_before_bootstrap(home):
    r = _Runner()
    ran = U.load_agent("com.gomoku.autolab.train", runner=r)
    assert len(r.calls) == 2
    bootout, bootstrap = r.calls
    assert bootout[:2] == ["launchctl", "bootout"]
    assert bootout[2].endswith("/com.gomoku.autolab.train")
    assert bootstrap[:2] == ["launchctl", "bootstrap"]
    assert bootstrap[2].startswith("gui/")
    assert bootstrap[3].endswith("com.gomoku.autolab.train.plist")
    assert ran == r.calls   # returns what it ran


def test_load_agent_survives_bootout_failure(home):
    """bootout of a not-loaded agent returns nonzero — we must still bootstrap."""
    r = _Runner(rc_for={("launchctl", "bootout"): 1})
    U.load_agent("com.gomoku.autolab.arena", runner=r)
    assert any(c[:2] == ["launchctl", "bootstrap"] for c in r.calls)


def test_unload_agent_argv(home):
    r = _Runner()
    argv = U.unload_agent("com.gomoku.autolab.monitor", runner=r)
    assert argv[:2] == ["launchctl", "bootout"]
    assert argv[2].endswith("/com.gomoku.autolab.monitor")
    assert r.calls == [argv]


# ---- stop-file lifecycle ------------------------------------------------

def test_up_removes_stop_file(home, monkeypatch):
    # plant a stale stop-file (a prior `down` left it)
    import os
    os.makedirs(str(home), exist_ok=True)
    sp = U.stop_file_path()
    with open(sp, "w") as f:
        f.write("stale\n")
    assert os.path.exists(sp)
    # patch the launchctl runner so up touches no real launchctl
    r = _Runner()
    monkeypatch.setattr(U, "_real_runner", r)
    U.cmd_up(_args("up"))
    assert not os.path.exists(sp)   # up removed it

def test_down_creates_stop_file(home, monkeypatch):
    import os
    r = _Runner()
    monkeypatch.setattr(U, "_real_runner", r)
    U.cmd_down(_args("down"))
    assert os.path.exists(U.stop_file_path())   # down wrote it


def test_clear_stop_file_missing_is_noop(home):
    # no stop-file present → no error
    U.clear_stop_file()


# ---- cmd_up wiring: makedirs + seed + load all four ---------------------

def test_cmd_up_makes_home_subdirs_and_loads_all(home, monkeypatch):
    import os
    r = _Runner()
    monkeypatch.setattr(U, "_real_runner", r)
    rc = U.cmd_up(_args("up"))
    assert rc == 0
    for sub in U.HOME_SUBDIRS:
        assert os.path.isdir(os.path.join(str(home), sub))
    # the seed landed and pick(train) returns it
    state = L.fold(L.read_all(D.default_ledger_path()))
    assert state.pick("train", L.utcnow())["id"] == "9x9-champ-recipe@0"
    assert state.experiments["9x9-champ-recipe@0"]["commit"] == "cafef00d"
    # all four plists got bootstrapped
    bootstrapped = [c[3] for c in r.calls if c[:2] == ["launchctl", "bootstrap"]]
    for label in U.LABELS:
        assert any(p.endswith(f"{label}.plist") for p in bootstrapped)


def test_cmd_up_idempotent_second_run_no_double_seed(home, monkeypatch):
    r = _Runner()
    monkeypatch.setattr(U, "_real_runner", r)
    U.cmd_up(_args("up"))
    U.cmd_up(_args("up"))
    rows = L.read_all(D.default_ledger_path())
    exps = [x for x in rows if x.get("type") == L.EXPERIMENT]
    assert len(exps) == 1            # second up did not add a second seed row
    state = L.fold(rows)
    assert state.pick("train", L.utcnow())["id"] == "9x9-champ-recipe@0"


def test_cmd_down_bootouts_all_four(home, monkeypatch):
    r = _Runner()
    monkeypatch.setattr(U, "_real_runner", r)
    U.cmd_down(_args("down"))
    booted = [c[2] for c in r.calls if c[:2] == ["launchctl", "bootout"]]
    for label in U.LABELS:
        assert any(b.endswith(f"/{label}") for b in booted)


def test_cmd_down_force_pkill_scoped_to_worktrees(home, monkeypatch):
    r = _Runner()
    monkeypatch.setattr(U, "_real_runner", r)
    U.cmd_down(_args("down", force=True))
    pk = next((c for c in r.calls if c[0] == "pkill"), None)
    assert pk is not None
    assert "run_sweep.py" in pk[-1] and "worktrees" in pk[-1]


# ---- dry-run prints instead of executing --------------------------------

def test_dry_run_does_not_call_real_runner(home, monkeypatch, capsys):
    sentinel = _Runner()
    monkeypatch.setattr(U, "_real_runner", sentinel)
    U.cmd_up(_args("up", dry_run=True))
    assert sentinel.calls == []      # dry-run never touched the real runner
    out = capsys.readouterr().out
    assert "DRY-RUN: launchctl bootstrap" in out
