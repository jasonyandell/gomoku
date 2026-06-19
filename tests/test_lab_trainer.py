"""Tests for the autolab trainer role (epic #53, P3). run_sweep/git/HF mocked."""
from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import pytest

from gomoku.lab import daemon as D
from gomoku.lab import ledger as L
from gomoku.lab import trainer as T


def _fake_run(*, sweep_rc=0, make_latest=True):
    """A subprocess.run that fakes `git worktree` + `run_sweep` (creates output)."""
    calls = []

    def run(cmd, **kw):
        calls.append((list(map(str, cmd)), kw))
        if cmd[0] == "git" and "add" in cmd:
            work = Path(cmd[cmd.index("--detach") + 1])
            (work / "scripts").mkdir(parents=True, exist_ok=True)
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[0] == "git" and "remove" in cmd:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if any(str(c).endswith("run_sweep.py") for c in cmd):
            rb = Path(kw["env"]["GOMOKU_RUN_DIR"])
            ck = rb / "sweep_runs" / "SMOKE-bundle-plumbing" / "checkpoints"
            ck.mkdir(parents=True, exist_ok=True)
            if make_latest:
                (ck / "latest.pt").write_text("x")
                (ck / "epoch0000.pt").write_text("x")
                (ck / "eval_results.jsonl").write_text(
                    json.dumps({"eval/model_elo": 1234.0}) + "\n")
            return types.SimpleNamespace(returncode=sweep_rc)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    run.calls = calls
    return run


def _role(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    return T.TrainerRole(repo_root=tmp_path, check_tenants=False, **kw)


def _item(**kw):
    base = dict(id="mvp@0", role="train", commit="abc123", base="scratch",
                config={"cell": "SMOKE", "lane": "mvp"}, priority=50, status="open")
    base.update(kw)
    return base


# ---- the happy path -----------------------------------------------------

def test_run_chunk_dry_run_offline(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch, dry_run=True)
    fake = _fake_run()
    monkeypatch.setattr(T.subprocess, "run", fake)
    res = role.run_chunk(_item())
    assert res.status == L.DONE
    assert res.artifact_ref.startswith("local://") and res.artifact_ref.endswith("latest.pt")
    assert res.metrics["eval/model_elo"] == 1234.0
    assert res.metrics["epochs_ran"] == 1
    assert res.buffer.endswith("latest.pt")
    # followups: continuation (train) + arena eval
    cont, ev = res.followups
    assert cont["id"] == "mvp@1" and cont["role"] == "train"
    assert cont["base"].startswith("local://") and cont["base"].endswith("latest.pt")
    assert cont["config"]["seq_n"] == 1
    assert ev["id"] == "eval-mvp@0" and ev["role"] == "arena" and ev["base"] == res.artifact_ref
    # teardown happened
    assert any(c[0][:1] == ["git"] and "remove" in c[0] for c in fake.calls)


def test_run_sweep_argv_and_env(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch, dry_run=True)
    fake = _fake_run()
    monkeypatch.setattr(T.subprocess, "run", fake)
    role.run_chunk(_item(base="local:///somewhere/latest.pt"))
    sweep = next(c for c in fake.calls if any(x.endswith("run_sweep.py") for x in c[0]))
    argv, kw = sweep
    assert "--cell" in argv and "SMOKE" in argv
    assert "--final-eval" in argv and "--foreground" in argv
    assert "--max-wall-secs" in argv and argv[argv.index("--max-wall-secs") + 1] == "1.0"
    assert "--resume" in argv and argv[argv.index("--resume") + 1] == "/somewhere/latest.pt"
    assert kw["env"]["GOMOKU_RUN_DIR"].endswith("/runs/mvp")


def test_board_size_threads_into_sweep_env(tmp_path, monkeypatch):
    """config board_size=15 → GOMOKU_BOARD_SIZE=15 in the run_sweep subprocess env."""
    role = _role(tmp_path, monkeypatch, dry_run=True)
    fake = _fake_run()
    monkeypatch.setattr(T.subprocess, "run", fake)
    role.run_chunk(_item(config={"cell": "SMOKE", "lane": "mvp", "board_size": 15}))
    sweep = next(c for c in fake.calls if any(x.endswith("run_sweep.py") for x in c[0]))
    assert sweep[1]["env"]["GOMOKU_BOARD_SIZE"] == "15"


def test_no_board_size_does_not_force_env(tmp_path, monkeypatch):
    """Without board_size, GOMOKU_BOARD_SIZE is not forced (native 9x9)."""
    monkeypatch.delenv("GOMOKU_BOARD_SIZE", raising=False)
    role = _role(tmp_path, monkeypatch, dry_run=True)
    fake = _fake_run()
    monkeypatch.setattr(T.subprocess, "run", fake)
    role.run_chunk(_item())  # default config has no board_size
    sweep = next(c for c in fake.calls if any(x.endswith("run_sweep.py") for x in c[0]))
    assert "GOMOKU_BOARD_SIZE" not in sweep[1]["env"]


def test_prod_mode_uses_1h_cap(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch, dry_run=True, mvp_mode=False)
    fake = _fake_run()
    monkeypatch.setattr(T.subprocess, "run", fake)
    role.run_chunk(_item())
    sweep = next(c for c in fake.calls if any(x.endswith("run_sweep.py") for x in c[0]))
    argv = sweep[0]
    assert argv[argv.index("--max-wall-secs") + 1] == "3600.0"


def test_checkout_uses_detach_and_commit(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch, dry_run=True)
    fake = _fake_run()
    monkeypatch.setattr(T.subprocess, "run", fake)
    role.run_chunk(_item(commit="deadbeef"))
    add = next(c for c in fake.calls if c[0][0] == "git" and "add" in c[0])
    assert "--detach" in add[0] and add[0][-1] == "deadbeef"


# ---- HF delivery --------------------------------------------------------

def test_run_chunk_pushes_slice(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch, dry_run=False)
    monkeypatch.setattr(T.subprocess, "run", _fake_run())
    pushed = {}

    def fake_push(latest, *, repo_id, revision, provenance, tag=None):
        pushed.update(revision=revision, provenance=provenance, repo_id=repo_id)
        return f"{repo_id}@{revision}"

    monkeypatch.setattr(T.hf, "push_slice", fake_push)
    res = role.run_chunk(_item())
    assert pushed["revision"] == "mvp-mvp@0"
    assert pushed["provenance"]["git_sha"] == "abc123"
    assert pushed["provenance"]["model_elo"] == 1234.0
    assert res.artifact_ref == f"{pushed['repo_id']}@mvp-mvp@0"


# ---- failure modes ------------------------------------------------------

def test_no_latest_raises_and_tears_down(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch, dry_run=True)
    fake = _fake_run(make_latest=False)
    monkeypatch.setattr(T.subprocess, "run", fake)
    with pytest.raises(T.SliceFailed):
        role.run_chunk(_item())
    assert any(c[0][0] == "git" and "remove" in c[0] for c in fake.calls)  # teardown ran


def test_run_sweep_nonzero_raises_and_tears_down(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch, dry_run=True)
    fake = _fake_run(sweep_rc=1)
    monkeypatch.setattr(T.subprocess, "run", fake)
    with pytest.raises(T.SliceFailed):
        role.run_chunk(_item())
    assert any(c[0][0] == "git" and "remove" in c[0] for c in fake.calls)


def test_checkout_failure_raises(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch, dry_run=True)

    def bad_run(cmd, **kw):
        if cmd[0] == "git" and "add" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="no such commit")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(T.subprocess, "run", bad_run)
    with pytest.raises(T.SliceFailed):
        role.run_chunk(_item(commit="nope"))


# ---- base → resume ------------------------------------------------------

def test_resolve_base(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch)
    assert role._resolve_base({"base": "scratch"}) is None
    assert role._resolve_base({}) is None
    assert role._resolve_base({"base": "local:///a/b/latest.pt"}) == "/a/b/latest.pt"


def test_resolve_base_hf_downloads(tmp_path, monkeypatch):
    role = _role(tmp_path, monkeypatch)
    import huggingface_hub
    seen = {}

    def fake_dl(repo, name, revision=None):
        seen.update(repo=repo, name=name, revision=revision)
        return "/cache/model.pt"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    out = role._resolve_base({"base": "hf://jasonyandell/gomoku-9x9@mvp-mvp@0"})
    assert out == "/cache/model.pt"
    assert seen["repo"] == "jasonyandell/gomoku-9x9" and seen["revision"] == "mvp-mvp@0"


# ---- preflight ----------------------------------------------------------

def test_preflight_defers_on_foreign_tenant(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    role = T.TrainerRole(repo_root=tmp_path, check_tenants=True)
    monkeypatch.setattr(T.subprocess, "check_output",
                        lambda *a, **k: "9999 gomoku.train --some other run\n")
    with pytest.raises(D.PreflightDeferred):
        role.preflight()


def test_preflight_passes_when_box_free(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    role = T.TrainerRole(repo_root=tmp_path, check_tenants=True)

    def raise_called(*a, **k):
        raise subprocess.CalledProcessError(1, "pgrep")  # nothing matched

    monkeypatch.setattr(T.subprocess, "check_output", raise_called)
    role.preflight()  # should not raise
