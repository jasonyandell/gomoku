"""Unit tests for the worktree reclamation janitor's pure logic."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import reclaim_worktrees as rw  # noqa: E402

REPO = "/Users/jason/code/gomoku"
AGENT = "/.claude/worktrees/"
EXTERNAL = (os.path.expanduser("~/.codex/"),)

PORCELAIN = """\
worktree /Users/jason/code/gomoku
branch refs/heads/main

worktree /Users/jason/.codex/worktrees/08c7/gomoku
detached

worktree /Users/jason/code/gomoku-perf-L09i-fix
branch refs/heads/feat/perf-L09i-fix

worktree /Users/jason/code/gomoku/.claude/worktrees/agent-dead
branch refs/heads/feat/v3-gumbel
locked claude agent agent-dead (pid 67879)

worktree /Users/jason/code/gomoku/.claude/worktrees/agent-live
branch refs/heads/feat/derby-v2-engine
locked claude agent agent-live (pid 11684)

worktree /Users/jason/code/gomoku/.claude/worktrees/agent-nolock
branch refs/heads/worktree-agent-nolock
"""


def test_parse_worktrees_roundtrip():
    wts = rw.parse_worktrees(PORCELAIN)
    assert len(wts) == 6
    assert wts[0].branch == "main"
    assert wts[1].detached is True and wts[1].branch is None
    assert wts[3].branch == "feat/v3-gumbel"
    assert wts[3].lock_reason == "claude agent agent-dead (pid 67879)"
    assert wts[5].lock_reason is None


def test_parse_lock_pid():
    assert rw.parse_lock_pid("claude agent agent-x (pid 67879)") == 67879
    assert rw.parse_lock_pid("(no reason)") is None
    assert rw.parse_lock_pid(None) is None


def test_classify_dead_pid_is_reclaimed():
    wts = {w.path.split("/")[-1]: w for w in rw.parse_worktrees(PORCELAIN)}
    alive = lambda pid: pid == 11684  # only the live session
    disp, _ = rw.classify_worktree(wts["agent-dead"], REPO, AGENT, EXTERNAL, alive)
    assert disp == rw.RECLAIM


def test_classify_live_pid_is_kept():
    wts = {w.path.split("/")[-1]: w for w in rw.parse_worktrees(PORCELAIN)}
    alive = lambda pid: pid == 11684
    disp, why = rw.classify_worktree(wts["agent-live"], REPO, AGENT, EXTERNAL, alive)
    assert disp == rw.KEEP_LIVE
    assert "ALIVE" in why


def test_classify_external_and_manual_and_main():
    wts = rw.parse_worktrees(PORCELAIN)
    alive = lambda pid: False
    by_path = {w.path: w for w in wts}
    assert rw.classify_worktree(by_path["/Users/jason/code/gomoku"],
                                REPO, AGENT, EXTERNAL, alive)[0] == rw.KEEP_MAIN
    assert rw.classify_worktree(by_path["/Users/jason/.codex/worktrees/08c7/gomoku"],
                                REPO, AGENT, EXTERNAL, alive)[0] == rw.KEEP_EXTERNAL
    assert rw.classify_worktree(by_path["/Users/jason/code/gomoku-perf-L09i-fix"],
                                REPO, AGENT, EXTERNAL, alive)[0] == rw.KEEP_MANUAL


def test_classify_nolock_agent_worktree_reclaimed():
    wts = {w.path.split("/")[-1]: w for w in rw.parse_worktrees(PORCELAIN)}
    disp, _ = rw.classify_worktree(wts["agent-nolock"], REPO, AGENT, EXTERNAL,
                                   lambda pid: False)
    assert disp == rw.RECLAIM


# ---------------------------------------------------------------------------
# derby-o3s — gauge must NEVER wedge on a blocking git subprocess.
#
# Root cause hypothesis: a git subprocess (worktree list / branch --merged)
# blocked indefinitely on a credential prompt, a stale lock, or an implicit
# pager — observed wedging the invoking shell for ~2min on 2026-05-28, even
# under external `timeout 30`. Fix is preventative:
#
#   * stdin=DEVNULL on every git call (no inherited stdin to wait on),
#   * GIT_TERMINAL_PROMPT=0 + GIT_ASKPASS=/bin/true (no credential prompt),
#   * --no-pager on every invocation (no implicit pager),
#   * per-call subprocess.run(timeout=...) (bounds the rest),
#   * hard internal wall budget around the gauge path with degraded output
#     ("[partial: ...]") so the metric is *always* reported.
#
# These tests assert the gauge returns FAST even when subprocesses hang.
# ---------------------------------------------------------------------------


def test_git_run_is_noninteractive(monkeypatch):
    """_git_run must invoke git with --no-pager, stdin=DEVNULL,
    GIT_TERMINAL_PROMPT=0 — the three blocking surfaces closed."""
    captured = {}

    class _CP:
        def __init__(self):
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["stdin"] = kwargs.get("stdin")
        captured["env"] = kwargs.get("env")
        captured["timeout"] = kwargs.get("timeout")
        return _CP()

    monkeypatch.setattr(rw.subprocess, "run", fake_run)
    rw._git_run(["worktree", "list", "--porcelain"], cwd="/tmp", timeout=5.0)

    assert captured["cmd"][0] == "git"
    assert "--no-pager" in captured["cmd"], "git must run with --no-pager"
    assert captured["stdin"] is subprocess.DEVNULL, "stdin must be DEVNULL"
    assert captured["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert captured["env"]["GIT_ASKPASS"] == "/bin/true"
    assert captured["timeout"] == 5.0


def test_git_run_raises_git_timeout_on_subprocess_timeout(monkeypatch):
    """A TimeoutExpired from subprocess.run becomes a typed GitTimeoutError —
    callers (gauge) catch this to emit a degraded line, not propagate."""
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(rw.subprocess, "run", fake_run)

    try:
        rw._git_run(["branch"], cwd="/tmp", timeout=0.5)
    except rw.GitTimeoutError as exc:
        assert exc.timeout == 0.5
    else:
        raise AssertionError("expected GitTimeoutError")


def test_gauge_returns_fast_even_when_every_git_call_hangs(monkeypatch):
    """The derby-o3s acceptance test: simulate every git call hanging for 30s
    and assert gauge returns in well under 15s with a degraded `[partial:]`
    line. No real subprocess work — we patch subprocess.run to always
    raise TimeoutExpired, which is what bounded subprocess timeouts produce
    when a child wedges."""
    class _CP:
        def __init__(self, stdout=""):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def always_times_out(cmd, **kwargs):
        # subprocess.run(timeout=N) raises TimeoutExpired when N elapses; we
        # raise immediately to model the *budgeted* worst case (the real
        # 30s hang would be capped at PER_CALL_TIMEOUT_S anyway).
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 30))

    monkeypatch.setattr(rw.subprocess, "run", always_times_out)

    t0 = time.monotonic()
    line = rw.gauge(REPO, AGENT, EXTERNAL, budget_s=15.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 15.0, f"gauge took {elapsed:.2f}s, must be <15s"
    assert "repo-hygiene:" in line, f"gauge must always emit the metric line, got: {line}"
    assert "[partial:" in line, f"degraded output marker missing: {line}"
    assert "⚠" in line, "degraded gauge must flag for human attention"


def test_gauge_degrades_on_one_call_timing_out(monkeypatch):
    """If only the SECOND git call hangs (worktree list works, branch hangs),
    the gauge still reports the worktree counts it managed to collect and
    only marks the branch half degraded."""
    porcelain = (
        "worktree /Users/jason/code/gomoku\n"
        "branch refs/heads/main\n"
    )
    call_count = {"n": 0}

    class _CP:
        def __init__(self, stdout=""):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def selective_hang(cmd, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # worktree list --porcelain succeeds
            return _CP(stdout=porcelain)
        # everything else times out
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 30))

    monkeypatch.setattr(rw.subprocess, "run", selective_hang)

    t0 = time.monotonic()
    line = rw.gauge(REPO, AGENT, EXTERNAL, budget_s=15.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 15.0
    assert "worktrees=1" in line, (
        f"first-call data must survive a later timeout, got: {line}"
    )
    assert "branches=?" in line, (
        f"second-call timeout must show as `?`, got: {line}"
    )
    assert "[partial:" in line


def test_gauge_clean_path_does_not_degrade(monkeypatch):
    """The happy path — all git calls return fast, no `[partial:]` marker."""
    porcelain = (
        "worktree /Users/jason/code/gomoku\n"
        "branch refs/heads/main\n"
    )

    class _CP:
        def __init__(self, stdout=""):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    outputs = iter([
        porcelain,           # build_plan: worktree list --porcelain
        "* main\n",          # git branch
        "* main\n",          # git branch --merged main
        porcelain,           # second worktree list (in gauge for checked-out set)
    ])

    def fast(cmd, **kwargs):
        return _CP(stdout=next(outputs, ""))

    monkeypatch.setattr(rw.subprocess, "run", fast)

    line = rw.gauge(REPO, AGENT, EXTERNAL, budget_s=15.0)
    assert "[partial:" not in line
    assert "clean" in line
    assert "worktrees=1" in line
    assert "branches=1" in line
