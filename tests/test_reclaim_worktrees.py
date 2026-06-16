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


# ---------------------------------------------------------------------------
# GitHub issue #47 — auto-reclaim clean+merged manual-sibling worktrees.
#
# A manual-sibling worktree (a sibling dir, NOT under .claude/worktrees/) is
# safe to reclaim ONLY when its work is provably already in main: the working
# tree is clean (no uncommitted AND no untracked changes) AND its branch is an
# ancestor of main. ANY dirt OR ANY unmerged commit => keep-by-hand.
#
# These tests build a REAL throwaway git repo + sibling worktrees and exercise
# the four cases from the issue against build_plan / is_stale_sibling / gauge.
# ---------------------------------------------------------------------------


def _git(cwd, *args):
    """Run a git command in cwd, raising on failure (test helper, not prod)."""
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True,
        capture_output=True, text=True,
    )


def _make_repo(tmp_path):
    """Create a real git repo with main + one commit; return its path (str)."""
    root = tmp_path / "repo"
    root.mkdir()
    root = str(root)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    (tmp_path / "repo" / "seed.txt").write_text("seed\n")
    _git(root, "add", "seed.txt")
    _git(root, "commit", "-m", "seed")
    return root


def _add_sibling(root, tmp_path, name, branch):
    """Create a sibling worktree (NOT under .claude/worktrees) on a new branch."""
    path = str(tmp_path / name)
    _git(root, "worktree", "add", "-b", branch, path, "main")
    return path


def _build(root):
    # agent_subdir set to the production token so a plain sibling dir (which
    # does not contain it) is classified KEEP_MANUAL, then refined.
    return rw.build_plan(root, "/.claude/worktrees/", ())


def test_issue47_clean_merged_sibling_is_reclaimed(tmp_path):
    """(a) clean + merged -> RECLAIM_STALE_SIBLING (new category)."""
    root = _make_repo(tmp_path)
    path = _add_sibling(root, tmp_path, "sib-clean-merged", "feat/clean-merged")
    # branch == main's commit, no edits => clean and an ancestor of main.

    wt = next(w for w in rw.parse_worktrees(
        rw._git_run(["worktree", "list", "--porcelain"], cwd=root).stdout)
        if w.path == path)
    assert rw.is_stale_sibling(wt, root) is True

    plan = _build(root)
    paths = {w.path for w, _ in plan.stale_siblings}
    assert path in paths, "clean+merged sibling must be a stale-sibling reclaim"
    assert path not in {w.path for w, _, _ in plan.keep}


def test_issue47_clean_unmerged_sibling_is_kept(tmp_path):
    """(b) clean + UNMERGED (unique commit not in main) -> KEPT."""
    root = _make_repo(tmp_path)
    path = _add_sibling(root, tmp_path, "sib-clean-unmerged", "feat/clean-unmerged")
    # Commit a unique change on the branch so it is NOT an ancestor of main.
    (tmp_path / "sib-clean-unmerged" / "extra.txt").write_text("work\n")
    _git(path, "add", "extra.txt")
    _git(path, "commit", "-m", "unique work not in main")

    wt = next(w for w in rw.parse_worktrees(
        rw._git_run(["worktree", "list", "--porcelain"], cwd=root).stdout)
        if w.path == path)
    assert rw.is_stale_sibling(wt, root) is False

    plan = _build(root)
    assert path not in {w.path for w, _ in plan.stale_siblings}
    assert path in {w.path for w, _, _ in plan.keep}


def test_issue47_dirty_merged_sibling_is_kept(tmp_path):
    """(c) DIRTY (uncommitted tracked change) + merged -> KEPT."""
    root = _make_repo(tmp_path)
    path = _add_sibling(root, tmp_path, "sib-dirty-merged", "feat/dirty-merged")
    # Modify the tracked seed file but do NOT commit => dirty working tree.
    (tmp_path / "sib-dirty-merged" / "seed.txt").write_text("seed\nlocal edit\n")

    wt = next(w for w in rw.parse_worktrees(
        rw._git_run(["worktree", "list", "--porcelain"], cwd=root).stdout)
        if w.path == path)
    assert rw.worktree_is_clean(path) is False
    assert rw.is_stale_sibling(wt, root) is False

    plan = _build(root)
    assert path not in {w.path for w, _ in plan.stale_siblings}
    assert path in {w.path for w, _, _ in plan.keep}


def test_issue47_untracked_only_merged_sibling_is_kept(tmp_path):
    """(d) untracked-only (a brand-new file) + merged -> KEPT."""
    root = _make_repo(tmp_path)
    path = _add_sibling(root, tmp_path, "sib-untracked", "feat/untracked")
    # Branch tip == main (merged) but an untracked file makes porcelain non-empty.
    (tmp_path / "sib-untracked" / "scratch.tmp").write_text("scratch\n")

    wt = next(w for w in rw.parse_worktrees(
        rw._git_run(["worktree", "list", "--porcelain"], cwd=root).stdout)
        if w.path == path)
    assert rw.worktree_is_clean(path) is False, "untracked file => not clean"
    assert rw.is_stale_sibling(wt, root) is False

    plan = _build(root)
    assert path not in {w.path for w, _ in plan.stale_siblings}
    assert path in {w.path for w, _, _ in plan.keep}


def test_issue47_secondary_main_worktree_is_never_stale(tmp_path):
    """Scope guard: a SECOND worktree checked out on `main` is never reclaimed.

    `main` is trivially an ancestor of itself, so without the explicit guard a
    clean secondary `main` checkout would be misread as a stale sibling. It
    must always be kept — the integration branch is not disposable."""
    root = _make_repo(tmp_path)
    path = str(tmp_path / "sib-main")
    # A linked worktree on main is unusual but legal via --force; assert the
    # branch-level guard regardless of how it was created.
    _git(root, "worktree", "add", "--force", path, "main")

    wt = next(w for w in rw.parse_worktrees(
        rw._git_run(["worktree", "list", "--porcelain"], cwd=root).stdout)
        if w.path == path)
    assert wt.branch == "main"
    assert rw.is_stale_sibling(wt, root) is False

    plan = _build(root)
    assert path not in {w.path for w, _ in plan.stale_siblings}


def test_issue47_gauge_counts_only_clean_merged_as_stale(tmp_path):
    """Gauge surfaces stale-siblings=N: only (a) counts; (b)/(c)/(d) are kept."""
    root = _make_repo(tmp_path)

    # (a) clean + merged -> the one reclaimable stale sibling.
    _add_sibling(root, tmp_path, "sib-a", "feat/a-clean-merged")

    # (b) clean + unmerged.
    pb = _add_sibling(root, tmp_path, "sib-b", "feat/b-unmerged")
    (tmp_path / "sib-b" / "b.txt").write_text("b\n")
    _git(pb, "add", "b.txt")
    _git(pb, "commit", "-m", "b unique")

    # (c) dirty + merged.
    pc = _add_sibling(root, tmp_path, "sib-c", "feat/c-dirty")
    (tmp_path / "sib-c" / "seed.txt").write_text("seed\ndirty\n")

    # (d) untracked-only + merged.
    pd = _add_sibling(root, tmp_path, "sib-d", "feat/d-untracked")
    (tmp_path / "sib-d" / "new.tmp").write_text("x\n")

    line = rw.gauge(root, "/.claude/worktrees/", (), budget_s=15.0)
    assert "stale-siblings=1" in line, f"only (a) is reclaimable, got: {line}"
    # Non-degraded run, and the stale sibling trips the action flag.
    assert "[partial:" not in line
    assert "⚠" in line, f"a stale sibling must flag for action, got: {line}"

    # Cross-check the plan: exactly the (a) path is a stale sibling; b/c/d kept.
    plan = _build(root)
    stale_paths = {w.path for w, _ in plan.stale_siblings}
    assert len(stale_paths) == 1
    kept_paths = {w.path for w, _, _ in plan.keep}
    assert pb in kept_paths and pc in kept_paths and pd in kept_paths


def test_issue47_detached_head_sibling_is_kept(tmp_path):
    """Scope guard: a detached-HEAD sibling (branch is None) is never reclaimed.

    We cannot prove a detached worktree's HEAD is captured by name, so it stays
    keep-by-hand even if its tree is clean and the commit is in main."""
    root = _make_repo(tmp_path)
    path = str(tmp_path / "sib-detached")
    head = rw._git_run(["rev-parse", "HEAD"], cwd=root).stdout.strip()
    _git(root, "worktree", "add", "--detach", path, head)

    wt = next(w for w in rw.parse_worktrees(
        rw._git_run(["worktree", "list", "--porcelain"], cwd=root).stdout)
        if w.path == path)
    assert wt.detached is True and wt.branch is None
    assert rw.is_stale_sibling(wt, root) is False

    plan = _build(root)
    assert path not in {w.path for w, _ in plan.stale_siblings}
    assert path in {w.path for w, _, _ in plan.keep}
