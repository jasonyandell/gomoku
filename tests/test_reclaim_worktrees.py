"""Unit tests for the worktree reclamation janitor's pure logic."""
import os
import sys

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
