#!/usr/bin/env python3
"""The autolab supervisor — ``autolab up/down/status/restart`` (epic #53, P7).

A thin, idempotent CLI that turns the tested autolab *library* into a *running*
self-driving lab under launchd. It is **not** long-lived: it makes the home
subtree, clears the cooperative stop-file, seeds the ledger with exactly one
overnight train lane (idempotently), renders the four LaunchAgent plists, and
loads them with ``launchctl`` (bootout-before-bootstrap so re-``up`` is safe).

It is glue over the spine — ``daemon`` (home + lockfile paths), ``ledger`` (the
seed row), ``status`` (the board). Stdlib only: ``argparse``, ``os``,
``plistlib``, ``subprocess``, ``json``.

Everything that touches the real machine is injectable so the tests run without
a launchctl/git/LaunchAgents dir:

  * ``--dry-run`` (or ``runner=...``) — print the launchctl argv instead of
    executing it; tests pass a recording runner.
  * ``--launchd-dir`` / ``$AUTOLAB_LAUNCHD_DIR`` — where plists are written
    (default ``~/Library/LaunchAgents``).
  * ``--commit`` / ``git_head=...`` — pin the seed SHA (default: resolve
    ``git -C <MAIN_REPO> rev-parse HEAD`` at seed time).

Canonical invocation: ``uv run python -m gomoku.lab.up up`` (the per-worktree
uv env resolves the module with no reinstall). ``gomoku-lab = gomoku.lab.up:main``
is the convenience console script (``uv run gomoku-lab up``).
"""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys

from . import daemon, ledger, status

# ---- the locked decisions (the §c seed + §b plist contract) -------------

MAIN_REPO = "/Users/jason/code/gomoku"
# uv is the one unified python runner (#87): launchd cannot use PATH lookup, so
# ProgramArguments[0] must be the absolute uv binary. `uv run` (cwd=WORKDIR)
# resolves to main repo's per-worktree .venv (uv.lock-pinned).
UV_BIN = "/opt/homebrew/bin/uv"
UV_RUN_PYTHON = (UV_BIN, "run", "python")
WORKDIR = "/Users/jason/code/gomoku"
MONITOR_SCRIPT = "/Users/jason/code/gomoku/scripts/autolab_monitor.py"

# Created by ``up`` (os.makedirs exist_ok=True) before the daemons load.
HOME_SUBDIRS = ("logs", "runs", "worktrees", "arena", "monitor", "research")

# The overnight seed lane (§c). ``id`` = ``<lane>@<seq_n>``.
SEED_LANE = "9x9-champ-recipe"
SEED_CELL = "derby-v9-small"
SEED_MAX_WALL_SECS = 3600
SEED_SEQ_N = 0
SEED_PRIORITY = 10
SEED_BASE = "scratch"
SEED_ID = f"{SEED_LANE}@{SEED_SEQ_N}"
SEED_NOTE = ("overnight seed: fresh 9x9 v8-champion recipe (derby-v9-small), "
             "scratch, 1h slices; flywheel chains continuations")

# The four LaunchAgents. Order is load order (daemons first, periodics last).
LABELS = (
    "com.gomoku.autolab.train",
    "com.gomoku.autolab.arena",
    "com.gomoku.autolab.monitor",
    "com.gomoku.autolab.research",
)


# ---- path helpers -------------------------------------------------------

def stop_file_path() -> str:
    return os.path.join(daemon.home(), "stop")


def launchd_dir(explicit: str | None = None) -> str:
    """Where the plists live. Arg > $AUTOLAB_LAUNCHD_DIR > ~/Library/LaunchAgents."""
    if explicit:
        return os.path.expanduser(explicit)
    return os.path.expanduser(
        os.environ.get("AUTOLAB_LAUNCHD_DIR", "~/Library/LaunchAgents"))


def plist_path(label: str, *, launchd_dir_path: str | None = None) -> str:
    return os.path.join(launchd_dir(launchd_dir_path), f"{label}.plist")


def _logs(name: str, ext: str) -> str:
    """A path under ~/data/autolab/logs/<name>.<ext>."""
    return os.path.join(daemon.home(), "logs", f"{name}.{ext}")


def _sub(name: str, fname: str) -> str:
    """A path under ~/data/autolab/<name>/<fname>."""
    return os.path.join(daemon.home(), name, fname)


# ---- plist rendering ----------------------------------------------------

# The single source of env for the whole launchd-spawned process tree (§b). The
# plist env is inherited all the way down to gomoku.train.
_DAEMON_ENV = {
    "AUTOLAB_HOME": daemon.home,           # resolved at render time (honors AUTOLAB_HOME)
    "HOME": "/Users/jason",
    # PATH includes /opt/homebrew/bin so child processes can find uv too.
    "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "WANDB_MODE": "offline",
}
# The periodics (monitor/research) touch no GPU — minimal env (no MPS/HF/W&B).
_PERIODIC_ENV = {
    "AUTOLAB_HOME": daemon.home,
    "HOME": "/Users/jason",
    "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
}


def _env(template: dict) -> dict:
    """Resolve any callables (AUTOLAB_HOME) in an env template."""
    return {k: (v() if callable(v) else v) for k, v in template.items()}


def _daemon_plist(label: str, program_args: list[str], log_stem: str,
                  board_size: int | None = None) -> dict:
    """A KeepAlive(SuccessfulExit=false) long-running daemon (train/arena)."""
    env = _env(_DAEMON_ENV)
    if board_size is not None:
        env["GOMOKU_BOARD_SIZE"] = str(board_size)
    return {
        "Label": label,
        "ProgramArguments": list(program_args),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "WorkingDirectory": WORKDIR,
        "EnvironmentVariables": env,
        "StandardOutPath": _logs(log_stem, "out.log"),
        "StandardErrorPath": _logs(log_stem, "err.log"),
        "ProcessType": "Standard",
        "Nice": 5,
    }


def _periodic_plist(label: str, program_args: list[str], interval: int,
                    out_path: str, err_path: str) -> dict:
    """A StartInterval one-shot (monitor/research) — no KeepAlive."""
    return {
        "Label": label,
        "ProgramArguments": list(program_args),
        "RunAtLoad": True,
        "StartInterval": interval,
        "WorkingDirectory": WORKDIR,
        "EnvironmentVariables": _env(_PERIODIC_ENV),
        "StandardOutPath": out_path,
        "StandardErrorPath": err_path,
    }


def render_plists(*, board_size: int | None = None) -> dict[str, dict]:
    """The four plist dicts keyed by label (§b literal config). Pure — no I/O.

    ``board_size`` (when set) bakes ``GOMOKU_BOARD_SIZE`` into the train + arena
    daemon env so the long-running daemons gate at the requested board size; None
    means native 9x9 (no key)."""
    stop = stop_file_path()
    return {
        "com.gomoku.autolab.train": _daemon_plist(
            "com.gomoku.autolab.train",
            [*UV_RUN_PYTHON, "-m", "gomoku.lab.trainer", "--prod", "--stop-file", stop],
            "train", board_size=board_size),
        "com.gomoku.autolab.arena": _daemon_plist(
            "com.gomoku.autolab.arena",
            [*UV_RUN_PYTHON, "-m", "gomoku.lab.arena", "--stop-file", stop],
            "arena", board_size=board_size),
        "com.gomoku.autolab.monitor": _periodic_plist(
            "com.gomoku.autolab.monitor",
            [*UV_RUN_PYTHON, MONITOR_SCRIPT],
            600, _sub("monitor", "launchd.out.log"), _sub("monitor", "launchd.err.log")),
        "com.gomoku.autolab.research": _periodic_plist(
            "com.gomoku.autolab.research",
            [*UV_RUN_PYTHON, "-m", "gomoku.lab.research", "--once"],
            1800, _sub("research", "launchd.out.log"), _sub("research", "launchd.err.log")),
    }


def write_plists(*, launchd_dir_path: str | None = None,
                 board_size: int | None = None) -> dict[str, str]:
    """Render + write all four plists; returns label → written path."""
    d = launchd_dir(launchd_dir_path)
    os.makedirs(d, exist_ok=True)
    written = {}
    for label, doc in render_plists(board_size=board_size).items():
        p = os.path.join(d, f"{label}.plist")
        with open(p, "wb") as f:
            plistlib.dump(doc, f)
        written[label] = p
    return written


# ---- launchctl / git seams ----------------------------------------------

def _gui_target() -> str:
    return f"gui/{os.getuid()}"


def _real_runner(argv: list[str]) -> subprocess.CompletedProcess:
    """Execute a command, capturing output (the default for up/down)."""
    return subprocess.run(argv, capture_output=True, text=True)


def _dry_runner(argv: list[str]) -> subprocess.CompletedProcess:
    """Print the argv instead of executing it (``--dry-run``)."""
    print("DRY-RUN:", " ".join(argv))
    return subprocess.CompletedProcess(argv, 0, "", "")


def resolve_commit(git_head=None) -> str | None:
    """The seed lane's pinned SHA = MAIN repo HEAD. ``git_head`` overrides:
    a bare string is used verbatim; a callable is invoked. Default: shell out to
    ``git -C <MAIN_REPO> rev-parse HEAD``; None on failure (caller may keep it)."""
    if git_head is not None:
        return git_head() if callable(git_head) else git_head
    try:
        r = subprocess.run(["git", "-C", MAIN_REPO, "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except OSError:
        pass
    return None


def load_agent(label: str, *, runner=_real_runner,
               launchd_dir_path: str | None = None) -> list[list[str]]:
    """bootout (ignore error) then bootstrap — single-owner, re-``up``-safe.
    Returns the argv list it ran (for inspection/tests)."""
    target = _gui_target()
    plist = plist_path(label, launchd_dir_path=launchd_dir_path)
    bootout = ["launchctl", "bootout", f"{target}/{label}"]
    bootstrap = ["launchctl", "bootstrap", target, plist]
    runner(bootout)          # ignore failure: not-loaded is fine
    runner(bootstrap)
    return [bootout, bootstrap]


def unload_agent(label: str, *, runner=_real_runner) -> list[str]:
    """bootout one agent (de-register so KeepAlive can't respawn)."""
    argv = ["launchctl", "bootout", f"{_gui_target()}/{label}"]
    runner(argv)
    return argv


# ---- seed ---------------------------------------------------------------

def seed_row(commit: str | None) -> dict:
    """The exact §c overnight seed experiment row."""
    return ledger.experiment(
        id=SEED_ID, role="train", commit=commit, base=SEED_BASE,
        config={"lane": SEED_LANE, "cell": SEED_CELL,
                "max_wall_secs": SEED_MAX_WALL_SECS, "seq_n": SEED_SEQ_N},
        priority=SEED_PRIORITY, note=SEED_NOTE)


def _has_open_train(state: ledger.LedgerState) -> bool:
    """Is there any open or claimed train experiment already? (idempotence)"""
    for e in state.experiments.values():
        if e.get("role") == "train" and e.get("status") in (ledger.OPEN, ledger.CLAIMED):
            return True
    return False


def seed_ledger(ledger_path: str | None = None, *, commit=None,
                git_head=None) -> dict | None:
    """Append the seed row iff no open/claimed train experiment exists. Returns
    the stored row, or None if a seed lane is already live (never double-seed).

    ``commit`` pins the SHA verbatim; else ``git_head`` (str|callable) else
    ``git rev-parse HEAD`` of the main repo at seed time."""
    path = ledger_path or daemon.default_ledger_path()
    state = ledger.fold(ledger.read_all(path))
    if _has_open_train(state):
        return None
    sha = commit if commit is not None else resolve_commit(git_head)
    return ledger.append(path, seed_row(sha))


# ---- stop-file lifecycle ------------------------------------------------

def clear_stop_file() -> None:
    """``up``: a stale stop-file would make both daemons exit on first poll."""
    try:
        os.remove(stop_file_path())
    except FileNotFoundError:
        pass


def write_stop_file() -> None:
    """``down``: cooperative graceful stop (daemons honor --stop-file)."""
    p = stop_file_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(ledger.now_iso() + "\n")


# ---- subcommands --------------------------------------------------------

def cmd_up(args) -> int:
    runner = _dry_runner if args.dry_run else _real_runner
    # (1) home subtree
    for sub in HOME_SUBDIRS:
        os.makedirs(os.path.join(daemon.home(), sub), exist_ok=True)
    # (2) clear the stop-file so daemons don't instantly exit
    clear_stop_file()
    # (3) seed the ledger (idempotent — never double-seed)
    seeded = seed_ledger(args.ledger, commit=args.commit)
    if seeded is not None:
        print(f"seeded {seeded['id']} (commit {seeded.get('commit') or 'HEAD'}, "
              f"priority {seeded.get('priority')})")
    else:
        print("seed: an open train lane already exists — not double-seeding")
    # (4) render + write the four plists
    written = write_plists(launchd_dir_path=args.launchd_dir,
                           board_size=args.board_size)
    # (5) load each idempotently (bootout-before-bootstrap)
    for label in LABELS:
        load_agent(label, runner=runner, launchd_dir_path=args.launchd_dir)
        print(f"loaded {label} ({written[label]})")
    print()
    return cmd_status(args)


def cmd_down(args) -> int:
    runner = _dry_runner if args.dry_run else _real_runner
    # graceful: stop-file first (daemons honor --stop-file via run_daemon)
    write_stop_file()
    print(f"wrote stop-file {stop_file_path()}")
    # hammer: bootout each (de-register so KeepAlive can't respawn; SIGTERMs the daemon)
    for label in LABELS:
        unload_agent(label, runner=runner)
        print(f"booted out {label}")
    if args.force:
        # best-effort hard stop of any run_sweep scoped to OUR worktrees only
        worktrees = os.path.join(daemon.home(), "worktrees")
        runner(["pkill", "-TERM", "-f", f"scripts/run_sweep.py.*{worktrees}"])
        print(f"pkill -TERM run_sweep scoped to {worktrees}")
    return 0


def cmd_status(args) -> int:
    path = args.ledger or daemon.default_ledger_path()
    print(status.format_board(status.lane_board(path)))
    print("\nlaunchd agents:")
    runner = _dry_runner if args.dry_run else _real_runner
    for label in LABELS:
        r = runner(["launchctl", "print", f"{_gui_target()}/{label}"])
        loaded = getattr(r, "returncode", 1) == 0
        print(f"  {label:<32} {'LOADED' if loaded else 'not loaded'}")
    sp = stop_file_path()
    print(f"\nstop-file: {'present' if os.path.exists(sp) else 'absent'} ({sp})")
    return 0


def cmd_restart(args) -> int:
    cmd_down(args)
    return cmd_up(args)


# ---- CLI ----------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=None,
                    help="ledger JSONL path (default: $AUTOLAB_HOME/ledger.jsonl)")
    ap.add_argument("--launchd-dir", default=None,
                    help="LaunchAgents dir (default: $AUTOLAB_LAUNCHD_DIR or "
                         "~/Library/LaunchAgents)")
    ap.add_argument("--commit", default=None,
                    help="pin the seed SHA (default: git rev-parse HEAD of main)")
    ap.add_argument("--board-size", type=int, default=None,
                    help="bake GOMOKU_BOARD_SIZE into the train + arena daemon "
                         "plists (default: None = native 9x9). Must precede the "
                         "subcommand, e.g. `--board-size 15 up`")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the launchctl argv instead of executing it")
    ap.add_argument("--force", action="store_true",
                    help="down: also pkill run_sweep scoped to the autolab worktrees")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up", help="makedirs + seed + render plists + load agents").set_defaults(func=cmd_up)
    sub.add_parser("down", help="stop-file + bootout agents").set_defaults(func=cmd_down)
    sub.add_parser("status", help="print the board + which plists are loaded").set_defaults(func=cmd_status)
    sub.add_parser("restart", help="down then up").set_defaults(func=cmd_restart)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
