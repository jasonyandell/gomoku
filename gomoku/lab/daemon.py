#!/usr/bin/env python3
"""The autolab daemon — the shared loop the trainer and arena both plug into.

One shape (epic #53): a guaranteed-singleton process that reads the whole ledger,
picks the first-priority open item for its role, runs ONE bounded chunk, appends a
result (+ follow-up rows = the flywheel), and repeats. Trainer and arena differ
only in their ``Role`` (``run_chunk`` / ``preflight``).

Two deliberate simplifications (Jason's spec, 2026-06-18):

  * **No claim/lease rows.** Mutual exclusion comes from an OS ``flock`` that the
    kernel auto-releases when the process dies — so a crash/SIGKILL needs NO
    stale-lock cleanup (contrast the PID-text lock in ``lab_train_cell.py`` which
    must probe ``_pid_alive`` and clobber). Crash recovery is just *re-pick*: a
    fresh daemon reads the ledger and picks the first-priority item again; the
    in-flight lane's progress is safe in its own ``latest.pt``. (The ``claim``
    primitive stays in ``ledger.py`` unused — the multi-machine escape hatch.)
  * **Legibility via the lockfile**, not the ledger: the flocked lockfile carries
    ``{pid, role, item, status, started_at, host}`` so ``autolab status`` can show
    "trainer on exp-7 since T" without a claim row. Liveness is the flock probe
    (``probe_alive``); the PID is the human-readable fallback.

Stdlib-only, same as ``ledger.py``.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from . import ledger

try:  # POSIX file locking (macOS/Linux). No singleton guarantee without it.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

EXIT_SINGLETON_HELD = 75  # EX_TEMPFAIL — another daemon for this role is running


# ---- home + canonical lockfile path -------------------------------------

def home() -> str:
    """The out-of-git autolab home. Everything lives here: ``ledger.jsonl``,
    ``runs/<lane>/``, ``worktrees/<row>/``, ``daemon-<role>.lock``."""
    return os.path.expanduser(os.environ.get("AUTOLAB_HOME", "~/data/autolab"))


def default_ledger_path() -> str:
    return os.path.join(home(), "ledger.jsonl")


def lockfile_path(role: str) -> str:
    """Canonical per-role lockfile. Derived from role ONLY (never a CLI arg) so
    two invocations of the same role can't pick different files and both 'win'."""
    return os.path.join(home(), f"daemon-{role}.lock")


# ---- the singleton lock -------------------------------------------------

class SingletonLock:
    """An flock-backed singleton for one role.

    ``acquire()`` returns True if we became the singleton, False if another live
    daemon holds it. The lock auto-releases when this process dies (the fd
    closes) — that is the whole point: no stale-lock cleanup, ever. The lockfile
    body carries metadata for ``autolab status``.
    """

    def __init__(self, role: str, *, path: str | None = None):
        self.role = role
        self.path = path or lockfile_path(role)
        self._fd: int | None = None

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        # Don't let run_chunk's subprocesses inherit (and thus pin) the lock fd:
        # otherwise a child outliving the daemon keeps the role looking ALIVE.
        os.set_inheritable(fd, False)
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                os.close(fd)
                if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    return False
                raise
        self._fd = fd
        self.write_meta(item=None, status="starting")
        return True

    def write_meta(self, *, item: str | None, status: str,
                   started_at: str | None = None) -> None:
        if self._fd is None:
            return
        meta = {
            "pid": os.getpid(),
            "role": self.role,
            "item": item,
            "status": status,
            "started_at": started_at or ledger.now_iso(),
            "host": socket.gethostname().split(".")[0],
        }
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, (json.dumps(meta) + "\n").encode("utf-8"))
        os.fsync(self._fd)

    def release(self) -> None:
        if self._fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "SingletonLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def read_lockfile(path: str) -> dict | None:
    """Parse the metadata a daemon wrote; None if absent/empty/garbage."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            txt = f.read().strip()
        return json.loads(txt) if txt else None
    except (OSError, ValueError):
        return None


def probe_alive(path: str) -> bool:
    """Is a live daemon holding this lockfile? A reader tries the lock: if it
    ACQUIRES, nobody holds it => DOWN (release at once); EWOULDBLOCK => ALIVE."""
    if fcntl is None or not os.path.exists(path):
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)  # we got it => no one holds it => down
        return False
    except OSError as e:
        return e.errno in (errno.EWOULDBLOCK, errno.EAGAIN)
    finally:
        os.close(fd)


# ---- the role contract --------------------------------------------------

class PreflightDeferred(Exception):
    """Raised by ``Role.preflight`` when the box isn't the lab's right now (a
    foreign MPS tenant). Transient: the loop logs an ``event`` and re-polls — it
    does NOT mark the lane FAILED (a momentary IDE tenant must not poison a lane).
    """


@dataclass
class ChunkResult:
    """What a role's ``run_chunk`` returns; the loop turns it into a result row."""

    artifact_ref: str | None = None        # e.g. "hf://repo@rev" or "local://…/latest.pt"
    metrics: dict = field(default_factory=dict)
    followups: list[dict] = field(default_factory=list)  # ledger rows to append (flywheel)
    buffer: str | None = None
    wall_s: float | None = None
    status: str = ledger.DONE              # DONE | FAILED


class Role(Protocol):
    role_name: str
    poll_interval: float

    def preflight(self) -> None: ...
    def run_chunk(self, item: dict) -> ChunkResult: ...


# ---- the loop -----------------------------------------------------------

def _interruptible_sleep(seconds: float, stop: dict) -> None:
    """Sleep in small slices so SIGTERM during an idle wait returns promptly."""
    end = time.monotonic() + seconds
    while not stop["flag"] and time.monotonic() < end:
        time.sleep(min(0.25, max(0.0, end - time.monotonic())))


def run_daemon(role: Role, ledger_path: str, *, once: bool = False,
               stop_file: str | None = None,
               clock: Callable[[], object] = ledger.utcnow) -> int:
    """Run the role's loop as the singleton for its role.

    Returns 0 on a clean stop, EXIT_SINGLETON_HELD if another daemon holds the
    role's lock. No claim rows: the flock is the guarantee, re-pick the recovery.
    """
    lock = SingletonLock(role.role_name)
    if not lock.acquire():
        sys.stderr.write(f"[daemon] {role.role_name} singleton already held; exiting\n")
        return EXIT_SINGLETON_HELD

    stop = {"flag": False}

    def _handle(_signum, _frame):
        stop["flag"] = True

    old_int = signal.signal(signal.SIGINT, _handle)
    old_term = signal.signal(signal.SIGTERM, _handle)
    try:
        while not stop["flag"]:
            if stop_file and os.path.exists(stop_file):
                break
            state = ledger.fold(ledger.read_all(ledger_path))
            item = state.pick(role.role_name, clock())
            if item is None:
                lock.write_meta(item=None, status="idle")
                if once:
                    break
                _interruptible_sleep(role.poll_interval, stop)
                continue

            started = ledger.now_iso()
            lock.write_meta(item=item["id"], status="running", started_at=started)
            try:
                role.preflight()
            except PreflightDeferred as e:
                ledger.append(ledger_path, ledger.event(
                    scope=role.role_name, summary=f"preflight deferred: {e}",
                    data={"item": item["id"]}))
                if once:
                    break
                _interruptible_sleep(role.poll_interval, stop)
                continue

            t0 = time.perf_counter()
            try:
                res = role.run_chunk(item)
                wall = res.wall_s if res.wall_s is not None else time.perf_counter() - t0
                ledger.append(ledger_path, ledger.result(
                    item["id"], status=res.status, model=res.artifact_ref,
                    buffer=res.buffer, metrics=res.metrics, wall_s=wall))
                for f in res.followups:
                    ledger.append(ledger_path, f)
            except Exception as e:  # a broken slice is terminal unless reopened
                ledger.append(ledger_path, ledger.result(
                    item["id"], status=ledger.FAILED,
                    error=f"{type(e).__name__}: {e}",
                    wall_s=time.perf_counter() - t0))
            if once:
                break
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        lock.release()
    return 0


# ---- a GPU-free role for smoke tests ------------------------------------

class NoOpRole:
    """A role that does no real work — exercises the whole loop without a GPU."""

    role_name = "noop"
    poll_interval = 0.05

    def __init__(self, *, followups: list[dict] | None = None, fail: bool = False,
                 defer: bool = False):
        self._followups = followups or []
        self._fail = fail
        self._defer = defer

    def preflight(self) -> None:
        if self._defer:
            raise PreflightDeferred("noop forced defer")

    def run_chunk(self, item: dict) -> ChunkResult:
        if self._fail:
            raise RuntimeError("noop forced failure")
        return ChunkResult(artifact_ref=f"noop://{item['id']}",
                           metrics={"noop": 1}, followups=list(self._followups),
                           wall_s=0.0)


# ---- CLI ----------------------------------------------------------------

def _resolve_role(name: str) -> Role:
    if name == "noop":
        return NoOpRole()
    # train/arena roles register in P3/P4.
    raise SystemExit(f"unknown role {name!r} (P2 knows only 'noop')")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=default_ledger_path(),
                    help="path to the ledger JSONL (default: $AUTOLAB_HOME/ledger.jsonl)")
    ap.add_argument("--role", default="noop")
    ap.add_argument("--once", action="store_true", help="run a single iteration then exit")
    ap.add_argument("--stop-file", help="break the loop when this file appears")
    args = ap.parse_args(argv)
    return run_daemon(_resolve_role(args.role), args.ledger,
                      once=args.once, stop_file=args.stop_file)


if __name__ == "__main__":
    sys.exit(main())
