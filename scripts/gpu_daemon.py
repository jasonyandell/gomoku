"""gpu_daemon — the lab's single serial GPU-lane scheduler.

The M5 Max runs **one** MPS tenant at a time (train / gen / eval / perf). Until
now *Claude-the-orchestrator* was that scheduler, dispatching cells by hand off
`wiki/ops/gpu-queue.md`. This daemon formalises that role: jobs are **submitted**
as config files, **queued** on disk, and **run serially** on the GPU. `delo_derby`
and manual dispatch become clients that submit; nobody else launches an MPS tenant.

Design (deliberately boring + auditable):

* **A job is a config file.** One JSON file, fully declarative. Its *directory*
  encodes its state — moving the file IS the state transition, so the whole queue
  is greppable, diff-able, and survives a daemon restart:

      lab_queue/
        pending/   <id>.json     # waiting to run, sorted by (tier, -priority, submitted_at)
        running/   <id>.json      # the one tenant on the box right now
        done/      <id>.json
        failed/    <id>.json
        cancelled/ <id>.json
        logs/      <id>.log       # the job's subprocess output
        events.jsonl              # append-only audit log of every transition
        daemon.pid                # {pid, started, queue_dir}; flock'd so only one daemon runs

* **Two job kinds.** `train` builds the established clean-stop slice command
  `run_sweep.py --cell C [--resume latest.pt] --max-wall-secs N --final-eval`
  (the daemon self-caps, clean-saves a resumable `latest.pt`, and reads the final
  `eval/model_elo` from `eval_results.jsonl`). `raw` runs an arbitrary argv on the
  serial lane — the escape hatch for gen / eval / perf cells (canonical_sweep,
  eval_worker, …). Everything else is just fields in the config file.

* **Serial + polite.** Exactly one job runs at a time. When idle the daemon
  preflights for *foreign* MPS tenants (another non-daemon process on the box, e.g.
  a hand-launched derby) and waits rather than barging in.

* **Clean stop + crash resume.** SIGTERM to the daemon forwards SIGTERM to the
  running job's process group (run_sweep tears down its trainer+workers and saves
  `latest.pt`), then the daemon exits. An interrupted train job is re-queued with
  `resume_from=auto` so the next start continues it with no cold buffer refill. A
  hard crash leaves the job in `running/`; the next daemon start reconciles it back
  to `pending` the same way.

CLI::

    # run the scheduler (background it yourself: `nohup ... &` or a launchd plist)
    python scripts/gpu_daemon.py daemon [--queue-dir DIR] [--poll-secs 5] [--once]

    # submit a training slice (writes an auditable config file into pending/)
    python scripts/gpu_daemon.py submit --kind train --cell WL5 \
        --max-wall-secs 600 [--final-eval] [--resume auto|PATH] \
        [--tier 1] [--priority 5] [--note "..."]

    # submit any GPU work via a raw command
    python scripts/gpu_daemon.py submit --kind raw --note "S400 gen sweep" \
        --cmd "python -u scripts/canonical_sweep.py --out-dir ... --cells-from ..."

    # submit a pre-written config file
    python scripts/gpu_daemon.py submit path/to/job.json

    python scripts/gpu_daemon.py status [--json]      # daemon + queue snapshot
    python scripts/gpu_daemon.py show <id>            # one job's config + log tail
    python scripts/gpu_daemon.py cancel <id>          # drop pending / clean-stop running
    python scripts/gpu_daemon.py stop                 # ask the daemon to drain + exit
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
RUN_SWEEP_PY = SCRIPTS_DIR / "run_sweep.py"
PYTHON = sys.executable

DEFAULT_QUEUE_DIR = Path(
    os.environ.get("GOMOKU_LAB_QUEUE", str(REPO_ROOT / "lab_queue"))
)

STATES = ("pending", "running", "done", "failed", "cancelled")
ACTIVE_STATES = ("pending", "running")

# pgrep pattern for "an MPS tenant is on the box". Mirrors lab_train_cell.preflight_idle
# and the session-runbook idle check. NOTE: keep these strings OUT of the daemon's own
# command line (it's `gpu_daemon.py daemon`, which matches none of them) so preflight
# never self-collides — see the lab skill's friction log.
TENANT_RE = r"selfplay_worker|gomoku\.train|run_sweep|eval_worker"

# Teardown budget mirrors run_sweep's own constants so the daemon's safety timeout
# never fires before run_sweep has had its chance to clean-save + final-eval.
TRAINER_CAP_GRACE_SEC = 180.0
TEARDOWN_GRACE_SEC = 90.0
FINAL_EVAL_TIMEOUT_SEC = 360.0
SAFETY_PAD_SEC = 180.0
DRAIN_GRACE_SEC = TEARDOWN_GRACE_SEC + FINAL_EVAL_TIMEOUT_SEC + 60.0

ELO_KEY = "eval/model_elo"
EPOCH_KEY = "eval_worker/epoch_evaluated"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, obj: Any) -> None:
    """tmp + fsync + rename so a crash never leaves a partial config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# queue layout
# ---------------------------------------------------------------------------
class Queue:
    """Thin wrapper over the maildir-style queue directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        for st in STATES:
            (self.root / st).mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.pid_path = self.root / "daemon.pid"

    # -- paths ----------------------------------------------------------------
    def state_dir(self, state: str) -> Path:
        return self.root / state

    def job_path(self, state: str, job_id: str) -> Path:
        return self.state_dir(state) / f"{job_id}.json"

    def log_path(self, job_id: str) -> Path:
        return self.root / "logs" / f"{job_id}.log"

    def cancel_marker(self, job_id: str) -> Path:
        return self.root / "logs" / f"{job_id}.cancel"

    # -- io -------------------------------------------------------------------
    def list_jobs(self, state: str) -> list[dict]:
        out = []
        for p in sorted(self.state_dir(state).glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def find(self, job_id: str) -> Optional[tuple[str, dict]]:
        for st in STATES:
            p = self.job_path(st, job_id)
            if p.exists():
                try:
                    return st, json.loads(p.read_text())
                except (OSError, json.JSONDecodeError):
                    return st, {}
        return None

    def write(self, state: str, job: dict) -> Path:
        job["status"] = state
        path = self.job_path(state, job["id"])
        atomic_write_json(path, job)
        return path

    def move(self, from_state: str, to_state: str, job: dict) -> Path:
        """Transition a job between states. The new file is the source of truth;
        the old one is removed only after the new is durably written."""
        src = self.job_path(from_state, job["id"])
        dst = self.write(to_state, job)
        if src != dst and src.exists():
            try:
                src.unlink()
            except OSError:
                pass
        self.event(job["id"], "move", {"from": from_state, "to": to_state})
        return dst

    def event(self, job_id: str, kind: str, extra: Optional[dict] = None) -> None:
        rec = {"ts": now_iso(), "job": job_id, "event": kind}
        if extra:
            rec.update(extra)
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def daemon_pid(self) -> Optional[int]:
        if not self.pid_path.exists():
            return None
        try:
            rec = json.loads(self.pid_path.read_text())
            return int(rec["pid"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None

    def daemon_alive(self) -> bool:
        pid = self.daemon_pid()
        return pid is not None and pid_alive(pid)


def sort_pending(jobs: list[dict]) -> list[dict]:
    """tier ascending (1 architectural first), then priority descending, then FIFO."""
    return sorted(
        jobs,
        key=lambda j: (
            int(j.get("tier", 2)),
            -float(j.get("priority", 0.0)),
            str(j.get("submitted_at", "")),
            str(j.get("id", "")),
        ),
    )


# ---------------------------------------------------------------------------
# job construction + command building
# ---------------------------------------------------------------------------
def new_job_id(kind: str, label: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(c if c.isalnum() else "-" for c in (label or kind)).strip("-")[:32]
    return f"{ts}-{kind}-{slug}-{uuid.uuid4().hex[:4]}"


def normalize_job(job: dict) -> dict:
    """Fill defaults + assign an id. Raises ValueError on an invalid spec."""
    kind = job.get("kind")
    if kind not in ("train", "raw"):
        raise ValueError(f"job 'kind' must be 'train' or 'raw', got {kind!r}")

    job.setdefault("tier", 2)
    job.setdefault("priority", 0.0)
    job.setdefault("note", "")
    job.setdefault("submitted_at", now_iso())

    if kind == "train":
        if not job.get("cell"):
            raise ValueError("train job requires 'cell'")
        job.setdefault("max_wall_secs", 600.0)
        job.setdefault("final_eval", True)
        job.setdefault("resume_from", None)
        job.setdefault("overrides", {})
        _validate_cell(job["cell"])
        label = job["cell"]
    else:  # raw
        cmd = job.get("cmd")
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
            job["cmd"] = cmd
        if not isinstance(cmd, list) or not cmd:
            raise ValueError("raw job requires a non-empty 'cmd' (list or string)")
        label = job["note"] or "raw"

    if not job.get("id"):
        job["id"] = new_job_id(kind, label)
    return job


def _validate_cell(cell: str) -> None:
    try:
        run_sweep = _import_run_sweep()
    except Exception:
        return  # can't import here; the daemon validates again at dispatch
    if cell not in run_sweep.CELLS:
        raise ValueError(
            f"unknown cell {cell!r}; one of {sorted(run_sweep.CELLS)}"
        )


def _import_run_sweep():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import run_sweep  # stdlib-only at module load; safe to import
    return run_sweep


def resolve_resume(job: dict) -> Optional[Path]:
    rf = job.get("resume_from")
    if rf in (None, "", "none", "null", "None"):
        return None
    if rf == "auto":
        try:
            run_sweep = _import_run_sweep()
            cell = run_sweep.CELLS[job["cell"]]
            latest = run_sweep.cell_dirs(cell)["checkpoint_dir"] / "latest.pt"
        except Exception:
            latest = REPO_ROOT / "sweep_runs" / job["cell"] / "checkpoints" / "latest.pt"
        return latest if latest.exists() else None
    p = Path(rf)
    return p if p.is_absolute() else (REPO_ROOT / p)


def build_cmd(job: dict) -> list[str]:
    if job["kind"] == "raw":
        return [str(x) for x in job["cmd"]]

    # train -> the established clean-stop slice via run_sweep
    cell = job["cell"]
    _validate_cell(cell)
    cmd = [PYTHON, "-u", str(RUN_SWEEP_PY), "--cell", cell]
    resume = resolve_resume(job)
    if resume is not None:
        cmd += ["--resume", str(resume)]
    mws = float(job.get("max_wall_secs") or 0.0)
    if mws > 0:
        cmd += ["--max-wall-secs", str(mws)]
    if job.get("final_eval", True) and mws > 0:
        cmd += ["--final-eval"]
    ov = job.get("overrides") or {}
    if "epochs" in ov:
        cmd += ["--epochs", str(int(ov["epochs"]))]
    return cmd


def read_last_elo(cell: str) -> Optional[tuple[float, Optional[int]]]:
    """Final eval/model_elo for a train cell, scanning eval_results.jsonl backwards."""
    try:
        run_sweep = _import_run_sweep()
        ckpt = run_sweep.cell_dirs(run_sweep.CELLS[cell])["checkpoint_dir"]
    except Exception:
        ckpt = REPO_ROOT / "sweep_runs" / cell / "checkpoints"
    jsonl = ckpt / "eval_results.jsonl"
    if not jsonl.exists():
        return None
    try:
        lines = jsonl.read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ELO_KEY in rec:
            ep = rec.get(EPOCH_KEY)
            return float(rec[ELO_KEY]), (int(ep) if ep is not None else None)
    return None


def safety_timeout(job: dict) -> float:
    if job["kind"] == "train":
        mws = float(job.get("max_wall_secs") or 0.0)
        return mws + TRAINER_CAP_GRACE_SEC + TEARDOWN_GRACE_SEC + FINAL_EVAL_TIMEOUT_SEC + SAFETY_PAD_SEC
    return float(job.get("timeout_secs") or 24 * 3600.0)


# ---------------------------------------------------------------------------
# foreign-tenant preflight (politeness when idle)
# ---------------------------------------------------------------------------
def foreign_tenants(exclude_pids: set[int]) -> list[str]:
    """Lines for MPS tenants NOT spawned by us. Empty => box is ours to use."""
    try:
        out = subprocess.check_output(["pgrep", "-fl", TENANT_RE], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    foreign = []
    for line in out.splitlines():
        if not line.strip():
            continue
        try:
            pid = int(line.split()[0])
        except (ValueError, IndexError):
            continue
        if pid in exclude_pids:
            continue
        foreign.append(line)
    return foreign


# ---------------------------------------------------------------------------
# the daemon
# ---------------------------------------------------------------------------
class Daemon:
    def __init__(self, queue: Queue, poll_secs: float, preflight: bool, once: bool):
        self.q = queue
        self.poll_secs = poll_secs
        self.preflight = preflight
        self.once = once
        self.stop = False
        self.proc: Optional[subprocess.Popen] = None
        self._termed = False
        self._lock_fd: Optional[int] = None

    # -- logging --------------------------------------------------------------
    def log(self, msg: str) -> None:
        print(f"[{now_iso()}] {msg}", flush=True)

    # -- single-instance lock -------------------------------------------------
    def acquire(self) -> None:
        lock_path = self.q.root / "daemon.lock"
        self._lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pid = self.q.daemon_pid()
            raise SystemExit(
                f"another gpu_daemon already holds {lock_path} (pid {pid}); refusing to start."
            )
        atomic_write_json(
            self.q.pid_path,
            {"pid": os.getpid(), "started": now_iso(), "queue_dir": str(self.q.root)},
        )

    def release(self) -> None:
        try:
            if self.q.pid_path.exists():
                self.q.pid_path.unlink()
        except OSError:
            pass
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except OSError:
                pass

    # -- signals --------------------------------------------------------------
    def install_signals(self) -> None:
        def handler(signum, _frame):
            self.stop = True
            self.log(f"signal {signum} received; draining current job then exiting.")
            self._terminate_child()
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _terminate_child(self) -> None:
        if self.proc and self.proc.poll() is None and not self._termed:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)  # child is a session leader
                self._termed = True
            except (ProcessLookupError, PermissionError):
                pass

    # -- startup reconcile ----------------------------------------------------
    def reconcile(self) -> None:
        """A hard crash leaves a job in running/. Re-queue it (train jobs resume)."""
        for job in self.q.list_jobs("running"):
            self.log(f"reconcile: re-queueing interrupted job {job['id']}")
            if job.get("kind") == "train":
                job["resume_from"] = "auto"
            job["requeued"] = job.get("requeued", 0) + 1
            job.pop("started_at", None)
            self.q.move("running", "pending", job)

    # -- main loop ------------------------------------------------------------
    def run(self) -> None:
        self.acquire()
        self.install_signals()
        self.log(f"gpu_daemon up (pid {os.getpid()}) queue={self.q.root}")
        try:
            self.reconcile()
            while not self.stop:
                pending = sort_pending(self.q.list_jobs("pending"))
                if not pending:
                    if self.once:
                        self.log("queue drained (--once); exiting.")
                        break
                    time.sleep(self.poll_secs)
                    continue
                if self.preflight:
                    foreign = foreign_tenants(exclude_pids={os.getpid()})
                    if foreign:
                        self.log("box busy with a foreign MPS tenant; waiting:")
                        for ln in foreign:
                            self.log(f"    {ln}")
                        time.sleep(max(self.poll_secs, 10.0))
                        continue
                self.run_job(pending[0])
            self.log("daemon stopped.")
        finally:
            self.release()

    def run_job(self, job: dict) -> None:
        job["started_at"] = now_iso()
        job["host_pid"] = os.getpid()
        try:
            cmd = build_cmd(job)
        except Exception as e:  # bad spec discovered at dispatch
            job["error"] = f"build_cmd failed: {e}"
            job["ended_at"] = now_iso()
            self.q.move("pending", "failed", job)
            self.log(f"job {job['id']} FAILED to build: {e}")
            return

        job["cmd_run"] = cmd
        self.q.move("pending", "running", job)
        log_path = self.q.log_path(job["id"])
        self.log(f"START {job['id']} :: {' '.join(shlex.quote(c) for c in cmd)}")
        self.q.event(job["id"], "start", {"cmd": cmd})

        self.proc = None
        self._termed = False
        t0 = time.monotonic()
        with open(log_path, "a") as logf:
            logf.write(f"\n===== {now_iso()} :: {' '.join(cmd)} =====\n")
            logf.flush()
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,  # own process group -> we can signal the whole bundle
            )
            rc = self._supervise(job, t0)

        wall = time.monotonic() - t0
        self._record_result(job, rc, wall)

    def _supervise(self, job: dict, t0: float) -> int:
        """Wait for the child; escalate to SIGKILL on drain-deadline or safety-timeout."""
        deadline = t0 + safety_timeout(job)
        drain_deadline: Optional[float] = None
        while True:
            rc = self.proc.poll()
            if rc is not None:
                return rc
            now = time.monotonic()
            if self.stop and drain_deadline is None:
                self._terminate_child()  # idempotent; handler likely already did it
                drain_deadline = now + DRAIN_GRACE_SEC
            if drain_deadline is not None and now > drain_deadline:
                self.log(f"job {job['id']} did not drain in time; SIGKILL.")
                self._killpg(signal.SIGKILL)
                return self._final_rc()
            if now > deadline:
                self.log(f"job {job['id']} exceeded safety timeout; SIGKILL.")
                job["error"] = "safety_timeout"
                self._killpg(signal.SIGKILL)
                return self._final_rc()
            time.sleep(1.0)

    def _killpg(self, sig: int) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _final_rc(self) -> int:
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        return self.proc.returncode if self.proc.returncode is not None else -9

    def _record_result(self, job: dict, rc: int, wall: float) -> None:
        job["ended_at"] = now_iso()
        job["returncode"] = rc
        job["wall_secs"] = round(wall, 1)
        result: dict[str, Any] = {"log": str(self.q.log_path(job["id"]))}
        if job["kind"] == "train":
            elo = read_last_elo(job["cell"])
            if elo is not None:
                result["model_elo"], result["epoch"] = elo[0], elo[1]
        job["result"] = result

        cancel_requested = self.q.cancel_marker(job["id"]).exists()
        stop_term = self.stop  # daemon is shutting down

        if cancel_requested:
            self.q.cancel_marker(job["id"]).unlink(missing_ok=True)
            to_state = "cancelled"
        elif stop_term and rc != 0:
            # interrupted by daemon shutdown before a clean finish -> resume next start
            if job["kind"] == "train":
                job["resume_from"] = "auto"
            job["requeued"] = job.get("requeued", 0) + 1
            self.q.move("running", "pending", job)
            self.log(f"REQUEUE {job['id']} (interrupted by daemon stop; will resume)")
            self.proc = None
            return
        else:
            to_state = "done" if rc == 0 else "failed"

        self.q.move("running", to_state, job)
        elo_s = ""
        if job["result"].get("model_elo") is not None:
            elo_s = f" elo={job['result']['model_elo']:.1f}"
        self.log(f"{to_state.upper()} {job['id']} rc={rc} wall={wall:.0f}s{elo_s}")
        self.proc = None


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------
def cmd_daemon(args) -> int:
    q = Queue(args.queue_dir)
    Daemon(q, poll_secs=args.poll_secs, preflight=not args.no_preflight, once=args.once).run()
    return 0


def cmd_submit(args) -> int:
    q = Queue(args.queue_dir)
    if args.spec_file:
        job = json.loads(Path(args.spec_file).read_text())
    else:
        job = _job_from_flags(args)
    try:
        job = normalize_job(job)
    except ValueError as e:
        print(f"submit: {e}", file=sys.stderr)
        return 2
    q.write("pending", job)
    q.event(job["id"], "submit", {"kind": job["kind"], "tier": job["tier"]})
    print(f"queued {job['id']}  (tier {job['tier']}, priority {job['priority']})")
    print(f"  config: {q.job_path('pending', job['id'])}")
    if not q.daemon_alive():
        print("  note: no daemon is running — start one with "
              "`python scripts/gpu_daemon.py daemon`")
    return 0


def _job_from_flags(args) -> dict:
    job: dict[str, Any] = {
        "kind": args.kind,
        "tier": args.tier,
        "priority": args.priority,
        "note": args.note or "",
    }
    if args.kind == "train":
        job["cell"] = args.cell
        job["max_wall_secs"] = args.max_wall_secs
        job["final_eval"] = args.final_eval
        job["resume_from"] = args.resume
        if args.epochs is not None:
            job["overrides"] = {"epochs": args.epochs}
    else:
        if not args.cmd:
            raise SystemExit("submit --kind raw requires --cmd \"...\"")
        job["cmd"] = args.cmd
        if args.timeout_secs:
            job["timeout_secs"] = args.timeout_secs
    return job


def cmd_status(args) -> int:
    q = Queue(args.queue_dir)
    snap = {
        "daemon": {
            "alive": q.daemon_alive(),
            "pid": q.daemon_pid(),
            "queue_dir": str(q.root),
        },
        "running": q.list_jobs("running"),
        "pending": sort_pending(q.list_jobs("pending")),
        "done": q.list_jobs("done")[-10:],
        "failed": q.list_jobs("failed")[-10:],
    }
    if args.json:
        print(json.dumps(snap, indent=2, sort_keys=True))
        return 0

    d = snap["daemon"]
    state = "RUNNING" if d["alive"] else "DOWN"
    print(f"daemon: {state}  pid={d['pid']}  queue={d['queue_dir']}")
    for job in snap["running"]:
        print(f"\nRUNNING  {job['id']}")
        print(f"    {_one_line(job)}")
        print(f"    started {job.get('started_at')}  log {q.log_path(job['id'])}")
    pend = snap["pending"]
    print(f"\nPENDING ({len(pend)}):")
    for i, job in enumerate(pend):
        print(f"  {i+1:>2}. {job['id']}  [tier {job.get('tier')} prio {job.get('priority')}]")
        print(f"        {_one_line(job)}")
    for label, key in (("recent DONE", "done"), ("recent FAILED", "failed")):
        rows = snap[key]
        if rows:
            print(f"\n{label}:")
            for job in rows:
                elo = (job.get("result") or {}).get("model_elo")
                elo_s = f" elo={elo:.1f}" if isinstance(elo, (int, float)) else ""
                print(f"  {job['id']}  rc={job.get('returncode')}{elo_s}")
    return 0


def _one_line(job: dict) -> str:
    if job["kind"] == "train":
        rf = job.get("resume_from") or "fresh"
        return (f"train cell={job['cell']} cap={job.get('max_wall_secs')}s "
                f"resume={rf} note={job.get('note','')!r}")
    return f"raw cmd={' '.join(job.get('cmd', []))!r} note={job.get('note','')!r}"


def cmd_show(args) -> int:
    q = Queue(args.queue_dir)
    found = q.find(args.id)
    if not found:
        print(f"no job {args.id!r}", file=sys.stderr)
        return 1
    state, job = found
    print(f"state: {state}")
    print(json.dumps(job, indent=2, sort_keys=True))
    log = q.log_path(args.id)
    if log.exists():
        print(f"\n--- log tail ({log}) ---")
        lines = log.read_text().splitlines()
        print("\n".join(lines[-args.tail:]))
    return 0


def cmd_cancel(args) -> int:
    q = Queue(args.queue_dir)
    found = q.find(args.id)
    if not found:
        print(f"no job {args.id!r}", file=sys.stderr)
        return 1
    state, job = found
    if state == "pending":
        q.move("pending", "cancelled", job)
        print(f"cancelled (was pending) {args.id}")
        return 0
    if state == "running":
        q.cancel_marker(args.id).touch()  # daemon files it under cancelled on exit
        pid = job.get("host_pid")
        cpid = None
        # the daemon recorded its own pid; the child is in the daemon's child group.
        # We can't address the child's pgid directly here, so ask the daemon to act:
        # touch the marker + SIGTERM the run_sweep tree via pkill on this job's cell.
        if job["kind"] == "train":
            subprocess.run(["pkill", "-TERM", "-f", f"sweep_runs/{job['cell']}/"],
                           check=False)
        print(f"cancel requested for RUNNING {args.id} (clean stop in progress)")
        return 0
    print(f"job {args.id} is already {state}; nothing to cancel")
    return 0


def cmd_stop(args) -> int:
    q = Queue(args.queue_dir)
    pid = q.daemon_pid()
    if pid is None or not pid_alive(pid):
        print("no live daemon to stop")
        return 1
    os.kill(pid, signal.SIGTERM)
    print(f"sent SIGTERM to daemon pid {pid}; it will drain the current job and exit")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR,
                   help=f"queue root (default {DEFAULT_QUEUE_DIR}; env GOMOKU_LAB_QUEUE)")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("daemon", help="run the serial GPU-lane scheduler")
    d.add_argument("--poll-secs", type=float, default=5.0)
    d.add_argument("--no-preflight", action="store_true",
                   help="skip the foreign-tenant idle check (use only if you own the box)")
    d.add_argument("--once", action="store_true", help="drain pending then exit (testing)")
    d.set_defaults(func=cmd_daemon)

    s = sub.add_parser("submit", help="queue a job (config file or flags)")
    s.add_argument("spec_file", nargs="?", help="path to a job config JSON")
    s.add_argument("--kind", choices=["train", "raw"], default="train")
    s.add_argument("--cell", help="run_sweep CELLS key (train)")
    s.add_argument("--max-wall-secs", type=float, default=600.0)
    s.add_argument("--final-eval", action="store_true", default=True)
    s.add_argument("--no-final-eval", dest="final_eval", action="store_false")
    s.add_argument("--resume", default=None,
                   help="'auto' (resume sweep_runs/<cell>/.../latest.pt if present) or a path")
    s.add_argument("--epochs", type=int, default=None, help="train: override cell.epochs")
    s.add_argument("--cmd", help="raw: the argv to run on the GPU lane")
    s.add_argument("--timeout-secs", type=float, default=None, help="raw: safety timeout")
    s.add_argument("--tier", type=int, default=2, help="1 architectural > 2 compound > 3 speculative")
    s.add_argument("--priority", type=float, default=0.0, help="higher runs first within a tier")
    s.add_argument("--note", help="human description")
    s.set_defaults(func=cmd_submit)

    st = sub.add_parser("status", help="daemon + queue snapshot")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_status)

    sh = sub.add_parser("show", help="one job's config + log tail")
    sh.add_argument("id")
    sh.add_argument("--tail", type=int, default=40)
    sh.set_defaults(func=cmd_show)

    c = sub.add_parser("cancel", help="drop a pending job / clean-stop a running one")
    c.add_argument("id")
    c.set_defaults(func=cmd_cancel)

    sp = sub.add_parser("stop", help="ask the daemon to drain the current job and exit")
    sp.set_defaults(func=cmd_stop)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
