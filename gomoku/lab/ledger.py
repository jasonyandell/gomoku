#!/usr/bin/env python3
"""The autolab ledger — the spine (epic #53, P1 #54).

ONE external, out-of-git, append-only JSONL file that every autolab loop reads to
pick first-priority work and appends results to. It is the single source of truth
for "what should run next" and "what happened".

Design (Jason's spec, 2026-06-18):

  * **Out of git.** It lives wherever ``--ledger`` points (default under
    ``~/code``), NOT inside the repo, so git can't stomp the high-frequency
    machine stream. Config + human synthesis stay in git; this stream does not.
  * **Append-only, corrected like financial transactions.** A row is never
    edited or deleted. To change something, append a ``correction`` row that
    supersedes fields of a prior entity by ``ref``. History is the audit trail.
  * **Reduced by replay.** ``fold(read_all(path))`` replays every row in order
    into the current state (``LedgerState``). 10 MB is nothing on this Mac, so we
    always read the whole file — there is no incremental cache to get stale.

Row types (one JSON object per line):

  experiment  work definition   {id, role, commit, base, config, priority, status, note}
  claim       a lease on work   {ref, by, lease_until}
  result      outcome of work   {ref, status, model, buffer, metrics, wall_s, error}
  correction  supersede fields  {ref, set:{...}, reason}      <- the financial-journal move
  eval        an arena report   {ref, model, panel, metrics}
  verdict     a gate decision   {ref, gate, win_rate, ci, n}
  event       log-only note     {scope, summary, data}

Every appended row also carries an auto-assigned monotonic ``seq`` (its ordinal
in the file, assigned under an flock) and an ISO-8601 ``ts``.

This module is stdlib-only on purpose — the spine has no project dependencies so
every loop (and a bare ``python -m gomoku.lab.ledger show``) can read it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable

try:  # POSIX inter-process append lock (macOS/Linux). No-op fallback elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

# Leading keys, ordered for human-readable rows.
_LEAD = ("seq", "ts", "type", "id", "ref", "role", "status")

EXPERIMENT = "experiment"
CLAIM = "claim"
RESULT = "result"
CORRECTION = "correction"
EVAL = "eval"
VERDICT = "verdict"
EVENT = "event"

OPEN = "open"
CLAIMED = "claimed"
DONE = "done"
FAILED = "failed"
BLOCKED = "blocked"     # created-but-not-runnable: awaiting a research decision
SUPERSEDED = "superseded"  # parked by a correction; never runnable again


# ---- time helpers -------------------------------------------------------

def utcnow() -> _dt.datetime:
    """Timezone-aware current UTC time (the canonical clock for leases)."""
    return _dt.datetime.now(_dt.timezone.utc)


def now_iso() -> str:
    return utcnow().isoformat(timespec="seconds")


def parse_ts(s: str) -> _dt.datetime:
    """Parse an ISO-8601 stamp; naive stamps are assumed UTC."""
    dt = _dt.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt


def lease_until(seconds: float, *, now: _dt.datetime | None = None) -> str:
    """ISO stamp ``seconds`` into the future — a claim's lease expiry."""
    base = now or utcnow()
    return (base + _dt.timedelta(seconds=seconds)).isoformat(timespec="seconds")


def claimant(role: str, *, host: str | None = None, pid: int | None = None) -> str:
    """A stable, legible claim owner string, e.g. ``trainer@m5max/4412``."""
    host = host or socket.gethostname().split(".")[0]
    pid = os.getpid() if pid is None else pid
    return f"{role}@{host}/{pid}"


# ---- row constructors (pure; ``append`` stamps seq + ts) ----------------

def experiment(
    id: str,
    role: str,
    *,
    commit: str | None = None,
    base: str = "scratch",
    config: dict | None = None,
    priority: int = 0,
    note: str = "",
    status: str = OPEN,
    **extra,
) -> dict:
    """A unit of work for a role to run (a commit + a start model + config)."""
    return {
        "type": EXPERIMENT,
        "id": id,
        "role": role,
        "commit": commit,
        "base": base,
        "config": config or {},
        "priority": priority,
        "note": note,
        "status": status,
        **extra,
    }


def claim(ref: str, by: str, lease: str, **extra) -> dict:
    """A lease on an experiment by a daemon (``lease`` is an ISO expiry)."""
    return {"type": CLAIM, "ref": ref, "by": by, "lease_until": lease, **extra}


def result(
    ref: str,
    *,
    status: str = DONE,
    model: str | None = None,
    buffer: str | None = None,
    metrics: dict | None = None,
    wall_s: float | None = None,
    error: str | None = None,
    **extra,
) -> dict:
    """The outcome of a claimed experiment; terminal unless reopened."""
    return {
        "type": RESULT,
        "ref": ref,
        "status": status,
        "model": model,
        "buffer": buffer,
        "metrics": metrics or {},
        "wall_s": wall_s,
        "error": error,
        **extra,
    }


def correction(ref: str, set: dict, *, reason: str = "", **extra) -> dict:
    """Supersede fields of a prior entity — the only way to 'change' history.

    ``set`` is applied last-writer-wins over the target entity. Setting
    ``{"status": "open"}`` is how a dead-lease lane is reclaimed.
    """
    return {"type": CORRECTION, "ref": ref, "set": dict(set), "reason": reason, **extra}


def eval_row(
    ref: str,
    *,
    model: str | None = None,
    panel: list | None = None,
    metrics: dict | None = None,
    **extra,
) -> dict:
    """An arena evaluation report for a model."""
    return {"type": EVAL, "ref": ref, "model": model, "panel": panel or [], "metrics": metrics or {}, **extra}


def verdict(
    ref: str,
    *,
    gate: str,
    win_rate: float | None = None,
    ci: list | None = None,
    n: int | None = None,
    note: str = "",
    **extra,
) -> dict:
    """A gate decision (PROMOTE / REVERT / AMBIGUOUS) over an eval."""
    return {"type": VERDICT, "ref": ref, "gate": gate, "win_rate": win_rate, "ci": ci, "n": n, "note": note, **extra}


def event(scope: str, summary: str, *, data: dict | None = None, **extra) -> dict:
    """A log-only note (no entity state); the cockpit's narration."""
    return {"type": EVENT, "scope": scope, "summary": summary, "data": data or {}, **extra}


# ---- I/O ----------------------------------------------------------------

def _ordered(row: dict) -> dict:
    lead = {k: row[k] for k in _LEAD if k in row}
    rest = {k: v for k, v in row.items() if k not in lead}
    return {**lead, **rest}


def append(path: str | os.PathLike, row: dict, *, ts: str | None = None) -> dict:
    """Append one row, assigning a monotonic ``seq`` + ``ts`` under an flock.

    Returns the stored row (with ``seq``/``ts`` filled in). Never mutates an
    existing line — that is the whole contract. ``seq`` is the row's ordinal in
    the file (= line count before the write), so it is stable forever.
    """
    path = os.fspath(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    out = dict(row)
    out["ts"] = ts or out.get("ts") or now_iso()
    # a+ so the file is created if missing and we can read its current length
    with open(path, "a+", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            seq = sum(1 for _ in f)
            out["seq"] = seq
            f.write(json.dumps(_ordered(out), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return out


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    return json.loads(line)


def read_all(path: str | os.PathLike) -> list[dict]:
    """Read every row from the ledger (missing file → empty).

    Tail-tolerant: a power-pull (or process death) mid-``append`` can leave a
    truncated FINAL line on disk — ``append`` fsyncs, but the kill can land
    between the partial ``write`` and that fsync. Tolerate ONLY that torn last
    line, so a single bad byte can never brick every loop on restart. An interior
    malformed line is real corruption and MUST surface (silently dropping it would
    lose a committed row — the one thing an append-only ledger must never do)."""
    path = os.fspath(path)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    rows = []
    for i, ln in enumerate(lines):
        try:
            r = parse_line(ln)
        except (ValueError, json.JSONDecodeError):
            if i == len(lines) - 1:
                break            # a torn trailing line from a power-pull — drop it
            raise                # interior corruption — never swallow a committed row
        if r is not None:
            rows.append(r)
    return rows


# ---- the reducer --------------------------------------------------------

@dataclass
class LedgerState:
    """The current state after replaying every row in order."""

    experiments: dict[str, dict] = field(default_factory=dict)
    evals: dict[str, dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def claimable(self, role: str, now: _dt.datetime) -> list[dict]:
        """Experiments a ``role`` daemon may claim right now: ``open``, or
        ``claimed`` with an expired lease and no result (crash recovery)."""
        out = []
        for e in self.experiments.values():
            if e.get("role") != role:
                continue
            st = e.get("status")
            if st == OPEN:
                out.append(e)
            elif st == CLAIMED and "result" not in e:
                lu = (e.get("claim") or {}).get("lease_until")
                if lu and parse_ts(lu) < now:
                    out.append(e)
        return out

    def pick(
        self,
        role: str,
        now: _dt.datetime,
        priority_fn: Callable[[dict], object] | None = None,
    ) -> dict | None:
        """First-priority claimable experiment for ``role`` (or None)."""
        cands = self.claimable(role, now)
        if not cands:
            return None
        return max(cands, key=priority_fn or default_priority)


def default_priority(e: dict):
    """Default policy: highest ``priority``, then never-*completed* (starvation
    floor — a lane that has produced no result yet outranks one that has), then
    oldest (lowest ``seq``). Keys on ``_results`` not ``_claims`` because the
    daemon no longer writes claim rows (singleton via flock; recovery via
    re-pick). The clean seam for P5's Δelo/hr ``pick_priority`` is just passing a
    different ``priority_fn`` to ``pick``."""
    return (e.get("priority", 0) or 0, 1 if e.get("_results", 0) == 0 else 0, -e.get("seq", 0))


def fold(rows: Iterable[dict]) -> LedgerState:
    """Replay rows in order into current state. Corrections apply
    last-writer-wins; claims/results drive the status lifecycle."""
    st = LedgerState()
    exps, evals = st.experiments, st.evals
    for row in rows:
        t = row.get("type")
        if t == EXPERIMENT:
            e = dict(row)
            e.setdefault("status", OPEN)
            e["_claims"] = 0
            e["_results"] = 0
            e["_corrections"] = 0
            exps[row["id"]] = e
        elif t == CLAIM:
            e = exps.get(row.get("ref"))
            if e is not None:
                e["status"] = CLAIMED
                e["claim"] = {"by": row.get("by"), "lease_until": row.get("lease_until"), "seq": row.get("seq")}
                e["_claims"] += 1
        elif t == RESULT:
            e = exps.get(row.get("ref"))
            if e is not None:
                e["status"] = row.get("status", DONE)
                e["result"] = {k: row.get(k) for k in ("model", "buffer", "metrics", "wall_s", "error")}
                # the seq the evidence LANDED at (not the experiment's creation seq)
                # — the watermark for "decide once per evidence arrival" (#61).
                e["result"]["_seq"] = row.get("seq")
                e["_results"] += 1
        elif t == CORRECTION:
            tgt = exps.get(row.get("ref")) or evals.get(row.get("ref"))
            if tgt is not None:
                tgt.update(row.get("set") or {})
                tgt["_corrections"] = tgt.get("_corrections", 0) + 1
        elif t == EVAL:
            evals[row["ref"]] = dict(row)
        elif t == VERDICT:
            ev = evals.get(row.get("ref"))
            if ev is not None:
                ev["verdict"] = {k: row.get(k) for k in ("gate", "win_rate", "ci", "n", "note")}
                ev["verdict"]["_seq"] = row.get("seq")   # evidence-landing seq (#61 watermark)
        elif t == EVENT:
            st.events.append(dict(row))
    return st


# ---- a thin convenience handle ------------------------------------------

class Ledger:
    """Bound to one ledger path; the obvious read/append/fold/pick handle."""

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)

    def append(self, row: dict, *, ts: str | None = None) -> dict:
        return append(self.path, row, ts=ts)

    def read_all(self) -> list[dict]:
        return read_all(self.path)

    def fold(self) -> LedgerState:
        return fold(self.read_all())

    def pick(self, role: str, *, now: _dt.datetime | None = None, priority_fn=None) -> dict | None:
        return self.fold().pick(role, now or utcnow(), priority_fn)


# ---- show CLI (the eyeball view; the real cockpit is P6 status.py) -------

def _fmt_exp(e: dict) -> str:
    base = e.get("base", "scratch")
    base = base if len(str(base)) <= 28 else "…" + str(base)[-27:]
    extra = ""
    if e.get("status") == CLAIMED:
        extra = f"  lease<{(e.get('claim') or {}).get('lease_until', '?')}>"
    elif "result" in e:
        m = e["result"].get("model") or ""
        extra = f"  -> {m}" if m else ""
    return (f"  #{e.get('seq'):<4} [{e.get('status',''):<7}] p{e.get('priority',0):<3} "
            f"{e.get('id',''):<16} {e.get('role',''):<6} base={base}{extra}")


def _cmd_show(args) -> int:
    rows = read_all(args.ledger)
    st = fold(rows)
    print(f"=== autolab ledger: {args.ledger} ===")
    print(f"{len(rows)} rows · {len(st.experiments)} experiments · "
          f"{len(st.evals)} evals · {len(st.events)} events")
    order = {OPEN: 0, CLAIMED: 1, FAILED: 2, DONE: 3}
    exps = sorted(st.experiments.values(),
                  key=lambda e: (order.get(e.get("status"), 9), -e.get("priority", 0), e.get("seq", 0)))
    if args.role:
        exps = [e for e in exps if e.get("role") == args.role]
    for e in exps:
        print(_fmt_exp(e))
    for role in sorted({e.get("role") for e in st.experiments.values()} - {None}):
        nxt = st.pick(role, utcnow())
        print(f"  next[{role}]: {nxt['id'] if nxt else '(idle)'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("show", help="dump folded ledger state")
    ps.add_argument("--ledger", required=True, help="path to the ledger JSONL")
    ps.add_argument("--role", help="filter experiments to one role")
    ps.set_defaults(func=_cmd_show)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
