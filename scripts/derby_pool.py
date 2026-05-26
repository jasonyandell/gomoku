#!/usr/bin/env python3
"""derby_pool — the open-entry candidate pool for the derby broker.

The Δelo Derby races training recipes ("cells") against each other on a single
serial GPU lane. To make it **OPEN-ENTRY**, researcher agents (and humans) need a
place to *register* candidate cells without touching the live run; a broker then
*claims* them as running-pool slots free up. This module is that producer/consumer
contract — deliberately boring, file-backed, and greppable.

Storage (mirrors the gpu_daemon maildir style — the file's *content* carries the
state, and every write is atomic so a crash never leaves a half-written record):

    sweep_runs/derby_pool/
      candidates/
        <name>.json        # one file per candidate; status in the JSON body

Candidate schema::

    {"name": "vct-teacher", "cell": "derby-x-vct", "lever": "+ VCT teacher signal",
     "status": "available", "submitted_by": "researcher:<id>", "submitted_at": <unix_ts>,
     "claimed_at": null, "retired_at": null, "retire_reason": null}

Lifecycle:  available --claim--> running --retire--> retired  (retire works from
any state). `claim()` is **double-claim-safe**: two brokers racing the same
candidate — exactly one wins, the loser gets None. The lock is an atomic
`os.rename` of the candidate file to a private `.claiming` marker: only the
process whose rename succeeds owns the transition; a second rename of the
now-missing source raises FileNotFoundError and is treated as "already taken".

CLI (style matches scripts/lab_log.py)::

    python scripts/derby_pool.py register --name vct-teacher --cell derby-x-vct \
        --lever "+ VCT teacher signal" [--by researcher:42]
    python scripts/derby_pool.py list [--status available|running|retired]
    python scripts/derby_pool.py claim [--name vct-teacher]
    python scripts/derby_pool.py retire --name vct-teacher --reason "superseded by v9"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_POOL_DIR = Path(
    os.environ.get("GOMOKU_DERBY_POOL", str(REPO_ROOT / "sweep_runs" / "derby_pool"))
)

STATUSES = ("available", "running", "retired")


# ---------------------------------------------------------------------------
# small helpers (atomic_write_json mirrors gpu_daemon / delo_derby)
# ---------------------------------------------------------------------------
def atomic_write_json(path: Path, obj: Any) -> None:
    """tmp + fsync + rename so a crash never leaves a partial candidate file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _pool_dir(pool_dir: Optional[Path]) -> Path:
    return Path(pool_dir) if pool_dir is not None else DEFAULT_POOL_DIR


def _candidates_dir(pool_dir: Optional[Path]) -> Path:
    d = _pool_dir(pool_dir) / "candidates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _candidate_path(name: str, pool_dir: Optional[Path]) -> Path:
    return _candidates_dir(pool_dir) / f"{name}.json"


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------
def register(
    name: str,
    cell: str,
    lever: str,
    by: str = "unknown",
    pool_dir: Optional[Path] = None,
) -> dict:
    """Create a new candidate (status='available'). Raises if `name` exists."""
    path = _candidate_path(name, pool_dir)
    if path.exists():
        raise ValueError(f"candidate {name!r} already exists at {path}")
    spec = {
        "name": name,
        "cell": cell,
        "lever": lever,
        "status": "available",
        "submitted_by": by,
        "submitted_at": int(time.time()),
        "claimed_at": None,
        "retired_at": None,
        "retire_reason": None,
    }
    # Create exclusively (O_EXCL) so two concurrent registers of the same name
    # cannot both win, then atomically replace with the full record.
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise ValueError(f"candidate {name!r} already exists at {path}")
    os.close(fd)
    atomic_write_json(path, spec)
    return spec


def list_candidates(
    status: Optional[str] = None, pool_dir: Optional[Path] = None
) -> list[dict]:
    """All candidates, or only those with the given status. Sorted by submitted_at."""
    out: list[dict] = []
    for p in sorted(_candidates_dir(pool_dir).glob("*.json")):
        spec = _read(p)
        if spec is None:
            continue
        if status is not None and spec.get("status") != status:
            continue
        out.append(spec)
    out.sort(key=lambda s: (int(s.get("submitted_at", 0)), str(s.get("name", ""))))
    return out


def claim(name: Optional[str] = None, pool_dir: Optional[Path] = None) -> Optional[dict]:
    """Atomically flip an available candidate available->running.

    With `name` set, claim that candidate; otherwise claim the OLDEST available
    one. Returns the updated spec, or None if there is nothing to claim / the
    candidate was already taken (double-claim guard).

    The lock is an atomic os.rename of the candidate file to a private
    `.claiming.<pid>.<uuid>` marker. POSIX rename is atomic; only one racer's
    rename of a given source succeeds, the rest raise FileNotFoundError. We then
    re-check status (defends against claiming an already-running file) before
    committing the running record back to <name>.json.
    """
    cdir = _candidates_dir(pool_dir)

    if name is None:
        avail = list_candidates(status="available", pool_dir=pool_dir)
        if not avail:
            return None
        candidates = [c["name"] for c in avail]  # oldest-first
    else:
        candidates = [name]

    for cand_name in candidates:
        src = cdir / f"{cand_name}.json"
        if not src.exists():
            continue
        marker = cdir / f"{cand_name}.json.claiming.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            os.rename(src, marker)  # the lock: atomic, single-winner
        except (FileNotFoundError, OSError):
            # someone else moved/claimed it first -> try the next candidate
            continue

        spec = _read(marker)
        if spec is None or spec.get("status") != "available":
            # not actually claimable (already running/retired or corrupt); put it back
            try:
                os.rename(marker, src)
            except OSError:
                pass
            continue

        spec["status"] = "running"
        spec["claimed_at"] = int(time.time())
        atomic_write_json(src, spec)  # restore the canonical path with running state
        try:
            os.unlink(marker)
        except OSError:
            pass
        return spec

    return None


def retire(name: str, reason: str, pool_dir: Optional[Path] = None) -> dict:
    """Flip a candidate to status='retired' from any state. Raises if unknown."""
    path = _candidate_path(name, pool_dir)
    spec = _read(path)
    if spec is None:
        raise ValueError(f"unknown candidate {name!r} (no file at {path})")
    spec["status"] = "retired"
    spec["retired_at"] = int(time.time())
    spec["retire_reason"] = reason
    atomic_write_json(path, spec)
    return spec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fmt_ts(ts: Any) -> str:
    if not ts:
        return "-"
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(int(ts)))
    except (ValueError, TypeError, OSError):
        return str(ts)


def cmd_register(args) -> int:
    try:
        spec = register(args.name, args.cell, args.lever, by=args.by,
                        pool_dir=args.pool_dir)
    except ValueError as e:
        print(f"register: {e}", file=sys.stderr)
        return 2
    print(f"registered {spec['name']}  cell={spec['cell']}  status={spec['status']}")
    print(f"  lever: {spec['lever']}")
    print(f"  -> {_candidate_path(spec['name'], args.pool_dir)}")
    return 0


def cmd_list(args) -> int:
    rows = list_candidates(status=args.status, pool_dir=args.pool_dir)
    label = args.status or "all"
    print(f"=== derby pool ({label}) — {len(rows)} candidate"
          f"{'' if len(rows) == 1 else 's'} ===")
    if not rows:
        return 0
    nw = max(4, max(len(r.get("name", "")) for r in rows))
    cw = max(4, max(len(r.get("cell", "")) for r in rows))
    print(f"  {'NAME':<{nw}}  {'CELL':<{cw}}  {'STATUS':<9}  {'SUBMITTED':<16}  LEVER")
    for r in rows:
        print(f"  {r.get('name',''):<{nw}}  {r.get('cell',''):<{cw}}  "
              f"{r.get('status',''):<9}  {_fmt_ts(r.get('submitted_at')):<16}  "
              f"{r.get('lever','')}")
    return 0


def cmd_claim(args) -> int:
    spec = claim(name=args.name, pool_dir=args.pool_dir)
    if spec is None:
        if args.name:
            print(f"claim: {args.name!r} not available (already claimed/retired or unknown)")
        else:
            print("claim: nothing available")
        return 1
    print(f"claimed {spec['name']}  cell={spec['cell']}  status={spec['status']}")
    print(f"  claimed_at: {_fmt_ts(spec['claimed_at'])}")
    return 0


def cmd_retire(args) -> int:
    try:
        spec = retire(args.name, args.reason, pool_dir=args.pool_dir)
    except ValueError as e:
        print(f"retire: {e}", file=sys.stderr)
        return 2
    print(f"retired {spec['name']}  reason: {spec['retire_reason']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR,
                   help=f"pool root (default {DEFAULT_POOL_DIR}; env GOMOKU_DERBY_POOL)")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("register", help="register a candidate cell into the pool")
    r.add_argument("--name", required=True, help="unique candidate name")
    r.add_argument("--cell", required=True, help="run_sweep cell key for this recipe")
    r.add_argument("--lever", required=True, help="the research lever, e.g. '+ VCT teacher'")
    r.add_argument("--by", default="unknown", help="submitter id, e.g. researcher:42")
    r.set_defaults(func=cmd_register)

    li = sub.add_parser("list", help="list candidates (optionally by status)")
    li.add_argument("--status", choices=list(STATUSES), default=None)
    li.set_defaults(func=cmd_list)

    cl = sub.add_parser("claim", help="claim the named (or oldest available) candidate")
    cl.add_argument("--name", default=None, help="specific candidate; omit for oldest available")
    cl.set_defaults(func=cmd_claim)

    rt = sub.add_parser("retire", help="retire a candidate (from any state)")
    rt.add_argument("--name", required=True)
    rt.add_argument("--reason", required=True, help="why it's being retired")
    rt.set_defaults(func=cmd_retire)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
