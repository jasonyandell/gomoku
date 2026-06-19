#!/usr/bin/env python3
"""``autolab status`` — the one-glance cockpit surface (epic #53).

Folds the whole ledger and probes each role's daemon lockfile into a single
per-lane board: what's queued, what's running, who owns it, and whether that
daemon is actually alive. The ledger is the dispatch surface; this is just its
read side (record ≠ report). Stdlib-only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys

from . import daemon, ledger

KNOWN_ROLES = ("train", "arena")

_STATUS_ORDER = {ledger.CLAIMED: 0, "running": 0, ledger.OPEN: 1,
                 ledger.FAILED: 2, ledger.DONE: 3}


def _roles_in(state: ledger.LedgerState) -> list[str]:
    seen = {e.get("role") for e in state.experiments.values()} - {None}
    return sorted(set(KNOWN_ROLES) | seen)


def lane_board(ledger_path: str, *, now: _dt.datetime | None = None) -> dict:
    state = ledger.fold(ledger.read_all(ledger_path))
    now = now or ledger.utcnow()
    daemons = {}
    for role in _roles_in(state):
        p = daemon.lockfile_path(role)
        daemons[role] = {"meta": daemon.read_lockfile(p), "alive": daemon.probe_alive(p)}
    return {"ledger": ledger_path, "state": state, "daemons": daemons, "now": now}


def _last_elo(e: dict) -> str:
    m = (e.get("result") or {}).get("metrics") or {}
    for k in ("eval/model_elo", "model_elo", "elo", "implied_elo"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            return f"{v:.0f}"
    return "-"


def _owner_tag(e: dict, daemons: dict) -> str:
    """If a live daemon names this experiment as its current item, tag it."""
    for role, d in daemons.items():
        meta = d.get("meta") or {}
        if meta.get("item") == e.get("id"):
            return f"  <- {role} {'ALIVE' if d['alive'] else 'DOWN'}"
    return ""


def format_board(board: dict) -> str:
    state, daemons = board["state"], board["daemons"]
    out = [f"=== autolab status: {board['ledger']} ===",
           f"{len(state.experiments)} experiments · {len(state.evals)} evals · "
           f"{len(state.events)} events"]
    out.append("daemons:")
    for role, d in daemons.items():
        meta = d.get("meta") or {}
        if d["alive"]:
            item = meta.get("item") or "(idle)"
            out.append(f"  {role:<6} ALIVE  pid {meta.get('pid','?')}  on {item}  "
                       f"since {meta.get('started_at','?')}")
        elif meta:
            out.append(f"  {role:<6} DOWN   (last: pid {meta.get('pid','?')} "
                       f"on {meta.get('item')})")
        else:
            out.append(f"  {role:<6} DOWN")
    out.append("lanes:")
    exps = sorted(state.experiments.values(),
                  key=lambda e: (_STATUS_ORDER.get(e.get("status"), 9),
                                 -(e.get("priority", 0) or 0), e.get("seq", 0)))
    for e in exps:
        out.append(f"  #{e.get('seq'):<4} [{e.get('status',''):<7}] "
                   f"p{e.get('priority', 0):<3} {e.get('id',''):<16} "
                   f"{e.get('role',''):<6} elo={_last_elo(e)}{_owner_tag(e, daemons)}")
    nexts = " ".join(f"next[{r}]: "
                     f"{(state.pick(r, board['now']) or {}).get('id', '(idle)')}"
                     for r in daemons)
    out.append("  " + nexts)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=daemon.default_ledger_path(),
                    help="path to the ledger JSONL (default: $AUTOLAB_HOME/ledger.jsonl)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)
    board = lane_board(args.ledger)
    if args.json:
        st = board["state"]
        print(json.dumps({
            "ledger": board["ledger"],
            "experiments": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                            for k, v in st.experiments.items()},
            "daemons": {r: {"alive": d["alive"], "meta": d["meta"]}
                        for r, d in board["daemons"].items()},
        }, indent=2, default=str))
    else:
        print(format_board(board))
    return 0


if __name__ == "__main__":
    sys.exit(main())
