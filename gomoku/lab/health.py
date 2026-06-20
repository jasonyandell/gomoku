#!/usr/bin/env python3
"""Autolab health — the one pure place that answers "what's wrong right now?"

A single ``scan(state)`` over a folded ledger returns a list of ``Alert``s. The
monitor renders + notifies on them, ``status`` can show them, and the simulator
asserts on them — one detector, three consumers, no duplicated "is this bad?"
logic. Pure + stdlib-only (same contract as ``ledger``): no torch, no I/O.

Detected today:
  * ``stalled``   — a role has no claimable work AND its lane(s) dead-ended on a
                    FAILED result. The trainer would otherwise sleep forever with
                    a stranded ``latest.pt`` and no human ever told.
  * ``needs_jason`` — an eval/verdict explicitly flagged for a human decision
                    (e.g. the first-champion gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import ledger


@dataclass
class Alert:
    kind: str                       # "stalled" | "needs_jason"
    summary: str
    item: str | None = None
    data: dict = field(default_factory=dict)


def scan(state: "ledger.LedgerState", *, now=None) -> list[Alert]:
    """Return the current alerts for a folded ledger state (most-urgent-ish first)."""
    now = now or ledger.utcnow()
    alerts: list[Alert] = []

    # Stalled lanes: a role can pick nothing, yet it has a FAILED row. Nothing
    # open => the daemon will idle forever; the FAILED row says it's not "done".
    for role in ("train", "arena"):
        if state.pick(role, now) is not None:
            continue
        for e in state.experiments.values():
            if e.get("role") == role and e.get("status") == ledger.FAILED:
                alerts.append(Alert(
                    kind="stalled", item=e.get("id"),
                    summary=f"{role} lane stalled: {e.get('id')} FAILED, no open work — "
                            f"latest.pt stranded (reopen to resume)",
                    data={"needs_jason": True,
                          "error": (e.get("result") or {}).get("error")}))

    # Explicit human-decision flags raised by a role (first-champion gate, etc.).
    for ev in state.evals.values():
        m = ev.get("metrics") or {}
        if m.get("needs_jason"):
            alerts.append(Alert(
                kind="needs_jason", item=ev.get("ref"),
                summary=f"needs Jason: {ev.get('ref')} ({m.get('verdict')} vs {m.get('vs')})",
                data={"verdict": m.get("verdict"), "vs": m.get("vs")}))

    return alerts
