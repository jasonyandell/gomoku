#!/usr/bin/env python3
"""``actionable(ledger)`` — the substrate's one pure read: what's to do, for whom.

The generalization of ``health.scan`` (epic #53 doctrine,
``wiki/topics/autolab-doctrine.md`` §3). A single pure fold of the ledger that
answers, deterministically and cheaply, **"is there something to do, and what is
available?"** across every lane — the *deterministic WHEN*. The *intelligent
WHAT* (which commit, which idea, promote-or-not) is a worker's job; this function
only says *there is a decision to make here, and here is its evidence.*

    actionable(state) -> Actionable(
        train     = next claimable train item  (or None)        # dumb worker
        arena     = next claimable arena item  (or None)        # dumb worker
        research  = [threads whose evidence landed, undecided]   # smart worker
        alerts    = [things needing a human]   (health.scan)     # human worker
    )

``train`` / ``arena`` are *exactly* ``state.pick(role)`` — this molecule never
invents a second pick policy; it bundles the existing atom so every dispatch
consumer (monitor, status, a cron trigger, the research reconciler) reads ONE
surface and cannot drift from the daemon. The simulator asserts
``actionable(state).train is state.pick("train")``.

Triggers are pluggable *because* this is the truth (doctrine §5): a cron tick ≡ an
event-append ≡ a manual run all do the identical thing — call ``actionable()`` and
hand a lane to a worker. Kill the loop and you lose timeliness, not correctness;
the next trigger of any kind reconciles from the log.

Pure + stdlib-only (same contract as ``ledger`` / ``health``): no torch, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import health, ledger, research


@dataclass
class Actionable:
    """The folded ledger's full work-surface at one instant."""

    train: dict | None = None
    arena: dict | None = None
    research: list["research.Thread"] = field(default_factory=list)
    alerts: list["health.Alert"] = field(default_factory=list)

    def for_role(self, role: str) -> dict | None:
        """The dumb-worker item for a role name (so a trigger can ask generically)."""
        return {"train": self.train, "arena": self.arena}.get(role)

    @property
    def idle(self) -> bool:
        return not (self.train or self.arena or self.research or self.alerts)


def actionable(state: "ledger.LedgerState", *, now=None) -> Actionable:
    """The one pure read of the folded ledger — every lane's next move + alerts."""
    now = now or ledger.utcnow()
    return Actionable(
        train=state.pick("train", now),
        arena=state.pick("arena", now),
        research=research.research_threads(state),
        alerts=health.scan(state, now=now),
    )
