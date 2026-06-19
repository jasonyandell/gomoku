"""The autolab substrate (epic #53).

A small, stdlib-only layer that turns the existing training/eval pieces into a
self-driving lab on Jason's M5 Max:

  * ``ledger`` — THE SPINE. One external, out-of-git, append-only JSONL file
    that every loop reads to pick first-priority work and appends results to.
    Corrected financial-journal style (never edit a row; append a ``correction``
    that supersedes by ``ref``). A reducer replays the rows into current state.

Later phases add the shared daemon contract (``daemon``), the trainer/arena
roles, the research loop, and the cockpit status surface — all of them just
read+append the ledger. See ``wiki/topics/autolab-architecture.md``.
"""
