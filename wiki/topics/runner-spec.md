# The runner — spec v0.2 (one queue, one lane, an append-only log)

> **Status: DRAFT v0.2** *(2026-07-05)* — simplified per the "throw work
> at it and it runs, eventually" mandate (Jason + Fable session). One
> queue, no tags, no priorities; the spool is replaced by an append-only
> log that is both the queue and the history; multi-step **composite
> items** replace resource routing. Working name: **the runner** —
> deliberately boring; the pattern is a batch spool, nothing newer
> than 1983.
>
> The test this page must pass: the normative core (§2–§9) never
> mentions the domain. If it can't, the library thesis is false.

## 1. One breath

A library — not a framework — with one job: **work you submit runs,
eventually.** You append a fully-specified item to a durable log and get
a ticket. A dumb dispatcher executes items **one at a time, in log
order** — each item a sequence of steps sharing a working directory,
each step time-boxed. Later you check your ticket: exactly one sentinel
says how it ended, and your opaque payload comes back verbatim — a
courtesy, never interpreted. All state is the log plus the filesystem;
every participant can be killed at any instruction and the system stays
decidable. The caller owns policy and meaning; the runner owns order,
exclusion, and witness. **Continuations are data, never callbacks: the
runner never calls, never imports, never interprets caller code.**

## 2. Assumptions (explicit, load-bearing)

- **Single host, single POSIX filesystem.** Atomicity primitives:
  append-under-flock, `mkdir`, `link`, `flock`.
- **Crashy but non-adversarial processes** (same uid; tamper-prevention
  out of scope).
- **No sandbox.** Steps run with full process authority. Named
  limitation, not an oversight.
- **Items are re-runnable.** The caller must tolerate re-execution of
  work whose first attempt may in fact have completed (B1, §10).

## 3. The store

```
<root>/
  queue.log                   THE append-only log: every submit and
                              cancel record, forever — queue AND history
  locks/
    queue.lock                serializes appends to queue.log
    dispatch.lock             single-dispatcher guard
    exec.lock                 THE one lane; fd-inherited by children
  exhaust/<run_id>/
    wrapper.pid               {pid, pid_start_time}; link-if-absent
    child.<n>.pid             per step; {pid, pgid, pid_start_time}
    step.<n>.out.log          step n stdout+stderr, merged
    wrapper.log               wrapper's own stderr
    exit.json                 THE sentinel; exactly one writer ever
    ctx/                      shared cwd for ALL steps — scratch,
                              outputs, and the pipe between steps
```

`run_id`: caller-supplied, `[a-z0-9-]{1,64}`. Uniqueness is the caller's
promise; the fold takes the **first** submit record per run_id and
ignores the rest — duplicate submission is harmless by construction.

## 4. Format 1 — the log records

One JSON object per line, appended under `flock(queue.lock)` + fsync.
Two kinds, closed vocabulary:

```jsonc
{"kind": "submit", "schema": 1,
 "run_id": "…",
 "steps": [                    // a COMPOSITE: run in order, stop at
   {"argv": ["…"],             // the first step that fails its exit
    "env": {"K": "v"},         // or its box; all steps share ctx/
    "box_s": 3600,
    "kill_grace_s": 30}
 ],
 "continuation": { },          // OPAQUE; ≤ 64 KiB; returned verbatim
 "note": "free text"}          // at harvest as a courtesy; never read

{"kind": "cancel", "schema": 1, "run_id": "…"}
```

**Env rule:** each step's environment is its `env` ∪ a fixed minimal
base `{PATH, HOME, TMPDIR, UV_CACHE_DIR}` — *nothing else is inherited*.
Pass secrets by reference; the runner does not solve secrets.

**Argv rule:** exec'd verbatim. "uv-compatible, SHA-pinned" is the
*caller's* convention; the runner does not know git or uv exist.

**Steps rule:** steps communicate **through `ctx/`** — step N writes,
step N+1 reads its cwd. The runner never parameterizes an argv from a
prior step's output; that convention belongs to the caller's scripts.

## 5. Format 2 — the queue, and the dispatcher contract

**The queue is a fold over the log.** An item's state is derived, never
stored: *submitted* = first submit record for its run_id; *canceled* = a
cancel record exists and no execution began; *done* = `exit.json`
exists. **The frontier** = the oldest submit record, in log order, that
is neither done nor canceled.

**Dispatch** (`dispatch(root)`, kicked by cron, by a fresh submit, by
anything; cadence is never load-bearing for safety):

```
D1  flock(locks/dispatch.lock, NB); held elsewhere → return
D2  probe locks/exec.lock — held → return (something is running, or an
    orphaned child still owns the lane; conservative either way)
D3  settle the settleable (see §8): link canceled / crashed sentinels
D4  frontier item exists → spawn its wrapper, detached; return
    (spawn is idempotent: a wrapper that finds exec.lock held or
     wrapper.pid present exits silently; dispatch simply tries again
     next kick — a dead spawn costs nothing and loses nothing)
```

**The queue is dumb forever — constitutional.** Strict FIFO by log
order. No priority field, no reordering, no deadlines, no tags, ever.
One queue; one lane. A FIFO server makes no decisions: order is fully
determined by what callers appended and when, so *policy cannot live
here*. Callers wanting scheduling power enqueue just-in-time (queue
depth ≈ 1); callers that don't care batch-append and accept arrival
order. Serialization of *anything* — a GPU, a repo, a wiki — is the same
mechanism: it all goes through the one lane, in order. The day this
section grows a second queue, the runner has become a framework and this
page has failed.

## 6. Format 3 — the wrapper contract

One detached wrapper per item attempt (double-fork, `setsid`, stdio →
`wrapper.log`). Small, dumb, hardened — the durable witness:

```
W1  read the item (frontier submit record)
W2  open locks/exec.lock; flock(LOCK_EX|LOCK_NB)
      on failure → exit silently (the lane is busy; dispatch retries)
W3  mkdir exhaust/<run_id> if absent; write wrapper.pid
      (link-if-absent; loser exits silently)
W4  for each step n, in order:
      spawn child: new pgid; cwd=ctx/; env per §4;
        stdout+stderr → step.<n>.out.log;
        the child INHERITS THE exec.lock FD (not close-on-exec)
      write child.<n>.pid
      wait for child exit, racing the step's box:
        at box_s:           SIGTERM to the child's process group
        at +kill_grace_s:   SIGKILL to the group
        still unreapable after +60s → exit.json{wedged, step: n}; exit
      nonzero exit → stop; killed by box → stop
W5  link exit.json{outcome, steps: [results…]}; exit
```

**The lock rides the fd (W4).** `flock` belongs to the open file
description, so the lane's lifetime is *the running child's*, not the
wrapper's: a murdered wrapper cannot leak the lane while its orphaned
child still computes, and an unkillable (wedged) child **keeps the
lane** — a poisoned machine quarantines itself at the kernel, with no
policy code. Conservative failure mode: a straggling grandchild that
inherited the fd holds the lane too long. Exclusion errs toward
exclusion.

## 7. Format 4 — `exit.json`, the sentinel

```jsonc
{"schema": 1, "run_id": "…",
 "outcome": "completed"        // every step exited 0
          | "failed"           // a step exited nonzero  {step, exit_code}
          | "killed_box"       // a step overran its box {step}
          | "wedged"           // a step would not die   {step}
          | "crashed"          // wrapper died mid-item; see B1
          | "canceled",
 "steps": [ {"exit_code": 0, "signal": null, "box_used_s": 812.4}, … ],
 "started_at": "…", "ended_at": "…",
 "written_by": "wrapper" | "harvest" | "dispatch"}
```

**Exactly-one-writer mechanics:** write `exit.json.tmp.<pid>` in full,
then `link(tmp, "exit.json")` — `EEXIST` means you lost; delete your
tmp. `rename` clobbers; `link` is the first-wins primitive. Immutable
forever after.

## 8. Format 5 — the status decision procedure

A **total function** of (log, exhaust); no in-process state anywhere.
`alive(pidfile)` ⇔ pid exists **and** its start-time matches the
recorded one (defeats pid reuse).

```
no submit record                               → ABSENT
exit.json valid                                → DONE(outcome)
else wrapper.pid ∧ wrapper alive               → RUNNING
else any child.<n>.pid alive                   → ORPHANED
else wrapper.pid (all dead)                    → settle: link
                                                 exit.json{crashed} → DONE
else cancel record present                     → settle: link
                                                 exit.json{canceled} → DONE
else                                           → QUEUED
```

There is no LAUNCHING and no stillborn: **an item is either in the log
or it isn't.** A wrapper that dies before `wrapper.pid` left nothing
behind; the item is simply still QUEUED and dispatch spawns another
wrapper next kick. Half-born states died with the log.

**ORPHANED** is deliberate: wrapper dead, a child computing, box no
longer enforced — but the lane is still held (the fd), so nothing else
starts. The runner *reports* and refuses to decide — killing versus
waiting is caller policy (on some hosts, killing mid-GPU-compile wedges
the machine; the runner cannot know that; the caller does).

## 9. The surface — a CLI, because bash is simpler

A uv-runnable CLI over six verbs (the library functions underneath are
the same six):

```
runner submit  [spec.json|-]   → run_id          (append + kick dispatch)
runner status  <run_id>        → QUEUED|RUNNING|ORPHANED|DONE(outcome)|ABSENT
runner harvest                 → JSON lines: every DONE item (id,
                                 outcome, steps, continuation — verbatim,
                                 as a courtesy). Caller dedups; the
                                 watermark belongs to the viewer.
runner dispatch                → one pass per §5; safe to kick anytime
runner cancel  <run_id>        → appended; effective unless running
runner queue                   → frontier, depth, lane state (board food)
```

No priorities. No retries. No GC. No push — completion notification is
polling; **never** execution of caller code. Anything resembling policy
that appears here means the page has failed.

## 10. Claims and blemishes

- **C1 — nothing is lost.** A submitted, uncanceled item reaches a
  terminal, eventually: the log is durable (append + fsync), the
  frontier is derived, dispatch is idempotent and respawns dead
  attempts. This is the mission: *throw work at it; it runs.* Progress
  needs dispatch cadence — cadence, never load-bearing for safety.
- **C2 — single terminal.** At most one `exit.json` can ever exist per
  item (link-if-absent); §8 is total.
- **C3 — one lane.** At most one item executes at a time,
  kernel-enforced, tied to the actual running child via fd inheritance;
  survives every row of §11, erring toward exclusion.
- **C4 — idempotent submit.** First submit record per run_id wins;
  duplicates are ignored by the fold. At-least-once callers get
  exactly-once semantics.
- **C5 — statelessness.** All state is (log, exhaust); every participant
  is killable at any instruction with §8 remaining total.
- **C6 — FIFO by log order,** explicit and immutable — never mtime
  vibes, never reordered.
- **B1 — `crashed` may mask `completed`.** Wrapper death after the last
  step exits but before W5 records `crashed` for work that finished —
  the classic RPC exactly-once impossibility. Callers MUST read
  `crashed` as "unknown — possibly completed"; `ctx/` remains for
  forensics; a step-written done-marker in `ctx/` is the caller-side
  disambiguator.
- **B2 — ORPHANED items are unboxed** until settled; settling is caller
  policy (§8). The box is a wrapper service, not a kernel one. The lane,
  however, stays held — orphans block, they don't leak.
- **Non-claims:** no concurrency, no fairness, no sandbox, no retention,
  no multi-host, no content validation, no secret handling, no push.

## 11. Failure enumeration (kill anything at any line)

| killed at | observable state | settled by | outcome |
|---|---|---|---|
| submit, before append | nothing | — | ABSENT; caller retries, idempotent |
| submit, after append | QUEUED | dispatch | runs eventually (C1) |
| dispatch, any line | dispatch.lock releases with it | next kick | unchanged |
| wrapper, before wrapper.pid | nothing durable | next kick | respawned; QUEUED throughout |
| wrapper, mid-steps (children die too) | wrapper.pid, all dead | harvest/dispatch | crashed |
| wrapper, mid-step (child survives) | child alive, lane held | caller policy | ORPHANED → crashed/… |
| a step (any cause) | wrapper reaps | wrapper | failed / killed_box |
| wrapper, after last step before W5 | all dead, no sentinel | harvest | **crashed — see B1** |
| cancel racing execution | first-wins on exit.json | — | canceled XOR runs |
| harvest / any caller loop | nothing — it holds no state | next kick | unchanged |

## 12. The first caller — notes across the seam

Everything below the line is domain; the seam is exactly here.

- **Enqueue discipline is the caller's scheduler.** Shares, niceness,
  admission — all implemented as *what the caller chooses to append, and
  when*, keeping queue depth ≈ 1 for full control.
- **Composite items are the pipeline:** smoke → train → eval as one
  item, piping through `ctx/`. Serialization of code landings and wiki
  commits needs no special machinery — they're just items, totally
  ordered with everything else.
- **Sibling clients compose:** anything cron-shaped may append when
  `runner queue` shows an empty frontier; the log records who asked for
  what, forever.
- **Pin before submit:** SHAs referenced in argv must stay reachable, or
  env materialization fails later, against the box.
- **Cold-SHA env builds burn lane time** at the head of an item; warm
  the uv cache during authoring, or accept the cost at laptop scale.
- **Anomaly policy:** `wedged` → the lane self-quarantines; dead-letter
  the human. A frontier that never advances is the one vital sign that
  matters — watch it.

## 13. Cross-refs

[autolab design v6](autolab-design-v6.md) — the first client ·
[red-team A1–A26](autolab-design-adversarial-review.md) ·
[doctrine](autolab-doctrine.md).
