# The runner — substrate spec v0.1 (a spool, five formats, a wrapper)

> **Status: DRAFT v0.1** *(2026-07-05)* — the brutally-simplified substrate
> beneath the autolab. v0.1 adds the **spool**: the queue and the semaphore
> both live on the server side of the call, per the "mundane job spool"
> reframe (Jason + Fable session). v0's busy-signal design is superseded —
> `contended` is now an anomaly, not a normal outcome. Not an autolab
> document: the autolab ([design v5](autolab-design-v5.md)) is the first
> caller. Working name: **the runner** — deliberately boring; the pattern
> is `at(1)`/`lpd`/sendmail-spool, nothing newer than 1983.
>
> The test this page must pass: the normative core (§2–§9) never mentions
> the domain. If it can't, the library thesis is false.

## 1. One breath

A library — not a framework — that runs queued work one-at-a-time per
resource, durably. You submit a fully-specified command with a time box
and an opaque payload; you get a ticket. A dumb dispatcher drains the
spool in strict FIFO order per resource tag. Later you check your ticket:
exactly one sentinel says how it ended, and your payload comes back
verbatim — **a courtesy, never interpreted**. All state is the
filesystem; every participant can be killed at any instruction and the
system stays decidable. The caller owns policy and meaning; the runner
owns order, exclusion, and witness. **Continuations are data, never
callbacks: the runner never calls, never imports, never interprets caller
code.**

## 2. Assumptions (explicit, load-bearing)

- **Single host, single POSIX filesystem.** Atomicity primitives:
  `mkdir`, `link`, `rename`, `flock`. Local mtimes suffice for grace
  windows.
- **Crashy but non-adversarial processes** (same uid; tamper-prevention
  out of scope).
- **No sandbox.** The child runs with full process authority. Named
  limitation, not an oversight.
- **Runs are re-runnable.** The caller must tolerate re-executing a run
  whose first attempt may in fact have completed (B1, §11).

## 3. The store

```
<root>/
  locks/
    dispatch.lock             single-dispatcher guard
    <tag>.lock                one per resource tag
  spool/<tag>/
    .seq                      monotonic counter file (guarded by .seq.lock)
    <seq08>-<run_id>          one entry per queued run; empty marker file
  exhaust/<run_id>/
    spec.json                 written by submit(); immutable
    claimed                   spool entry, renamed here by dispatch()
    wrapper.pid               {pid, pid_start_time}; link-if-absent
    child.pid                 {pid, pgid, pid_start_time}; link-if-absent
    wrapper.log               wrapper's own stderr
    out.log                   child stdout+stderr, merged
    exit.json                 THE sentinel; exactly one writer ever
    ctx/                      child cwd — scratch, outputs, the
                              continuation channel between stages
```

`run_id`: caller-supplied, `[a-z0-9-]{1,64}`. Uniqueness is the caller's
promise; `mkdir exhaust/<run_id>` is the kernel-enforced check.

## 4. Format 1 — `spec.json`

```jsonc
{
  "schema": 1,
  "run_id": "…",              // must equal the directory name
  "argv": ["uvx", "--from", "git+file:///…/repo@<sha>", "cmd", "…"],
  "env": {"EXAMPLE": "1"},    // strings only; see env rule
  "box_s": 3600,              // wall-clock budget, includes env build
  "kill_grace_s": 30,         // TERM → KILL gap
  "resource": "gpu",          // ONE tag, or null = run immediately,
                              // unqueued, concurrently
  "continuation": { },        // OPAQUE; ≤ 64 KiB; returned verbatim at
                              // harvest as a courtesy; never read
  "note": "free text"         // human breadcrumb, optional
}
```

**Env rule:** the child's environment is `spec.env` ∪ a fixed minimal
base `{PATH, HOME, TMPDIR, UV_CACHE_DIR}` — *nothing else is inherited*.
Pass secrets by reference (child looks them up); the runner does not
solve secrets.

**Argv rule:** exec'd verbatim. "uv-compatible, SHA-pinned" is the
*caller's* convention; the runner does not know git or uv exist.

## 5. Format 2 — the spool, and the dispatcher contract

**Enqueue** (inside `submit()`, after `spec.json` exists): under
`flock(spool/<tag>/.seq.lock)`, read-increment-write `.seq`, create the
empty entry `spool/<tag>/<seq08>-<run_id>`. Runs with `resource: null`
skip the spool — submit spawns their wrapper immediately.

**Dispatch** (`dispatch(root)`, kicked by cron or by anything; cadence is
never load-bearing for safety):

```
D1  flock(locks/dispatch.lock, NB); held elsewhere → return (single
    instance; the lock rides the pass)
D2  for each tag with spool entries, oldest sequence first:
      a. probe locks/<tag>.lock — held → skip tag (a straggling
         grandchild holding the fd also lands here; conservative)
      b. rename spool/<tag>/<seq>-<id> → exhaust/<id>/claimed
         (atomic; loser of a race with cancel() skips the entry)
      c. spawn the wrapper, detached
      d. wait ≤ 5s for wrapper.pid or exit.json to appear, then move on
         (the wrapper takes the tag lock BEFORE writing wrapper.pid, so
         observing wrapper.pid means the tag is truly occupied — this
         closes the double-launch race window)
D3  exit; the dispatch lock releases with the process
```

**The spool is dumb forever — constitutional.** Strict FIFO per tag. No
priority field, no reordering, no deadlines, ever. A FIFO server makes no
decisions: order is fully determined by what callers enqueued and when,
so *policy cannot live here*. Callers wanting scheduling power keep their
own queue and enqueue just-in-time (spool depth ≈ 1); callers that don't
care batch-enqueue and accept arrival order. The day this section grows a
`priority` field, the runner has become a framework and this page has
failed.

**Cancel** (`cancel(run_id)`): queue-only. Atomically rename the spool
entry aside and link `exit.json{canceled}`. If the entry is gone
(claimed, running, done) → `TooLate`; killing running work is the
caller's affair — it owns the process table.

## 6. Format 3 — the wrapper contract

`submit()` (null-resource) or `dispatch()` spawns one detached wrapper
per run (double-fork, `setsid`, stdio → `wrapper.log`). Small, dumb,
hardened — the durable witness:

```
W1  read spec.json
W2  if resource: open locks/<tag>.lock; flock(LOCK_EX|LOCK_NB)
      on failure → exit.json{contended}; exit   (anomaly — see §5 D2d)
W3  write wrapper.pid                (link-if-absent; loser exits)
W4  spawn child: new pgid; cwd=ctx/; env per §4; stdout+stderr→out.log;
      the child INHERITS THE LOCK FD (not close-on-exec)
W5  write child.pid
W6  wait for child exit, racing the box:
      at box_s:           SIGTERM to the child's process group
      at +kill_grace_s:   SIGKILL to the group
      still unreapable after +60s → exit.json{wedged}; exit
W7  exit.json{completed | killed_box, exit_code, …}; exit
```

**The lock rides the fd (W4).** `flock` belongs to the open file
description, so the lock's lifetime is *the child's*, not the wrapper's:
a murdered wrapper cannot leak the resource while its orphaned child
still computes, and an unkillable (wedged) child **keeps the lock** — the
poisoned resource quarantines itself at the kernel, with no policy code.
Conservative failure mode: a straggling grandchild that inherited the fd
holds the lock too long. Exclusion errs toward exclusion.

## 7. Format 4 — `exit.json`, the sentinel

```jsonc
{
  "schema": 1,
  "run_id": "…",
  "outcome": "completed" | "killed_box" | "canceled" | "contended"
           | "wedged" | "crashed" | "stillborn",
  "exit_code": 0,             // null unless the child was reaped
  "signal": null,
  "started_at": "…", "ended_at": "…", "box_used_s": 812.4,
  "written_by": "wrapper" | "harvest" | "cancel"
}
```

**Exactly-one-writer mechanics:** write `exit.json.tmp.<pid>` in full,
then `link(tmp, "exit.json")` — `EEXIST` means you lost; delete your tmp.
`rename` clobbers; `link` is the first-wins primitive. Immutable forever
after.

## 8. Format 5 — the status decision procedure

A **total function** of the directory; no in-process state anywhere.
`alive(pidfile)` ⇔ pid exists **and** its start-time matches the recorded
one (defeats pid reuse). `age` = now − mtime(`claimed` if present, else
`spec.json`) — the clock restarts at claim.

```
exit.json valid                                → DONE(outcome)
else wrapper.pid ∧ wrapper alive               → RUNNING
else wrapper.pid ∧ child.pid ∧ child alive     → ORPHANED
else wrapper.pid (all dead)                    → harvest links
                                                 exit.json{crashed}   → DONE
else spool entry present                       → QUEUED   (no timeout —
                                                 queued is a happy state)
else age > 120s                                → harvest links
                                                 exit.json{stillborn} → DONE
else                                           → LAUNCHING
```

**ORPHANED** is deliberate: wrapper dead, child computing, box no longer
enforced. The runner *reports* and refuses to decide — killing versus
waiting is caller policy (on some hosts, killing mid-GPU-compile wedges
the machine; the runner cannot know that; the caller does). An orphaned
run reaches DONE only after its child dies, naturally or by caller order.

## 9. The library API — the whole surface

```
submit(root, spec)  → Ticket | AlreadyExists    (idempotent: mkdir is
                                                 the commit point)
status(root, id)    → QUEUED|LAUNCHING|RUNNING|ORPHANED|DONE(outcome)|ABSENT
harvest(root)       → settle every settleable dir (crashed/stillborn),
                      then return ALL DONE runs (id, outcome, spec —
                      continuation returned verbatim, as a courtesy).
                      Caller dedups against its own ledger; the watermark
                      belongs to the viewer.
dispatch(root)      → one drain pass per §5; safe to kick anytime
cancel(root, id)    → Canceled | TooLate        (queue-only)
probe(root, tag)    → {free|held, queue_depth}  (board food)
```

No priorities. No retries. No GC. No push — completion notification is
polling (`harvest`); at most a future mailbox-file drop; **never**
execution of caller code. Anything resembling policy that appears here
means the page has failed.

## 10. Failure enumeration (kill anything at any line)

| killed at | observable state | settled by | outcome |
|---|---|---|---|
| submit, before mkdir | nothing | — | ABSENT; caller retries, idempotent |
| submit, after mkdir before spec link | empty dir | harvest, after grace | stillborn |
| submit, after spec before enqueue/spawn | spec only | harvest, after grace | stillborn |
| dispatch, after claim before spawn | claimed, no wrapper | harvest, after grace | stillborn |
| dispatch, mid-pass | dispatch.lock releases with it | next kick | unchanged; QUEUED persists |
| wrapper, W2–W4 | wrapper.pid (or not), dead | harvest | crashed / stillborn |
| wrapper, W5–W6 (child survives) | child alive, no babysitter | caller policy | ORPHANED → crashed/… |
| child (any cause) | wrapper reaps | wrapper | completed / killed_box |
| wrapper after child exit, before W7 | both dead, no sentinel | harvest | **crashed — see B1** |
| cancel racing claim | one rename wins | — | canceled XOR proceeds |
| harvest / any caller loop | nothing — it holds no state | next kick | unchanged |

## 11. Claims and blemishes

- **C1 — single terminal.** At most one `exit.json` can ever exist
  (link-if-absent); every run reaches one, given harvest/dispatch
  cadence. Safety is structural; progress needs cadence — cadence, never
  load-bearing for correctness.
- **C2 — mutual exclusion per tag,** kernel-enforced, tied to the actual
  resource user via fd inheritance; survives every row of §10, erring
  toward exclusion.
- **C3 — idempotent submit** from caller-deterministic run_ids;
  at-least-once caller logic yields exactly-once execution attempts.
- **C4 — statelessness.** All state is the store; every participant is
  killable at any instruction with §8 remaining total.
- **C5 — FIFO per tag.** For a given tag, launches occur in enqueue-
  sequence order (canceled entries excepted). Order is explicit
  (sequence numbers), never mtime vibes.
- **B1 — `crashed` may mask `completed`.** Wrapper death in the gap
  after child exit, before W7. Narrow, real, unfixable without a
  transactional kernel — it is the classic RPC exactly-once
  impossibility. Callers MUST read `crashed` as "unknown — possibly
  completed"; `ctx/` remains for forensics; a child-written done-marker
  in `ctx/` is the caller-side disambiguator.
- **B2 — ORPHANED runs are unboxed** until settled; settling is caller
  policy (§8). The box is a wrapper service, not a kernel one.
- **Non-claims:** no fairness across tags, no sandbox, no retention, no
  multi-host, no content validation, no secret handling, no push.

## 12. The first caller — notes across the seam

Everything below the line is domain; the seam is exactly here.

- **Enqueue discipline is the lab's scheduler.** The ⅓ exploration
  share, nice-then-age, admission — all implemented as *what the lab
  chooses to enqueue, and when*, keeping spool depth ≈ 1 per tag for
  full control. The spool never learns any of it.
- **One noun all the way down:** smoke, train, eval are runs; **apply is
  a run holding tag `canon`; curate holds tag `wiki`** — every
  serialization point in the lab is this same flock primitive.
- **Sibling clients compose:** a champion-cranker is just another
  mundane cron client that enqueues one slice when `probe(gpu)` shows an
  empty queue — the lab never knows it exists.
- **Pin before submit:** SHAs referenced in argv must stay reachable
  (e.g. `refs/autolab/runs/<run_id>`), or env materialization fails later.
- **Cold-SHA env builds count against the box** (and the tag lock, if
  held). Pre-warm pattern: a `resource: null` run of
  `uvx --from …@<sha> python -c pass` before the GPU stage.
- **Anomaly policy:** `wedged` → quarantine the tag, dead-letter the
  human. `contended` → investigate; it should never happen under a
  single dispatcher.

## 13. Cross-refs

[autolab design v5](autolab-design-v5.md) — the first client ·
[v4](autolab-design-v4.md) (pre-substrate; superseded) ·
[red-team A1–A26](autolab-design-adversarial-review.md) ·
[doctrine](autolab-doctrine.md).
