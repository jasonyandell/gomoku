# The Autolab DR Tabletop — pull the power at each table, see who survives

> **Status: DORMANT** *(2026-07-04)* — tabletop conducted 2026-06-24; the
> torn-line tail-guard (read path) + `triad_resume_under_crash` scenario shipped,
> #83/#84/#85 filed for the rest; autolab stopped. **2026-07-04 re-review added
> two REDs** (rows 7–8: the append-side torn tail; the multi-row-transaction
> class) — class fix = the [primary design](autolab-primary-design.md)'s two
> ledger walls. Under the [Autolab hub](../autolab.md).

**What this is.** A disaster-recovery tabletop for the self-driving lab — the
failure-mode map for a loop that must run **weeks unattended on a laptop**. It
walks the triad one table at a time, yanks the cord at each, and asks the only
question that matters for sleep: *does the lab restart clean, or does it stay
down until a human edits a file?* Sibling of
[autolab-arena-eval-lane.md](autolab-arena-eval-lane.md) (the measurement leg
this stresses) and [autolab-doctrine.md](autolab-doctrine.md) (the cage this
audits). Verified against the code on `feat/autolab-sim`, 2026-06-24.

The one-sentence thesis, in the doctrine's voice:

> **The backbone — crash → flock auto-frees → re-pick → resume from `latest.pt`
> — is real and sim-certified; the failures that actually threaten a sleeping
> lab are the few power-pull moments where an *external effect lands before its
> commit*, or where one torn byte makes the reducer refuse to fold.**

The triad under test, stated once: **research proposes** an experiment (with an
`evidence_contract`) → **trainer trains** a slice → **arena evals** → the result
flows back as evidence for the next research wakeup to decide on. Power gets
pulled at every seam in that ring.

---

## The walk — power pulled at each table

At each table: what's durable, and what a hard power-cut (true SIGKILL of the
box, not a clean SIGTERM) does to it on restart.

| Table | The action | Durable artifact | Power-pull verdict |
|---|---|---|---|
| **T0** research proposes | append an `experiment` row + `evidence_contract` | the row (flock + fsync) | **clean** — the row is durable or never existed; the next wakeup re-folds and re-proposes, idempotent on a stable id |
| **T1** trainer trains | pick → train → save `latest.pt` → push slim weights to HF → append `result` + followups (continuation + arena eval) | local `latest.pt` + the `result` row | **mostly clean** — no result row → re-pick resumes from `latest.pt` — **but two kinks** (below): (a) non-atomic `save_checkpoint`, (b) wifi-loss at HF push, only one of which is handled |
| **T2** arena evals | pick → resolve candidate + champion → play games → **move champion tag** → append `eval` + `verdict` | the `verdict` row (the commit) | **kinked** — the side effect (tag move) lands **before** the commit; a crash in the gap re-evals candidate-vs-itself → garbled verdict |
| **T3** back to research | `decision_due` fires (contract satisfied by the DONE arena experiment) → decide → append `research-decision` + watermark (`covers_through_seq`) | the decision row + the watermark | **clean** — the watermark makes a re-fire idempotent; a refused intent escalates to `needs_jason` |
| **ANY** table | a power-pull *mid* ledger-append | the fsync'd prefix | **scariest** — a **torn final line**; `read_all`/`parse_line` does a raw `json.loads` with no guard → the fold **raises** → every loop crashes on restart and **stays crashed** |

### T1, the two kinks, in detail

- **(a) `save_checkpoint` is not atomic on this branch.** A power-pull mid-write
  tears `latest.pt`; the re-pick then tries to resume from a torn checkpoint →
  the lane is **stuck**. This is **already fixed on `main` (`f661fd4`)** — an
  atomic write-then-rename — so the full merge of `main` into the branch closes
  it. Not a design question; a merge.
- **(b) wifi-loss at the HF push is handled *well*.** It is caught, the slice
  result is **downgraded to a `local://` artifact ref + a deferred event**, the
  lane lands **DONE**, and the continuation resumes from the local checkpoint.
  The champion-resolve path degrades the same way (no HF tag reachable → first
  candidate seeds the ladder). This is the model the other kinks should copy.

### T2, the kink, in detail

Today the order inside the arena's `run_chunk` is **[compute] → [move tag] →
[append eval + verdict]**. The external effect (the champion tag move) happens
**before** the commit (the verdict append). A crash in that window leaves the
champion already moved but no verdict recorded; the re-pick then re-evals against
the **just-moved champion** — the candidate plays *itself* — producing a
duplicate or garbled verdict. Last-writer-wins keeps the *ledger* uncorrupted,
but the **promotion semantics are wrong** (a self-play "PROMOTE"). This violates
the lab's load-bearing rule: **every external effect must come AFTER the commit,
or be idempotent.**

### ANY table, the torn line, in detail

The append path is correct — flock-guarded and `fsync`'d. The **read** path is
not defensive: `read_all` / `parse_line` calls a raw `json.loads` on every line
with no guard, so a single truncated trailing line (the exact signature of a
power-pull mid-append) makes the fold **raise**. Because every loop folds the
ledger first thing on startup, **one torn byte bricks the entire lab** —
trainer, arena, and research all refuse to start — until a human opens the file
and deletes the partial line. For a box meant to run unattended for weeks, this
is the single scariest failure: it defeats "just restart it."

---

## The kinks, ranked for weeks-unattended-on-a-laptop

Ranked by the only metric that matters here: *can the lab restart itself, or
does it need a human?*

| # | Sev | Kink | Fix | Where |
|---|---|---|---|---|
| 1 | ~~RED~~ **✅ FIXED 2026-06-24** *(read path only — see #7)* | Ledger reader intolerant of a torn final line (`read_all`/`parse_line` raw `json.loads`) | `read_all` now tolerates **only** a truncated **trailing** line (the power-pull signature) and still **raises** on an *interior* malformed line — that's corruption, not a torn tail. Falsified RED-when-off + new `torn_ledger_line_tolerated` sim scenario | `ledger.py` reader |
| 2 | **RED** | `save_checkpoint` not atomic → power-pull mid-write tears `latest.pt` → stuck lane | **Already fixed on `main` (`f661fd4`)**; the full merge closes it | trainer / checkpoint |
| 3 | **YELLOW** ([#83](https://github.com/jasonyandell/gomoku/issues/83)) | Arena moves the champion tag **before** the commit (T2) | Move the tag-set to **after** the `verdict` row is durable, **or** make resolve/re-eval idempotent | daemon/arena seam (design) |
| 4 | ~~YELLOW~~ **✅ BUILT 2026-06-24** | No end-to-end **triad** scenario in the loop sim | New `triad_resume_under_crash` scenario drives propose → train → eval → decide as **one chain** with a power-pull injected at **both** the train and arena tables, each recovered by re-pick | `tests/test_lab_sim.py` |
| 5 | **YELLOW** | The loop closes on the **gate**, not the **panel** | `EV_ARENA` ("arena-verdict") is satisfied by **any** DONE arena experiment — today the H2H-vs-champion gate, not "eval vs everyone → relative Elo." No `eval_kind` exists, so a contract can't require a **panel** specifically. Thread the [arena-eval-lane](autolab-arena-eval-lane.md) `eval_kind` discriminator into **both** the `eval` row and the `evidence_contract` | arena + researcher-contract seam |
| 6 | **GREEN** | W&B run-id poisoning on SIGKILL — abandoned `running` runs accumulate (cost/bloat) | **Self-heals** on the next resume via the wandb "run in use" fallback; cosmetic for correctness | W&B resume path |
| 7 | **RED** *(found 2026-07-04)* | **`append()` after a torn tail concatenates** the next committed row onto the fragment (`a+`, writes blind at EOF): while the mangled line is the tail, every fold **silently drops a committed, fsync'd row** *and* the next append **reuses its `seq`**; once any further row lands it's interior → the fold raises → full-lab brick. The write-path sibling of #1 — the read-path fix alone reopens the class. NB the obvious fix (prepend `\n` to seal the fragment) is **wrong**: it makes the fragment an interior malformed line, which the sim itself asserts must raise | Under the flock, if the file doesn't end in `\n`, **truncate back to the last `\n`** before writing — the fragment is provably uncommitted (`append` only returns after full-line+`\n` write **and** fsync). Plus a sim scenario that drives the **write** path (the existing torn scenario hand-seals with `\n`, stepping around this hole) | `ledger.py` append |
| 8 | **RED** *(found 2026-07-04)* | **`research.resume` half-decides under a kill**: the watermark event is appended FIRST and ALWAYS, side-effect rows after — a SIGKILL between them records `covers_through_seq` but never applies e.g. the `keep` unblock → the continuation stays BLOCKED forever, the thread never re-fires, no FAILED row exists so `health.scan` never flags it. The docstring's "#1 property" (watermark-first) inverts into a silent permanent deadlock. Same class: the trainer's `result`→followups gap in `daemon.run_daemon` (a kill between them → DONE slice, no continuation, trainer idles invisibly) | **The facts-not-commands wall** ([primary design](autolab-primary-design.md) §1 Wall A): the decision/result row itself carries what it implies; the fold derives the unblock/followups — one fsync'd append IS the whole transaction. Sim invariant: *no multi-row transaction exists*, RED-when-off | `research.py` resume · `daemon.py` |

**Genuinely handled — note it, don't fix it.** Wifi loss is graceful *end to
end*: the HF push catches and falls back to `local://` + a retry, and the
champion resolve degrades to first-champion-seeds-the-ladder. The deferred-event
path (T1 kink b) is the template the design-y kinks above should adopt.

---

## What the loop sim certifies today vs. what it does NOT

The [loop simulator](autolab-doctrine.md) drives the *real* loop with the ML
replaced by `random()` and throws chaos at it. Here is the honest cover/gap line
for the DR question specifically.

| | Covers (certified, RED-when-off) | Does **not** cover (gap) |
|---|---|---|
| **Crash** | true SIGKILL mid-slice (re-pick resumes from `latest.pt`); the **torn trailing line, read path** (#1, fixed + sim-covered 2026-06-24); the **full triad ring** under a power-pull at each table (#4, new scenario 2026-06-24) | the **write path onto a torn tail** (#7 — the existing torn scenario hand-seals with `\n`); a **kill between the appends of one logical transaction** (#8 — `SimCrash` fires only inside `run_chunk`); an **interior** corrupt ledger line (deliberately still raises — corruption ≠ a torn tail); the **arena tag-move-mid-crash** (#3) |
| **Network** | HF blip → deferred fallback (T1 kink b) | **real** HF latency / retry behavior |
| **Resource** | foreign GPU tenant (preflight-deferred retry); the 1h cap; silent-stall detection | **disk-full (ENOSPC)** on append/save |
| **Loop shape** | first-promotion gating; fold determinism; the era-cross (namespaced champion, no cross-era run) | the **full triad end-to-end in one run** (#4) |
| **Research governance** | evidence-contract WHEN; continuation-blocked; the intent-validation wall; decided-once (watermark) | **panel / `eval_kind` / panel-baseline cache** (#5); real MCTS/Elo in the gate; **W&B poisoning** (#6) |

The pattern in the gap column: the sim certifies each **piece** of the cage, and
certifies the **backbone** under a true crash — but it has never driven the
**specific triad ring** as one chain with a crash injected at each seam, and it
does not yet model the two storage-layer power-pull failures (torn line,
disk-full) that are exactly what a multi-week laptop run will eventually hit.

---

## Status — DESIGN / findings (2026-06-24, against `feat/autolab-sim`)

Tabletop only; the walk was traced against the code, not run as chaos. Verdict:

> The **backbone** (crash → flock-frees → re-pick → resume-from-`latest.pt`) is
> real and sim-certified. The specific **triad loop** closes *structurally* but
> on **gate-not-panel** evidence, is **never driven end-to-end**, and the
> **panel + `eval_kind` are unbuilt**. And **two power-pull kinks**
> (torn-line tolerance #1, atomic save #2) stand between here and "believe the
> number while asleep."

**Closed this session (2026-06-24):** #1 (the torn-line tail-guard in `read_all`,
falsified RED-when-off) and #4 (the `triad_resume_under_crash` scenario — the
first end-to-end propose→train→eval→decide chain with a power-pull at each table).
#2 is closed by merging `main` (atomic `save_checkpoint`, `f661fd4`). The
remaining three are design-y — they don't block a restart, they shape what the
loop *means* once it restarts — and are tracked as
[#83](https://github.com/jasonyandell/gomoku/issues/83) (arena tag-ordering),
[#84](https://github.com/jasonyandell/gomoku/issues/84) (panel-as-distinct-evidence
via `eval_kind`), and [#85](https://github.com/jasonyandell/gomoku/issues/85)
(W&B poisoning).

### 2026-07-04 addendum — the independent code review (fresh-eyes design session)

A full independent review re-walked the tables and found **two new REDs** (rows
7–8 above — the append-side torn tail; the multi-row-transaction class) plus
these residual **YELLOWs**, banked here so the failure-mode map stays the one
canonical list. Class fix for rows 3, 7, 8 = the two ledger walls in the
[primary design](autolab-primary-design.md) §1; sim-coverage gaps are the
chaos-coverage item of its phase-1 list.

- **Y2 — park-while-running race**: an in-flight slice is still OPEN (no claim
  rows, by design); a concurrent park appends SUPERSEDED corrections, but the
  in-flight `result` folds after them, sets DONE, and the followups enqueue a
  fresh continuation → **a parked lane resurrects** (≥1 wasted slice; with
  `default_decide` a rising fork can be re-kept, fully undoing the park). Wall A
  kills the resurrect (derived continuations honor the park at read time);
  the ≤1-quantum in-flight waste is accepted (review A10).
- **Y5** — `trainer._find_ckpt_dir` uses unordered `glob` → order-nondeterministic
  checkpoint resume if a lane's cell ever changed (works today: one lane ≡ one cell).
- **Y6** — `fuzz()` seeds only a train lane, so all five research invariants are
  **vacuously green under fuzz**; they're exercised only in hand-built scenarios.
- **Y7** — `scenario_fold_determinism` folds the same in-memory list twice
  (trivially passes); no order-sensitivity, correction/result interleaving, or
  disk-replay-vs-incremental check.
- **Y8** — the intent wall is in-process convention for a real agentic model
  (fix = the invocation shape, [primary design](autolab-primary-design.md) §3);
  `validate_intent` also accepts an intent citing **zero** evidence.
- Notes: `seq` counts comment/blank lines (harmless as ID, misleading as
  documented); `append` fsyncs data but not the directory (first-append
  durability); `rapfi_pool.pick` can surface a raw `queue.Empty` when the pool
  shrank to zero.

---

## Cross-refs
- [autolab-doctrine.md](autolab-doctrine.md) — the *why*; the sim certifies the
  cage (this page audits where the cage's certification stops).
- [autolab-architecture.md](autolab-architecture.md) — the ledger spine + the
  four lanes; the `run_chunk` shape this walk steps through table by table.
- [autolab-arena-eval-lane.md](autolab-arena-eval-lane.md) — the panel +
  `eval_kind` discriminator that kink #5 must thread into the evidence contract;
  the gate/panel split.
- [autolab-researcher-contract.md](autolab-researcher-contract.md) — the
  evidence contract / `arena-verdict` consumer that closes the triad; the
  watermark that makes T3 idempotent.
- [autolab-supervisor-and-monitor.md](autolab-supervisor-and-monitor.md) — the
  unattended operating contract this tabletop is the failure-mode map for.
- [cockpit-vs-autopilot.md](cockpit-vs-autopilot.md) — more autopilot needs more
  cockpit; a DR map is part of the cockpit's escalation instrument.
- Issues: [#83](https://github.com/jasonyandell/gomoku/issues/83) (arena
  tag-ordering), [#84](https://github.com/jasonyandell/gomoku/issues/84)
  (panel-as-distinct-evidence via `eval_kind`),
  [#85](https://github.com/jasonyandell/gomoku/issues/85) (W&B poisoning). Refs
  the autolab epic #53.
