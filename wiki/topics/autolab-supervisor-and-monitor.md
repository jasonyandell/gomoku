# Autolab supervisor + monitor + research-lite (P5–P7): the unattended-overnight operating contract

> **Status: DORMANT (operating appendix).** Built + integrated 2026-06-19 (#64) and
> proven live that night, but the **autonomous derby is stopped** — see
> [derby.md](../derby.md) for live status. This page is the *operating appendix* to the
> canonical [autolab-architecture.md](autolab-architecture.md); it describes how to bring
> a **currently-not-running** lab up/down. Read it as a build+runbook record, not a live
> procedure — the forward-looking "tonight / the overnight run" phrasing below is
> historical (2026-06-19). The literal launchd plist XML was moved to a separate
> page and has since been removed *(2026-07-04; recover: `git show ca76350:wiki/_archive/topics/autolab-launchd-plists.md`)*.
> (Marked 2026-07-04.)

**What this is.** The contract that turns the tested autolab *library*
(`gomoku/lab/{ledger,daemon,trainer,arena,status}.py` + `gomoku/hf.py` +
`scripts/run_sweep.py`/`sliding_gate.py`) into a *running* self-driving lab that
survives a closed-lid overnight on the M5 Max: launchd keeps the daemons alive, a
seed row starts a real 9×9 training lane, a periodic monitor writes a wake-up
digest + macOS notification, and a deterministic research-lite tick keeps a
"current thinking" note and queues ideas **strictly behind** the seed lane.

It is **glue, not new abstraction** (Karpathy-lean, a few hundred lines total).
The load-bearing primitives already exist and are unit-tested; this page says how
to wire them, with absolute paths and literal config. The build step de-stales
[autolab-architecture.md](autolab-architecture.md) (the design page); **this** page
is the *operating* contract.

> **Honesty note (the #61 caveat).** The research-lite lane ranks ideas on
> **proxies** — a per-slice Δelo/hr secant from the trainer's own ≥20-game
> final-eval (±100-elo noisy), **not** a real wall-clock-to-Δelo gate. The
> anchored MTTE/EPWH gate is **unbuilt** (GitHub #61;
> [wall-clock-to-elo-metric.md](wall-clock-to-elo-metric.md);
> `scripts/delta_e_harness.py`). Every ranking the researcher writes is a hint,
> not a verdict. The note carries this banner literally.

---

## (a) `~/data/autolab/` home layout

`daemon.home()` = `$AUTOLAB_HOME` or `~/data/autolab` (`daemon.py:50-53`). One
out-of-git tree; everything below is created by `autolab up` (`os.makedirs(...,
exist_ok=True)`) before the daemons load.

```
~/data/autolab/
  ledger.jsonl                  # the spine (ledger.append flocks; daemon.py/trainer/arena/research append)
  stop                          # the cooperative stop-file (present ⇒ daemons break their loop); absent in normal run
  daemon-train.lock             # flock singleton + JSON meta {pid,role,item,status,started_at,host} (daemon.py)
  daemon-arena.lock
  logs/                         # launchd stdout/stderr for the long-running daemons
    train.out.log  train.err.log
    arena.out.log  arena.err.log
  runs/<lane>/                  # trainer DATA home (GOMOKU_RUN_DIR); teardown can't touch it
    sweep_runs/<cell>/checkpoints/{latest.pt,epoch*.pt,eval_results.jsonl}
    sweep_logs/<cell>/trainer.log          # per-epoch gen=/train=/plies=/buf= (the plies-collapse read)
  worktrees/<row>/              # ephemeral per-slice CODE-only git worktree (added/removed each slice)
  arena/                        # arena gate outputs (sliding_gate.run_gate dry-run)
    board.json  verdicts.jsonl  peak.pt
  monitor/                      # MONITOR owns writes here
    latest.md                   # overwritten each tick — the wake-up artifact
    log.md                      # one compact line appended per tick
    launchd.out.log launchd.err.log
  research/                     # RESEARCH-LITE owns writes here; MONITOR reads read-only
    latest.md                   # overwritten each tick — "current thinking"
    NOTES.md                    # append-only chronological trail
    launchd.out.log launchd.err.log
```

**Path-ownership contract (resolves the monitor↔research collision):**
- `research/latest.md`, `research/NOTES.md` — **written by research-lite only**;
  the monitor reads them read-only.
- `monitor/latest.md`, `monitor/log.md` — **written by the monitor only.**
- `arena/board.json`, `arena/verdicts.jsonl` — written by the arena; monitor +
  research read read-only.
- `runs/<lane>/.../trainer.log` + `eval_results.jsonl` — written by `run_sweep`;
  monitor + research read read-only.
- `ledger.jsonl` — appended (under flock) by trainer, arena, **and** research;
  read by everyone. Concurrent appends are safe (`ledger.append` flocks,
  `ledger.py:222-234`).

---

## (b) Process tree + the four launchd jobs

**Two daemons run directly under launchd** (no parent respawn loop — the flock
singleton, `daemon.py:82-98`, already makes double-launch impossible; launchd
respawn + ledger re-pick is the entire crash-recovery story). **Two periodic
one-shots** (monitor, research) run on `StartInterval`. The supervisor
(`gomoku.lab.up`) is a thin idempotent CLI that installs/loads/unloads these — it
is **not** long-lived.

```
launchd (gui/$(id -u), uid 501)
├── com.gomoku.autolab.train     KeepAlive(SuccessfulExit=false), RunAtLoad
│     └── uv run python -m gomoku.lab.trainer --prod --stop-file ~/data/autolab/stop
│           └── (per slice) git worktree ~/data/autolab/worktrees/<row>/
│                 └── python scripts/run_sweep.py --cell derby-v9-small
│                        --max-wall-secs 3600 --final-eval --foreground [--resume <latest.pt>]
│                        ├── gomoku.train            (the MPS tenant — torch loads HERE, not in the daemon)
│                        └── N× gomoku.selfplay_worker
├── com.gomoku.autolab.arena     KeepAlive(SuccessfulExit=false), RunAtLoad
│     └── uv run python -m gomoku.lab.arena --stop-file ~/data/autolab/stop
│           └── (per eval) scripts.sliding_gate.run_gate(dry_run=True) + HF champion tag
├── com.gomoku.autolab.monitor   StartInterval=600, RunAtLoad
│     └── uv run python scripts/autolab_monitor.py        (writes monitor/latest.md, notifies on change)
└── com.gomoku.autolab.research  StartInterval=1800, RunAtLoad
      └── uv run python -m gomoku.lab.research --once      (writes research/latest.md, appends ≤2 rows)
```

**Why this shape (decisions, not options):**
- **Direct launchd, no caffeinated parent.** The flock loser exits
  `EXIT_SINGLETON_HELD=75` (`daemon.py:212-214`); `set_inheritable(fd,False)`
  (`daemon.py:87`) means the lock dies with the daemon PID, not with a
  `run_sweep` grandchild — so respawn is clean with zero stale-lock cleanup. A
  parent loop would re-implement exactly this.
- **Two daemons = two failure domains.** An arena OOM never takes the trainer
  down and vice-versa.
- **`KeepAlive={SuccessfulExit:false}`**, not bare `true`. Respawns on
  crash/nonzero, **honors a clean `exit 0`**. The daemon returns 0 on stop-file
  /SIGTERM (`daemon.py:269`), so `down` is not fought by respawn. (Bare
  `KeepAlive=true` would hot-loop against the stop-file — do not use it.)
- **User LaunchAgent in `~/Library/LaunchAgents`, never a system LaunchDaemon.**
  MPS compute works for a *user* agent (gui/uid domain has the user's Mach
  bootstrap); a root `/Library/LaunchDaemons` would run pre-login as the wrong
  user with the wrong `HOME`/keychain and may lack graphics access. Torch never
  loads in the agent itself — only in the `run_sweep` subprocess — so the agent
  needs near-minimal env, but the plist env is **inherited all the way down** to
  `gomoku.train` (`_run_slice` does `env = dict(os.environ)`, `trainer.py:138`).

### launchd env — every var that must be set explicitly

launchd user-agents start near-empty. The plist `EnvironmentVariables` is the
single source of env for the whole tree.

| Var | Value | Why |
|---|---|---|
| `AUTOLAB_HOME` | `/Users/jason/data/autolab` | Drives ledger, lockfiles, runs/, worktrees/. Default already matches (`daemon.py:53`); set it so a future default change can't strand the run. |
| `HOME` | `/Users/jason` | **Critical.** `huggingface_hub` reads the token from `~/.cache/huggingface/token` (no `HF_TOKEN` env on this box). Wrong/unset `HOME` ⇒ `push_slice` 401s silently — the "ran but nothing reached HF" failure. Also git config + MPS caches. |
| `PATH` | `/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin` | Trainer shells out to `git worktree` (`/opt/homebrew/bin/git`, trainer.py:117) and `pgrep` (`/usr/bin`, trainer.py:55). launchd's default PATH lacks homebrew. (#87: no `.venv/bin` — `uv run` is the python entry point, no activation needed.) |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | `_run_slice` already `setdefault`s it (trainer.py:140); set at top for auditability. |
| `HF_HUB_DISABLE_PROGRESS_BARS` | `1` | Matches `hf.py` setdefault; keeps launchd logs clean. |
| `WANDB_MODE` | `offline` | **Decision below:** unattended slices record W&B locally (syncable) without ever touching the network or prompting `wandb login`. Keeps "W&B from day one" technically intact while removing the auth/network failure mode for the first-ever prod run. |

**Not needed:** `WANDB_API_KEY` (in keychain; with `WANDB_MODE=offline` it isn't
read), `VIRTUAL_ENV`/`PYTHONPATH` (#87: `uv run` resolves the per-worktree
`.venv` from cwd, then editable-install resolves `gomoku`), `HF_TOKEN` (the cache file is the auth path),
`GOMOKU_BOARD_SIZE` (the seed cell is 9×9 — see Risk #1). `TMPDIR`: leave unset
(launchd gives a private per-UID temp; `push_slice` uses `TemporaryDirectory`).

`WorkingDirectory` = `/Users/jason/code/gomoku` (the main checkout / editable
install). The trainer computes `repo_root = Path(__file__).resolve().parents[2]`
(trainer.py:41) independent of cwd, so `git worktree add` targets the right repo
regardless; setting cwd to main just avoids relative-path surprises.

### Literal plists — removed

The four literal LaunchAgents plist XML blocks (train/arena/monitor/research) and the
arena/monitor/research per-plist deltas were archival for this **DORMANT** lab and have
since been removed *(see note above)*.
Key shape: two KeepAlive={SuccessfulExit:false} daemons (train/arena) + two StartInterval
one-shots (monitor 600s / research 1800s); env (AUTOLAB_HOME/HOME/PATH/MPS+W&B vars) per the
table above; WorkingDirectory = /Users/jason/code/gomoku.

---

## (c) The seed config + base/cell/cap decision

Append **exactly one** `experiment` row to `~/data/autolab/ledger.jsonl` (via
`ledger.experiment(...)` → `ledger.append`); the trainer's flywheel
(`trainer.py:172-189`) self-perpetuates the lane (each finished slice appends a
continuation at the **same** priority + an arena eval at `priority+1`).

```json
{
  "type": "experiment",
  "id": "9x9-champ-recipe@0",
  "role": "train",
  "commit": null,
  "base": "scratch",
  "config": {
    "lane": "9x9-champ-recipe",
    "cell": "derby-v9-small",
    "max_wall_secs": 3600,
    "seq_n": 0
  },
  "priority": 10,
  "note": "overnight seed: fresh 9x9 v8-champion recipe (derby-v9-small), scratch, 1h slices; flywheel chains continuations",
  "status": "open"
}
```

**Key-placement gotchas (verified against `trainer.py`):**
- `max_wall_secs` MUST live under `config` — `run_chunk` reads
  `config.get("max_wall_secs", ...)` (trainer.py:75). Top-level is silently
  ignored.
- `commit: null` → `_checkout` uses `HEAD` of the repo_root (trainer.py:113-122).
  Fine for an overnight on a quiet `main`; pin a real SHA for byte-reproducibility
  across all slices (see Risk #6).
- `seq_n: 0` so the first continuation is named `…@1` cleanly (trainer.py:173-174).
- **`priority: 10` is the seed band convention.** Seed lanes live at p10;
  research lanes land **strictly below** (p9/p8); the flywheel eval row is
  p11 (eval-the-slice wins over starting the next slice — keeps the arena fed).
  This is the legible band research-lite keys off (§e).

**CELL = `derby-v9-small`** (`scripts/run_sweep.py` ~L1415-1431): byte-identical
to the proven **v8 champion** `derby-v7-mate-discount` recipe but with its own
fresh `sweep_runs/` dir (cold start). Recipe = size `small` (64×4),
sims=100, fixed-step async (`wave_mode=False`), 8 workers, `global_pool` +
`--gumbel-root --gumbel-m 16 --vcf-teacher --value-discount 0.98` +
`--sgd-steps-per-epoch 64`, 1.5M buffer, EMA τ=0.99. **ML rationale:**
- Not `SMOKE` (no real signal). Not a `G15-*` cell — those are 15×15 and the
  trainer never sets `GOMOKU_BOARD_SIZE` (trainer.py:130-143), so they'd
  silently run at 9×9 and shape-mismatch their 15×15 warm-starts (**Risk #1**).
- Not `derby-v7-mate-discount` directly — that name *resumes* the 16k-epoch
  matured champion (saturated, near-zero Δelo). `derby-v9-small` is the
  purpose-built fresh-start clone for a "watch it learn from zero" night.
- The v8 champion recipe is the project's **known-good 9×9 winner** (lost 3/120
  vs Rapfi-2625 at 9×9; `TRAINING_WIKI.md`), so tonight is a *self-validating*
  "the lab trains and climbs" demonstration. (Real new strength is a 15×15
  problem, gated on the board-size fix — out of scope tonight.)

**BASE = `scratch`.** No 9×9 warm-start exists on HF (`jasonyandell/gomoku-9x9`
has only `main`, zero tags; `az-recipe-160k` is a recipe *label*, not an
artifact). Scratch gives clean, monotonic per-slice Δelo with no path to
mis-resolve. `_resolve_base` maps `"scratch"` → `resume=None` (trainer.py:98-101);
**continuation slices then warm-resume the lane's own growing `latest.pt`**
(`cont_base = local://…/latest.pt`, trainer.py:176-177) — cold start in slice 1,
true buffer-carrying resume in slices 2..N (the flywheel).

**CAP = `max_wall_secs = 3600`** (the hard-max). The trainer self-caps on an
epoch boundary, force-saves a resumable `latest.pt`, then `--final-eval` writes
the `eval/model_elo` line the trainer reads (trainer.py:148-158). At ~2.5 s/epoch
steady-state, a 3600 s cap yields **~1,200–1,300 epochs/slice** — well past the
cold inflection, deep in the fast-climbing regime. Bigger cap = more
epochs/slice = cleaner per-slice Δelo (short evals are noisy); the daemon chains
slices automatically so you don't lose slice count.

**Expected slices by ~8h morning: ~6–7.** Per-slice wall ≈ 3600 (cap) + ~110
(final-eval) + ~90 (teardown) + ~30 (poll) + checkout ≈ **~64 min**. 8 h / 64 min
≈ 7; budget 6 for thermal/MPS variance. Satisfies "trained several real slices."

**W&B = OFF for the daemon** via `WANDB_MODE=offline` in the plist. Reason: every
real cell passes `--wandb`, and `wandb.init()` can block on interactive login if
the key isn't in env/netrc — fatal for an unattended run. `offline` records
local, syncable run dirs without network/prompt. Progress is read from the
ledger + logs (the monitor, §d), not W&B.

---

## (d) The monitor digest — `scripts/autolab_monitor.py`

A pure-read, no-torch, no-GPU, no-W&B, no-Claude script. `main(argv)` is called
both by launchd (`StartInterval=600`) and in-session
(`python scripts/autolab_monitor.py --print --no-notify`). Reuse
`status.lane_board(ledger_path)` (status.py:29) — one call gives the folded
ledger **plus** both daemon liveness probes. Add three file-tail reads
(research note, `trainer.log` plies, arena `verdicts.jsonl`/`board.json`),
compute the lane-local Δelo, render the template, **atomic-write** `latest.md`
(tmp + `os.replace`), append `log.md`, notify-on-change.

**The four durable reads:**
1. **Researcher thinking** — `research/latest.md` first non-empty line as the
   one-glance hook (+ first ~8 lines in the body); `research/NOTES.md` last ~2
   lines; ledger `event` rows (last ~5, `state.events`) — **including
   `preflight deferred: …` events** (daemon.py:241), the foreign-tenant standoff
   Jason needs to wake up to.
2. **Lanes/tickets** — from `status.lane_board`: counts by status (open/claimed/
   done/failed), last ~6 result rows (`state.experiments[id].result.metrics`),
   `next[train]`/`next[arena]` via `state.pick`.
3. **Training** — last train result `metrics` (`eval/model_elo`, `epochs_ran`,
   `lane`, `cell`) + `wall_s`; **Δelo computed in the monitor** by diffing
   successive `done` train rows of the lane (there is **no `delta_elo` field** on
   train results — confirmed trainer.py:88); plies trend from
   `runs/<lane>/sweep_logs/<cell>/trainer.log` (regex `plies=([\d.]+)`, last ~5
   → falling = fast-attack-collapse watch). Use `metrics["cell"]` to pick the
   right `sweep_logs/<cell>` dir (don't glob — avoids a stale lane's log).
4. **Evals** — last few `state.evals` with `verdict.gate`
   (PROMOTE/REVERT/AMBIGUOUS) + `win_rate`/`ci`/`n`/`vs`; champion proxy from
   `arena/board.json` `peak_path`/`last_verdict` (local read, no network) **but**
   the authoritative "did we promote" signal is the ledger
   `eval.metrics["promoted"]` (arena.py:82), NOT `board.json.applied` (always
   `False` under dry-run); `arena/verdicts.jsonl` last line for the rich `reason`.

**`monitor/latest.md` template:**
```markdown
# Autolab digest — {ts_local}
**slice {lane}@{seq_n} · elo {elo} ({delta:+}) · train {ALIVE|DOWN} · arena {ALIVE|DOWN} · eval: {GATE} · champ {champ_short}**

## Researcher
> {first line of research/latest.md}
recent: {NOTES.md last 2 lines}
events: {last 2 ledger event summaries — incl. any "preflight deferred"}

## Lanes (tickets)
done {n_done} · failed {n_failed} · open {n_open} · claimed {n_claimed}
next[train]: {id|idle}   next[arena]: {id|idle}
recent results: (last 5)  #{seq} [{status}] {id} {role} elo={elo} ({wall_s}s)

## Training
trainer: {ALIVE on {item} since {started_at} | DOWN (last: {item})}
last slice: {lane}/{cell} epochs={epochs} elo={elo} Δ={delta:+} wall={wall_s}s
plies: {v1}→…→{vN} {▲|▬|▼ collapse-watch}

## Evals
arena: {ALIVE|DOWN}
last verdict: {GATE} win_rate={wr:.2f} ci=[{lo:.2f},{hi:.2f}] n={n} vs {vs}
  reason: {verdict.reason}
champion: {candidate} (HF tag: champion)   history: {GATE GATE GATE GATE}
```

**`monitor/log.md` line (one per tick):**
```
{ts}  slice {lane}@{n} elo {elo} ({delta:+})  train:{A/D} arena:{A/D}  eval:{GATE}  open:{n_open} fail:{n_failed} | {research_hook[:50]}
```

**Notification (`osascript -e 'display notification "…" with title "Autolab"'`,
≤~120 chars):**
```
Autolab · slice {lane}@{n} · elo {elo} ({delta:+}) · eval:{GATE} · {researcher_hook[:40]}
```
**Escape `"` and `\` and strip newlines** in dynamic parts (reason/research
strings contain quotes — unescaped ⇒ the notification silently fails):
`text.replace("\\","\\\\").replace('"','\\"')` + collapse whitespace.

**Notify gating (don't ping every 10 min for nothing):** fire **only on a state
change** vs the previous `log.md` line — compare `(slice_id, gate, train_alive,
arena_alive, failed_count)`; also fire on a new PROMOTE/REVERT, a daemon→DOWN
flip, or a fresh preflight-defer. `--always-notify` for testing.

**Empty-state rendering (critical — many ticks fire before the first slice
finishes):** every read degrades to a legible "warming up" line, never a
stacktrace. No ledger → `(ledger empty — seeding)`. No train result → `last
slice: (none — first slice in flight)`. No `eval/model_elo` (it can be `None`,
trainer.py:150-158) → `elo —` and **no `None` arithmetic** for Δelo. One data
point → `(—)` not `(+0)`. Daemons DOWN on the first tick (launchd may start the
monitor before the daemons) → render `DOWN` but **do not notify** on the baseline
tick (no prior `log.md` to diff).

---

## (e) Research-lite tick — `gomoku/lab/research.py`

**Deterministic Python, NOT an LLM agent (night 1).** An LLM tick adds
network/auth/rate-limit/nondeterminism to a loop whose only job is "read ledger,
append 0–2 low-priority rows, write a note" — and the honest-proxy constraint
makes the LLM's main advantage (richer ranking) moot, even a liability
(confident prose over noisy proxies). The LLM is an easy later swap: same tick,
ideation step swapped from rules to a model call, reading the same
`research/latest.md` as context. Note in the contract; don't build it tonight.

`research.py` is **not a `Role`** (no `run_chunk`, no flock) — a one-shot like
`status.py main()`, `python -m gomoku.lab.research --once`, stdlib + `ledger`
only (no torch/HF/network). Run via launchd `StartInterval=1800`.

**Tick algorithm:**
```
1. fold = ledger.fold(ledger.read_all(ledger_path))
2. summary  = summarize(fold)                          # pure digest dataclass
3. ideas    = gather_ideas()                           # gh derby-idea issues, try/except → []
4. proposals = propose_experiments(fold, ideas, summary)   # 0..2 experiment() rows, capped
5. for row in proposals: ledger.append(ledger_path, row)
6. ledger.append(ledger_path, ledger.event(scope="research",
       summary=<one-line digest>,
       data={"proposed":[r["id"] for r in proposals], "ranks_on":"proxies", "gate_open_issue":61}))
7. atomic-write research/latest.md (overwrite)
8. append "\n\n## <iso-ts>\n" + note to research/NOTES.md
9. print one line; exit 0
```

`summarize` pulls per-lane elo series (`result.metrics["eval/model_elo"]` ordered
by `seq`) → proxy **Δelo/hr** secant; last few `evals` verdicts; plies signal if
present; `probe_alive(lockfile_path("train"))` liveness; queue depth by status.

**Priority rules — the starvation guard (the load-bearing invariant):**
> Every research-proposed `experiment(role="train", …)` MUST have
> `priority < P_seed`, where `P_seed = max(priority of open|claimed train rows)`.
> Concretely `proposed_priority = P_seed - 1` (and `-2` for a 2nd proposal).

`priority` is the first element of `default_priority` (ledger.py:293-300), so a
strictly-lower priority **guarantees** the trainer never picks a research row
while any seed continuation is open — regardless of the never-completed/seq
tiebreaks (those only reorder *within* a band; **equal** priority is unsafe
because a fresh research row is also never-completed with a higher seq). With the
p10 seed convention, research always lands at p9/p8 — legible in `autolab status`.
**Cold start (no open/claimed train row): propose nothing as `train`** and emit a
note "no seed lane yet; not seeding research lanes — continuity over thrash" (the
human/seed enqueues lane 0; the researcher must not become *the* seed lane).

**Volume + de-dup:** ≤2 new train rows/tick, and only if open-research-rows < K
(=3); else propose 0 ("research queue saturated; deepening seed lane").
**Idempotent ids** keyed on idea+`P_seed` (e.g. `research-<slug>-p<P_seed>`) so a
re-fire re-states the same id (a no-op via fold's id-keying). `base` = the seed
lane's latest model (`local://runs/<lane>/sweep_runs/*/checkpoints/latest.pt`,
warm-start fork) else `"scratch"`. Only file a `train` row for ideas that map to
an **existing `--cell`**; code-heavy derby ideas stay note-only (carried by the
GitHub issue). `gather_ideas` failure (gh absent/network) degrades to `[]` +
"idea source unavailable" — never crashes the tick.

**`research/latest.md` template** (overwritten each tick; `NOTES.md` gets the
same body appended under `## <ts>`):
```markdown
# Autolab researcher — current thinking
_generated <iso-ts> by gomoku.lab.research (deterministic, --once)_

> ⚠️ HONESTY: ranks on PROXIES (per-slice Δelo/hr secant from the trainer's own
> ≥20-game eval, ±100-elo noisy), NOT a real wall-clock-to-Δelo gate. The
> anchored MTTE/EPWH gate is UNBUILT — GitHub #61, delta_e_harness.py. Every
> ranking below is a hint, not a verdict.

## What is happening
- Seed lane `<lane>`: trainer <ALIVE on exp-N|DOWN>, <n> slices, elo <e0>→<eN>
  (proxy Δelo/hr ≈ <r>, inside-noise? <y/n>).
- Champion: <last gate> (win_rate <w>, n <n>) — <moved|held>.
- Plies: <falling → collapse-watch | stable | not in metrics>.
- Queue: <x> open train, <y> open research, <z> arena evals pending.

## What I would try next (and why)
1. <idea> — why: <one line>. Maps to: <cell C | NOTE-ONLY: needs code, #NN>.

## What I queued this tick
- <none — seed-lane continuity preferred> | <id `research-…-p<P>` at priority
  <P_seed-1> (strictly below seed p<P_seed>), base <…>, cell <…>>.

## Caveat (always present)
Proxy-based ranking. Continuity over thrash: the seed lane keeps training;
research rows queue strictly below it. Real adjudication waits on #61.
```

---

## (f) Runbook

### Bring-up — `autolab up` (`python -m gomoku.lab.up up`, idempotent)
1. `makedirs` `~/data/autolab/{logs,runs,worktrees,arena,monitor,research}`.
2. `rm -f ~/data/autolab/stop` (a stale stop-file from a prior `down` would make
   both daemons exit on first poll — **Risk #7**).
3. **Seed the ledger if empty** — if `default_ledger_path()` is missing/zero
   rows, `ledger.append` the §c row (resolve `commit` with
   `git -C /Users/jason/code/gomoku rev-parse HEAD` if you want a pinned SHA).
4. Write all four plists with `plistlib.dump` to `~/Library/LaunchAgents/`.
5. Load each idempotently:
   `launchctl bootout gui/$(id -u)/<label> 2>/dev/null; launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/<label>.plist`
   (`bootout`-before-`bootstrap` guarantees a single owner; fall back to
   `launchctl load -w` only if `bootstrap` is unavailable). `RunAtLoad` starts them.
6. Print `status`.

Canonical invocation is **`python -m gomoku.lab.up`** (reinstall-free — the
editable install resolves the new module immediately). The `gomoku-lab` console
script is a convenience added in the same PR but needs a reinstall to appear on
PATH; don't depend on it for the overnight run.

### Attended PROD-slice proof — run BEFORE the unattended launch
The first prod slice via the trainer has **never run**. De-risk the train → eval
→ ledger → flywheel → (HF) path with one foreground slice, **HF off first**:
```bash
# 0. confirm the box is free (the preflight defers on any of these):
pgrep -fl 'selfplay_worker|gomoku\.train|run_sweep|eval_worker'
# 1. seed the ledger (step 3 above), then one dry-run prod slice in the foreground:
AUTOLAB_HOME=~/data/autolab WANDB_MODE=offline \
  /Users/jason/code/gomoku/uv run python -m gomoku.lab.trainer --prod --no-hf --once
# inspect: a DONE result row with eval/model_elo + a continuation + an eval row;
#   ~/data/autolab/runs/9x9-champ-recipe/sweep_runs/*/checkpoints/latest.pt exists.
# 2. prove HF delivery once (real push), still foreground:
AUTOLAB_HOME=~/data/autolab WANDB_MODE=offline \
  /Users/jason/code/gomoku/uv run python -m gomoku.lab.trainer --prod --once
# inspect: artifact_ref is hf://…  and the revision appears on jasonyandell/gomoku-9x9.
```
Only after both pass: `python -m gomoku.lab.up up`.

### Read the monitor (morning + in-session)
- Wake-up artifact: `~/data/autolab/monitor/latest.md` (the header line is the
  one-glance read) + the macOS notification.
- History: `~/data/autolab/monitor/log.md` (one line/tick).
- Researcher's thinking: `~/data/autolab/research/latest.md`.
- On demand: `python scripts/autolab_monitor.py --print --no-notify`, or
  `python -m gomoku.lab.status` for the raw lane board.

### Stop everything — `autolab down` (`python -m gomoku.lab.up down`)
Graceful (stop-file) primary, SIGTERM/`bootout` the hammer:
1. Write `~/data/autolab/stop`; poll `probe_alive(lockfile_path(role))` a few
   seconds for each daemon to drop its flock. An **idle** daemon exits within
   0.25 s; a daemon **mid-slice** won't see the stop-file until its current
   `run_sweep` returns (up to the 1h cap) — that's why `down` also bootouts.
2. `launchctl bootout gui/$(id -u)/com.gomoku.autolab.{train,arena,monitor,research}`
   — de-registers the jobs (so `KeepAlive` can't respawn) and SIGTERMs the
   daemons (same `stop["flag"]` path → clean `exit 0`, flock released in the
   `finally`, daemon.py:268).
3. `rm -f ~/data/autolab/stop` once both are DOWN so the next `up` starts clean.
4. **Hard stop** (`down --force`): after the grace window,
   `pkill -TERM -f 'scripts/run_sweep.py'` **scoped to the `~/data/autolab/`
   worktree path** (so a foreign tenant is never killed) before `bootout`.

### Don't compete with foreign tenants
The supervisor does **not** pgrep-gate — it relies entirely on the trainer's
per-loop `preflight` → `PreflightDeferred` path (trainer.py:51-64,
daemon.py:239-247), which is *continuous* (a foreign IDE tenant appearing after
`up` is caught next poll) and logs an `event` without failing the lane. A second
launch-time gate would be a useless snapshot **and** compound a false-positive:
the lab's own `selfplay_worker` argv lacks the home marker (run_sweep.py spawns
with `cwd=REPO_ROOT`), so a naive second gate could mis-class the lab's own
workers as foreign. The arena correctly only *shrinks* `n_games` under a live
trainer (arena.py:58-61), never hard-defers — keep that asymmetry.

---

## Risks (ranked; carried from the four P-reports)

1. **[HIGH] No `GOMOKU_BOARD_SIZE` passthrough → 9×9 only.** `_run_slice` sets
   only `GOMOKU_RUN_DIR` + the MPS fallback (trainer.py:130-143). Any `G15-*`
   cell runs at 9×9 and shape-mismatches its 15×15 warm-start. **Mitigated by
   picking a 9×9 cell.** File a follow-up: add `GOMOKU_BOARD_SIZE` from `config`
   before the autolab can run the 15×15 frontier.
2. **[HIGH] First-ever HF push.** `push_slice` is the last step of `run_chunk`; if
   it throws (bad/expired token, `HOME` wrong in plist), the whole slice is marked
   FAILED even though training+eval+`latest.pt` succeeded locally. **Mitigated**
   by `HOME` in the plist + the attended `--no-hf` then real-push proof above.
3. **[HIGH] Priority inversion starves the seed lane.** The single worst
   research-lite outcome. **Mitigated** by the strict `priority < P_seed`
   inequality computed live each tick + the cold-start refusal to seed. Unit-test
   it: synthesize a seed continuation at p10, assert all proposals < 10 and that
   `fold().pick("train")` still returns the seed.
4. **[MED] Foreign-tenant preflight over-defer.** A stray derby/IDE training
   process makes the trainer defer **every poll, silently** → zero slices by
   morning. **Mitigated** by the `pgrep -fl …` check in the attended proof before
   `up`.
5. **[MED] `eval/model_elo` can be `None`** (missing/empty `eval_results.jsonl`).
   Not fatal to a slice, but the monitor/research must render `elo —` and never do
   `None` arithmetic for Δelo. The single most likely naive-digest crash.
6. **[LOW] `commit: null` → HEAD races a live `main`.** A later continuation could
   check out a different SHA than slice 1 if another session pushes `main`
   mid-night. Pin a SHA in `commit` for strict reproducibility.
7. **[LOW] Stale stop-file wedges `up`.** Both daemons exit on first poll. `up`
   must `rm -f ~/data/autolab/stop` (bring-up step 2).
8. **[LOW] Singleton-held respawn loop (exit 75).** If a manual daemon already
   holds the lock when launchd loads, the launchd instance exits 75 and respawns
   every `ThrottleInterval`. Benign (ms each) but noisy; `up`'s bootout-before-
   bootstrap + a `probe_alive` warning avoids it.
9. **[LOW] `set_inheritable(fd,False)` is load-bearing** (daemon.py:87): if a
   `run_sweep` child inherited the flock fd, the role would look ALIVE after the
   daemon died and launchd respawn would deadlock on `acquire()`. Non-negotiable
   invariant; any future change to subprocess spawning must preserve it.

---

## Build handoff (ordered; each names target files)

1. **Seed** — `gomoku/lab/up.py` step appends the §c JSON row via
   `ledger.experiment`/`ledger.append` (or a tiny `scripts/autolab_seed.py`).
2. **Supervisor** — new `gomoku/lab/up.py` (`main(argv)`: up/down/status/restart;
   `plistlib`-render the four plists into `~/Library/LaunchAgents/`; drive
   `launchctl bootout`/`bootstrap`; manage `~/data/autolab/stop`; makedirs the
   home subtree). Stdlib only; imports `daemon`, `ledger`, `status`.
3. **Monitor** — new `scripts/autolab_monitor.py` (`build_digest`/`write_digest`/
   `notify`/`main`; reuse `status.lane_board`; atomic-write `monitor/latest.md`;
   append `monitor/log.md`; notify-on-change). No torch.
4. **Research-lite** — new `gomoku/lab/research.py` (`summarize`/`gather_ideas`/
   `propose_experiments`/`render_note`/`main` `--once`). Stdlib + `ledger`. No
   torch/HF/network.
5. **Entry points** — `pyproject.toml [project.scripts]`: add
   `gomoku-lab = "gomoku.lab.up:main"` (and optionally
   `gomoku-lab-research = "gomoku.lab.research:main"`). Convenience only; `-m`
   paths are canonical.
6. **Tests** — `tests/lab/test_research.py` (priority math: proposals < P_seed
   and `pick("train")` still returns the seed; idempotent ids; note render;
   cold-start refusal), `tests/lab/test_up.py` (plist render shape, idempotent
   bootout-before-bootstrap argv, stop-file lifecycle — `launchctl` mocked),
   `tests/test_autolab_monitor.py` (digest from a synthetic folded state; `None`
   elo + one-data-point Δelo render; osascript escaping; notify-on-change diff).
7. **De-stale** — `wiki/topics/autolab-architecture.md`: mark P5 (research) + P6
   (status/monitor) + P7 (supervisor) shipped; point its ops section at this page.

---

## Build log (#64)

P5–P7 built + integrated on branch `feat/autolab-p7-autolab` (2026-06-19).
Glue-only, stdlib-only (`gomoku.lab.*` imports no torch/third-party); the four
launchd jobs wire the already-tested spine into a self-driving overnight lab.

### Final file inventory

| File | Role |
|---|---|
| `gomoku/lab/up.py` | Supervisor CLI `main(argv)` — `up`/`down`/`status`/`restart`; renders+loads the four plists (`render_plists`/`write_plists`); seeds the §c row (`seed_ledger`/`seed_row`, commit pinned non-null, idempotent); manages `~/data/autolab/stop`. Top-level flags `--ledger --launchd-dir --commit --dry-run --force` precede the subcommand. |
| `scripts/autolab_monitor.py` | Monitor digest — `build_digest`/`build_header`/`build_log_line`/`build_notification`/`write_digest`/`notify`/`run_once`/`main`. Atomic-writes `monitor/latest.md`, appends `monitor/log.md`, `osascript` notify-on-change. Reuses `status.lane_board`; computes lane-local Δelo (no `delta_elo` field on results). |
| `gomoku/lab/research.py` | Research-lite one-shot `main(argv) --once` — folds ledger, proposes ≤2 train rows at priority **strictly below** the live seed lane max (`research-<slug>-p<P_seed>`, idempotent), cold-start refusal, writes `research/latest.md`+`NOTES.md` with the honest `#61`-proxy banner, appends one `event` row. |
| `tests/test_lab_up.py` | Plist render shape, idempotent bootout-before-bootstrap argv, stop-file lifecycle, seed idempotency (`launchctl` mocked, `AUTOLAB_HOME`→tmp). |
| `tests/test_autolab_monitor.py` | Digest from synthetic folded state; `None`-elo + one-data-point Δelo render; osascript escaping; notify-on-change diff. |
| `tests/test_lab_research.py` | Priority invariant (proposals < P_seed and `pick("train")` still returns the seed); idempotent ids; note render; cold-start refusal. |
| `pyproject.toml` | Added `[project.scripts]` `gomoku-lab = gomoku.lab.up:main` + `gomoku-lab-research = gomoku.lab.research:main` (convenience; `-m` invocations are canonical and reinstall-free). |

### Run the full suite (`uv run pytest`)

#87 root-fixed the editable-install gotcha — each worktree now gets its own
uv-managed `.venv` (`gomoku` editable → that worktree). From the worktree:

```bash
uv run pytest tests/test_lab_ledger.py tests/test_lab_daemon.py \
  tests/test_lab_trainer.py tests/test_lab_arena.py tests/test_hf.py \
  tests/test_run_sweep_runbase.py tests/test_lab_up.py \
  tests/test_autolab_monitor.py tests/test_lab_research.py -q
```

(No editable-install finder repointing needed; `uv run` resolves the venv from
cwd, so `gomoku` always points at the worktree you're standing in.)
### Attended PROD-slice proof

The attended de-risk sequence (foreground `--no-hf --once` slice, then a real-push `--once`,
then a hand-read digest) is documented once in **§(f) Runbook → Attended PROD-slice proof** above
— see there rather than duplicating the commands.
