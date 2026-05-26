# GPU broker architecture — the nonstop, open-entry, age-out derby

Design adopted with Jason 2026-05-26. Evolves the Δelo Derby from a *fixed-field
race that terminates* into a **nonstop, open-entry, performance-prioritized derby
that ages contestants out**. Jason's one-sentence model:

> All the contention goes to **one thing that has a queue**. Claude is free to do
> Claude things; the GPU stacks up without Claude managing box state. The thing
> decides nothing except *queue or not queue*.

That one thing is the **GPU daemon**. The brains live in a **broker** on top of it.

## The three roles (policy / mechanism / production)

```
 RESEARCHERS (agents · off-GPU · parallel · free)
   produce candidate cells (run_sweep.CELLS) and register them in the POOL.
        │  derby_pool.register(name, cell, lever)
        ▼
 CANDIDATE POOL  sweep_runs/derby_pool/candidates/*.json   (open entry)
        │  broker pulls when a lane frees
        ▼
 BROKER (policy · off-GPU · a tick loop — "Claude things", decisions only)
   - keeps ~N lanes in the running pool
   - submits 300s train chunks to the DAEMON (never runs the GPU itself)
   - CHUNK PRIORITY within the pool = anchored-elo climb rate  (cheap, in-race)
   - ~hourly VERDICT = pairwise H2H round-robin over lane peaks (off-GPU, CI<Δ gate)
   - CONDITIONAL PEAK-STAY AGE-OUT: retire a lane at its TTL unless it set a
     new peak in its last window  (== "never cut a climber", as a rule)
   - on freed slot → pull next candidate; on trustworthy verdict → crown/swap;
     on no-verdict → HOLD incumbent + emit the single "needs you" line
        │  daemon.submit(spec) → job_id ;  daemon.poll(job_id) → {state,result}
        ▼
 DAEMON (mechanism · the ONE GPU queue · owns box state & contention)
   - serial MPS, maildir queue (pending/running/done/failed)
   - decides nothing except run-next-by (tier, priority)
   - disk preflight · foreign-tenant politeness · crash-recovery · safety timeout
        │
        ▼
 GLANCE SURFACE  (cockpit · one verb)
   one view over: daemon queue · broker pool + age-out clocks · candidate pool ·
   last verdict · the "needs you" line.
```

This maps the [cockpit-vs-autopilot](cockpit-vs-autopilot.md) frame: daemon+broker
are the autopilot; the glance surface + CI<Δ gate + "needs you" line are the
cockpit. Sid's order (verifier → multi-agent → loops) is why the **gate is built
before the broker swaps on it**.

## Two signals, two jobs (the load-bearing distinction)

| | CLIMB signal | VERDICT signal |
|---|---|---|
| what | anchored implied-elo (`eval/model_elo`) | pairwise H2H round-robin over peaks |
| code | `eval_worker` + `rating.implied_elo` | `scripts/round_robin.py` + `delta_e_harness` |
| cost | cheap, already produced each chunk | ~30 CPU-min / 4-lane re-rank, **off-GPU** |
| CI | saturates ~1700, ±340 between lanes | ±62@120, ±48@200, ±34@400 games/pair |
| used for | **which lane gets the next 300s chunk** | **swap / crown / age-out decisions** |

Never swap on the anchored signal (it ties everything — the Derby v4 four-way
"tie" was the ±62 ruler judging ~15–50 deltas). Keep anchored elo only as the
in-race climb gradient for `pick_priority`.

## Contracts (lock these; components build against them)

### Daemon submit/poll API  (`scripts/gpu_daemon.py`, module-level)
```python
submit(queue: Queue, spec: dict) -> job_id: str
poll(queue: Queue, job_id: str)  -> {"state": str, "returncode": int|None,
                                     "result": {"model_elo": float|None,
                                                "wall_secs": float|None, ...}}
```
Train-chunk spec:
```json
{"kind":"train","cell":"derby-v7-control","max_wall_secs":300,"final_eval":true,
 "resume_from":"auto","tier":2,"priority":0.0,"note":"broker:control:chunk7"}
```
`result.model_elo` is ALWAYS present (None when the eval line is absent).

### Candidate pool  (`scripts/derby_pool.py`)
Dir `sweep_runs/derby_pool/candidates/<name>.json`, atomic writes:
```json
{"name":"vct-teacher","cell":"derby-x-vct","lever":"+ VCT teacher signal",
 "status":"available","submitted_by":"researcher:<session>","submitted_at":<ts>}
```
API: `register(name, cell, lever, by)` · `list(status=...)` · `claim(name)->spec`
· `retire(name, reason)`. Status: `available → running → retired`.

### Broker state  (`sweep_runs/<derby>/broker_state.json`, atomic)
```json
{"pool_size":4, "ttl_secs":5400, "peak_window":6,
 "lanes":{"<name>":{"cell":..,"wall_secs_total":..,"chunks_done":..,
                    "peak_elo":..,"peak_path":..,"climb_rate":..,
                    "chunks_since_peak":..,"last_picked":..,"status":"running|aged_out"}},
 "last_verdict":{"ts":..,"ranking":[..],"crowned":"<name>|null","escalate":bool,"reason":..},
 "needs_you": "<one line or null>"}
```

### Gate verdict API  (`scripts/derby_gate.py`)
```python
verdict(peaks: dict[name,path], games_per_pair=200, escalate_to=400) ->
  {"ranking":[(name,delo,ci_half)...], "crowned": name|None,
   "escalate": bool, "reason": str}
```
Wraps `round_robin.py`; crown only if leader's margin over #2 exceeds the combined
CI half-width (`hypot(ci_a, ci_b)`); else `escalate` the overlapping pair to
`escalate_to` games; if still overlapping, `crowned=None, escalate=True` and the
broker holds the incumbent + sets `needs_you`.

### Broker tick (one pass)
1. `poll` any in-flight chunk → fold `model_elo` into the lane's climb history.
2. Age-out pass: any lane with `wall_secs_total ≥ ttl_secs` AND
   `chunks_since_peak ≥ peak_window` → `status=aged_out`, free its slot.
3. Refill: for each free slot, `derby_pool.claim` the next candidate → new lane.
4. `pick_priority` over running lanes (entry-fee → starvation → climb-rate, the
   existing `delo_derby` logic) → choose one lane.
5. `daemon.submit` one 300s train chunk for that lane.
6. Every `verdict_period` (~1h of accrued GPU): run `derby_gate.verdict` over lane
   peaks → record `last_verdict`; crown/swap or set `needs_you`.
7. `atomic_write_json(broker_state)`; sleep; repeat forever.

## Cutover (PLAN — gated on Jason; nothing auto-executes)
1. `kill <watchdog-pid>` FIRST (it auto-resurrects the derby + bumps cap forever).
2. SIGINT the running `delo_derby` (or let the 300s slice self-cap); trainer
   force-saves a resumable `latest.pt`.
3. Verify no `delo_derby|run_sweep|selfplay_worker|gomoku.train|eval_worker`.
4. Start `gpu_daemon daemon`, then the broker. Single dispatcher confirmed by the
   flock'd `daemon.pid` and no `delo_derby` running.

See [gpu-daemon](gpu-daemon.md) for the daemon internals, [delo-derby](../../scripts/delo_derby.py)
for the `pick_priority` logic the broker reuses.
