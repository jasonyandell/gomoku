---
name: gomoku-derby-runner
description: Run the Δelo Derby as the single GPU executor — the proven 10-min cron-driven check loop. Each tick — assert health (one derby PID + watchdog), read the scoreboard (peak / chunks_since_new_peak / idle from derby_state.json), and SWAP plateaued/starved/result-locked lanes for fresh cells by judgement (never a climber). Round-robin the _peaks for the real H2H verdict when the field plateaus; record verdicts to the research board. Trigger on "derby-runner check", "keep the derby running", "run the derby loop", "scoreboard", "swap a lane", "derby v8" (or current vN), or any cron re-invocation of the derby loop. This is the HOW; gomoku-research-lab is the charter, project_derby_operating_model.md is the resume index.
---

# gomoku-derby-runner

The operational playbook for running the Δelo Derby as **the single GPU executor**.
The orchestrating session IS the "derby runner" — it owns the GPU, runs the derby in
~300s chunks doled by Δelo-rate, and keeps ~4 contestants stocked, **swapping
plateaued/starved/spent lanes for fresh cells by judgement**. This skill is the muscle
memory: the exact commands, the scoreboard script, the swap procedure, the verdict
ritual. Battle-tested over a 7h+ autonomous run (Derby v8, 2026-05-27) with two clean
verdict-driven swaps.

**Read once for context, then just run the loop:**
- `gomoku-research-lab` — the broad lab charter (two queues, receipts, stop-gates).
- `~/.claude/.../memory/project_derby_operating_model.md` — the resume index (current
  board, winner lineage, gotchas). **Read this first when picking up a live derby.**
- `wiki/topics/research-loop.md` — the researcher/gate/orchestrator roles.
- `wiki/ops/research-board.md` — the CURRENT callout + all vN verdicts.

## The operating model (Jason's, non-negotiable)

- **The derby owns the GPU.** One `delo_derby.py` process, doling ~300s chunks by
  Δelo-rate (peak-progress + patience). `cap_wall_secs` is a generous backstop, NOT a
  hard kill — the ~hourly assess+swap is the runner's job, so a climber is never cut off.
- **Beads never run the GPU.** A bead = CODE-only work for *another* session that lands
  a cell in `run_sweep.CELLS` "available for the derby." Two GPU executors collide.
  **Config-only levers (existing flags) skip beads** — the runner just adds the cell and
  races it. Reserve beads for code-heavy builds (new solver/sampler/harness).
- **"You can't pick wrong as long as you keep things moving."** Everything gets run
  eventually; a reasonable swap is always fine. Default to continuing. Keep reports tight.

## The check loop (every ~10 min, cron-driven)

Set a recurring cron (`CronCreate`) whose prompt is the derby-runner check. Pick an
off-minute cadence (e.g. `3,13,23,33,43,53 * * * *`) to dodge the :00/:30 fleet marks.
Each tick does SCAN → HEALTH → SCOREBOARD → SWAP-IF-WARRANTED → report. The cron prompt
should restate the whole procedure (so a post-compact session can run it cold) + the
rules: own-the-GPU, beads-are-code-only, never PushNotification unless the derby is dead
AND unrecoverable, next check ~10 min.

### 0. SCAN FOR NEW RESEARCH (the loop self-feeds — added 2026-05-27)

Other sessions land cells in `run_sweep.CELLS` "available for the derby" and close
`derby-idea` beads (new levers, or fixes to a crippled one). **The loop pulls these in
itself** — don't wait to be told "check main." Each tick, before health:
```bash
cd ~/code/gomoku && git fetch origin -q
git log --oneline HEAD..origin/main | head        # new commits landed?
```
- **New commits on origin/main** → integrate with **`git merge --no-ff origin/main`**
  (NEVER rebase). Other sessions add cells in their own region of `run_sweep.CELLS` and
  don't touch the board, so the merge is almost always clean & additive; if the board
  json or a cell conflicts, resolve by **keeping both** (additive). Commit the merge.
- **Find raceable cells** = a `derby-x-*` / `derby-*` cell in `run_sweep.CELLS` that is
  NOT on the current board:
  ```bash
  python -c "import sys;sys.path.insert(0,'scripts');import run_sweep,json; \
  board={x['cell'] for x in json.load(open('scripts/derby_vN_board.json'))['ideas']}; \
  print('OFF-BOARD CELLS:',[c for c in run_sweep.CELLS if c.startswith('derby') and c not in board])"
  bd list --label derby-idea 2>/dev/null | tail -20   # recently closed = a fix/lever landed
  ```
- **A new/fixed cell is a swap candidate** — race it by judgement (swap out a
  plateaued/characterized lane, §3). A cell that was a *fix* to a previously-crippled
  lane (e.g. derby-eda fixed derby-x-crossgame's O(N) ingest) → **re-race it FRESH**:
  archive the old `sweep_runs/<cell>/` (stale checkpoint + an incompatible store) so it
  starts seed-0 with the fixed code, then verify the fix held under live GPU load (the
  symptom that triggered the bead — here, epoch wall stays flat). **Verify at FULL load,
  not early:** wait until the buffer is full and gen floods (epoch ~50+, `new` large) —
  an early small-`new` reading can look flat and fool you into closing the bead too soon
  (it did, 2026-05-27: crossgame read 5s @epoch27/new=32, then settled ~30s @epoch55/
  new=848 once flooding kicked in). Only close the bead once it holds under flooding.
- **A code-heavy lever (esp. a per-move solver: VCF/VCT/defense-teacher) has TWO
  failure modes — check BOTH:** (1) slow TRAINING epochs (epoch wall grows — crossgame's
  O(store) sidecar), and (2) **generation STARVATION** — `buf=0 / games=0 / pl=nan` while
  the self-play workers are alive but stuck in an unbounded per-move solve (2026-05-27:
  `derby-x-vct` produced ZERO games in ~50s — VCT search ≫ VCF, no `--max-depth/-nodes`
  bound). So on a new solver lane's first peek, confirm **buf is FILLING** (not just that
  epoch wall is flat). If buf stays 0 → pull it, bead the fix (bound the solve), restore.
- This is read-mostly (fetch + log + grep); only the `merge` writes, and only when
  something landed. It costs ~1s/tick and keeps the board fed without a human nudge.

### 1. HEALTH (assert exactly ONE derby PID)

```bash
cd ~/code/gomoku
echo "derby=$(pgrep -f 'delo_derby.py --board scripts/derby_vN'|wc -l|tr -d ' ') \
watchdog=$(pgrep -f 'derby_watchdog.*derby_vN'|wc -l|tr -d ' ')"
```
- `derby=1 watchdog=1` → healthy. The watchdog auto-restarts a dead derby (it has a
  startup grace), so usually you do nothing.
- **Both down** → restart, then **assert ONE PID before starting the watchdog** (the
  watchdog can race-spawn a duplicate orchestrator if it checks during launch):
  ```bash
  nohup python scripts/delo_derby.py --board scripts/derby_vN_board.json --resume \
    >> "$CLAUDE_JOB_DIR/derby_vN.log" 2>&1 &
  sleep 6; pgrep -f 'delo_derby.py --board scripts/derby_vN' | wc -l   # MUST be 1
  nohup bash scripts/derby_watchdog.sh scripts/derby_vN_board.json >/dev/null 2>&1 &
  ```
- PIDs changing between checks is normal (watchdog restarted it, or you relaunched). What
  matters is **exactly one** derby PID.

### 2. SCOREBOARD (the canonical read script)

`derby_state.json`'s `elo_history` entries are `[epoch, elo, wall]` triples — use `[1]`
for elo. `chunks_since_new_peak` (csnp) = trailing history points that did NOT set a new
max; `idle` = chunks since this lane was last picked (starvation proxy).

```bash
python - <<'PY'
import json
s=json.load(open('sweep_runs/derby_vN/derby_state.json'))
tot=s.get('total_chunks_run'); print("chunks:",tot,"| updated:",s.get('updated'))
rows=[]
for n,i in s['ideas'].items():
    elos=[h[1] for h in (i.get('elo_history') or [])]
    csnp=0
    if elos:
        run=-1e9; lp=-1
        for idx,e in enumerate(elos):
            if e>run: run=e; lp=idx
        csnp=len(elos)-1-lp
    last3=[round(e) for e in elos[-3:]]
    rows.append((i.get('peak_elo') or -1,n,i.get('chunks_done'),i.get('status'),
                 csnp,tot-(i.get('last_picked') or 0),last3))
rows.sort(reverse=True)
print(f"{'lane':14}{'peak':>6}{'chk':>4}{'status':>9}{'csnp':>5}{'idle':>5}  last3")
for pk,n,ch,st,csnp,idle,l3 in rows:
    print(f"{n:14}{pk:6.0f}{str(ch):>4}{str(st):>9}{csnp:5d}{idle:5d}  {l3}")
PY
# live tenant + trainer health (plies = collapse tell; train/gen split = runaway tell):
live=$(pgrep -fl 'run_sweep.py --cell' | sed -E 's/.*--cell ([^ ]+).*/\1/' | head -1)
echo "live: ${live:-NONE}"; [ -n "$live" ] && tail -1 sweep_logs/$live/trainer.log
```

**Reading the trainer tail** (per [[feedback-self-play-eta]], [[project-light-all-engines]]):
- `plies` falling steadily + low value-loss = fast-attack collapse (bad). High/rising
  plies = defense improving (good). `plies=81` on 9×9 = full-board draws (a recency-lever
  characteristic — note it, don't panic). `plies=0.0 / new=0` = a no-new-games SGD epoch
  (transient, not collapse).
- `train=` ballooning vs `gen=` = SGD runaway / contention. `value-discount` lanes show
  very low vl (0.05–0.1) + long plies — that's the discounted-target signature, healthy.

### 3. SWAP BY JUDGEMENT (the whole point — don't agonize)

A lane is a **swap candidate** when:
- **PLATEAUED**: csnp ≥ ~8 (well past the patience window). NOTE: a *parked* lane can't
  climb to csnp 8 (it's not running) — a plateaued-AND-deprioritized lane that the engine
  won't feed is also a candidate, but **let the starvation floor force-feed it once first**
  — it often pops, and you avoid a blind swap (seen repeatedly in v8).
- **STARVED**: near-zero wall while others race many chunks (and the floor isn't feeding it).
- **ERRORED**: status error / repeated crashes.
- **RESULT-LOCKED**: a baseline/contender whose result is established (peak + H2H known)
  and re-running just re-confirms it — retire it to free GPU for a fresh question. Its
  `peak.pt` stays in `_peaks/` as a permanent round-robin anchor, so you lose nothing.

**NEVER swap a CLIMBER** (recently set a new peak / csnp low). And **never retire a
still-climbing FRESH seed-0 lane on an H2H verdict** — see the fresh-start lag below.

**The clean swap procedure** (config-only lever; ~2 min, derby resumes seamlessly):
```bash
# (a) add the new cell to run_sweep.CELLS (clone a sibling derby- cell, change ONE
#     lever: --value-discount / --buffer-recency-frac / --gumbel-m / --dirichlet-eps /
#     --global-pool / --max-plies). Validate:
python -m py_compile scripts/run_sweep.py
python -c "import sys;sys.path.insert(0,'scripts');import run_sweep;print('derby-x-NEW' in run_sweep.CELLS)"
# (b) kill the watchdog FIRST (so it can't respawn the derby mid-edit), then clean-stop:
pkill -f 'derby_watchdog.*derby_vN'
pkill -TERM -f 'delo_derby.py --board scripts/derby_vN'
pkill -TERM -f 'run_sweep.py --cell derby-'; pkill -TERM -f 'gomoku.selfplay_worker'
pkill -TERM -f 'gomoku.train .*derby-'; pkill -TERM -f 'gomoku.eval_worker'
sleep 5   # NEVER kill -9 — SIGTERM lets the trainer self-save a resumable latest.pt
# confirm 0 survivors, then:
# (c) edit scripts/derby_vN_board.json: replace the dead lane's idea entry (keep ~4).
# (d) drop the dead lane from sweep_runs/derby_vN/derby_state.json["ideas"] (its peak.pt
#     stays in _peaks/ as an anchor); --resume adds the new idea fresh from the board.
# (e) dry-run, relaunch derby --resume, ASSERT ONE PID, relaunch watchdog:
python scripts/delo_derby.py --board scripts/derby_vN_board.json --dry-run | grep IDEA
nohup python scripts/delo_derby.py --board scripts/derby_vN_board.json --resume >>"$CLAUDE_JOB_DIR/derby_vN.log" 2>&1 &
sleep 6; pgrep -f 'delo_derby.py --board scripts/derby_vN'|wc -l   # MUST be 1
nohup bash scripts/derby_watchdog.sh scripts/derby_vN_board.json >/dev/null 2>&1 &
# (f) commit + push the board + cell change (clean main fast-forward, not confirm-gated).
```
A **code-heavy** lever (new solver/sampler/harness) does NOT get built here — file a bead
(status `open`, label `derby-idea`, `external_ref=claude-session:$CLAUDE_CODE_SESSION_ID`,
description = a CODE-ONLY recipe that lands a cell, NO GPU run) for another session.

### 4. VERDICT — round-robin when the field plateaus

Anchored elo **saturates ~1700**; once 2+ lanes cluster near the ceiling it can't
separate them. When the field broadly plateaus (most lanes csnp ≥ 4), run the H2H:
```bash
P=sweep_runs/derby_vN/_peaks
nohup python scripts/round_robin.py \
  --model A=$P/A/peak.pt --model B=$P/B/peak.pt ... \
  --games 24 --sims 100 --workers 4 --device cpu \
  --out sweep_runs/derby_vN/round_robin_<NN>chunks.json >>"$CLAUDE_JOB_DIR/rr.log" 2>&1 &
```
It's CPU (4 workers keeps it light against the live chunk) + background; read the
mean-centered `ratings` next check. **Include retired lanes' peaks** to validate past
swaps. Record the verdict to `wiki/ops/research-board.md` (concise bullet) + commit/push.

**⚠ THE FRESH-START H2H LAG (the load-bearing caveat, seen 2× in v8):** a lane that
starts fresh seed-0 is **systematically undervalued** by round-robin until it matures —
its saved `peak.pt` lags its live trajectory. In v8: `disc-recency` went +16 (@52ch) →
+82 (@73ch); `vdisc-097` was −160 in RR while its anchored elo was *climbing* 1555→1620.
**Rule: judge fresh lanes on climb-RATE; judge warm-resumed lanes on peak H2H; never
retire a climbing fresh lane on an H2H number.**

## Reporting (the classifier reads only your message text)

Each tick: a one-line health (`derby=1 watchdog=1`), a compact scoreboard table sorted by
peak, any swap (what + why in one line), and a one-line vibe. Keep it tight — frequent
checks get brief commentary; expand only when something interesting happened (a swap, a
verdict, a new peak, a collapse). Narrate the swap *plan* one check ahead when you can, so
the decision is legible. Restate results in text (tool output is invisible to the reader).

## What "working" looked like (the v8 reference run, 2026-05-27)

~7h fully autonomous: 10-min cron loop; field climbed mate-discount→1699, two clean
verdict-driven swaps (`stack`/max-plies-45 → `disc-recency` after stack regressed;
`control`/spent-baseline → `vdisc-097` to probe the value-discount optimum); two
round-robins; findings = **value-discount 0.98 is the champion lever, recency is an
additive #2, the combo is competitive, max-plies hurts, 0.97 is too sharp**, plus the
fresh-start-lag methodology lesson. The derby never stopped; every swap was config-only,
committed, and pushed.
